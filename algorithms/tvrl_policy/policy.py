#!/usr/bin/env python

from ray.rllib.policy import Policy
from models.impala_tvrl_cnn_torch import ImpalaCNN, LinearProfile
from models.tvrl_mlp_torch import MLPPolicy
import numpy as np
import time
import torch


class CstAvgTV:
    def __init__(self, tv_max):
        self.tv_max = tv_max

    def __call__(self, p, q):
        return .5 * torch.mean(torch.sum(torch.abs(p - q), dim=1)) - self.tv_max


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        if batch_start + 2 * batch_size > data_set_size:
            yield batch_idx_list[batch_start:data_set_size]
            break
        else:
            yield batch_idx_list[batch_start:batch_start + batch_size]


class TVRLPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.device = torch.device(config['device'])
        self.model = ImpalaCNN(observation_space, action_space.n, config).to(self.device)
        # self.model = MLPPolicy(observation_space, action_space.n, config).to(self.device)
        self.lr = config['lr']
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.discount = config['gamma']
        self.lambda_profile = LinearProfile(config['lambda'], config['lambda_final'], x1=1.)
        self.tv_max = config['tv_max']
        self.cst_fct = CstAvgTV(self.tv_max)
        self.rwd_scale = 1.
        self.lr_scaling = 1.
        self.nb_learning_samples = 0.
        self.timesteps_total = config['timesteps_total']
        self.phase = 0
        self.sgd_minibatch_size = config['sgd_minibatch_size']
        self.lossf_v = torch.nn.L1Loss()
        self.nb_epochs = config['nb_sgd_epochs']
        self.soft_stepsize = 1

    def get_targets(self, v_values, v_values_next, rwd, last_from_ep):
        gen_adv = np.zeros_like(v_values)
        k = len(gen_adv) - 1
        lam = self.lambda_profile.get_target()
        for v, vn, r, new in zip(reversed(v_values), reversed(v_values_next), reversed(rwd), reversed(last_from_ep)):
            if new:
                gen_adv[k] = r + self.discount * vn - v
            else:
                gen_adv[k] = r + self.discount * vn - v + self.discount * lam * gen_adv[k + 1]
            k -= 1
        return gen_adv + v_values, gen_adv

    def get_v_vals_next(self, samples):
        v_vals_next = np.zeros_like(samples['v_vals'])
        v_vals_next[:-1] = samples['v_vals'][1:]  # most of the time, v_vals next is v_vals of next obs in dataset
        v_vals_next[samples['dones']] = 0.  # unless it is a terminal state

        last_from_ep = np.array([True] * len(samples['rewards']))  # check when traj segments switch to new ep
        last_from_ep[:-1] = (samples['eps_id'][:-1] - samples['eps_id'][1:]) != 0  # which is true when next eps id and current are not the same
        compute_v_next = last_from_ep & ~samples['dones']  # compute v_vals in these states using new_obs, unless it is a terminal state
        with torch.no_grad():
            v_vals_next[compute_v_next] = self.model.get_v_from_obs(torch.tensor(samples['new_obs'][compute_v_next], device=self.device)).cpu().numpy().squeeze(1)
        return v_vals_next, last_from_ep

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        with torch.no_grad():
            self.model.forward(torch.tensor(np.stack(obs_batch), device=self.device))
            action_batch = torch.distributions.Categorical(probs=self.model._probs).sample().cpu().numpy()
            info = {'v_vals': self.model._value.clone().squeeze_(1).cpu().numpy(), 'probs': self.model._probs.clone().cpu().numpy()}

        return action_batch, [], info

    def learn_on_batch(self, samples):
        # implement your learning code here
        with torch.no_grad():
            # computing v_vals of next state
            v_vals_next, last_from_ep = self.get_v_vals_next(samples)

            # scaling rewards
            # old_rwd_scale = self.rwd_scale
            rwd_scale_lr = .3
            # self.rwd_scale = rwd_scale_lr * np.std(samples['rewards']) + (1 - rwd_scale_lr) * self.rwd_scale
            # self.rwd_scale = max(np.std(samples['v_vals']), 1e-3)
            # print(self.rwd_scale)
            # rwd = samples['rewards'] / self.rwd_scale
            # v_vals = samples['v_vals'] * old_rwd_scale / self.rwd_scale
            # v_vals_next *= old_rwd_scale / self.rwd_scale

            # computing targets
            # v_targ, a_targ = self.get_targets(v_vals, v_vals_next, rwd, last_from_ep)
            # v_targ, a_targ = self.get_targets(samples['v_vals'], v_vals_next, samples['rewards'] / max(self.rwd_scale, 1e-2), last_from_ep)
            v_targ, a_targ = self.get_targets(samples['v_vals'], v_vals_next, samples['rewards'], last_from_ep)
            # self.rwd_scale = rwd_scale_lr * np.std(v_targ) + (1 - rwd_scale_lr) * self.rwd_scale
            self.rwd_scale = 1.
            print(self.rwd_scale)

            # updating number of samples and entropy profiles before update
            nb_samples = len(samples['rewards'])
            self.nb_learning_samples += nb_samples
            self.phase = self.nb_learning_samples / self.timesteps_total
            for eproj in self.model.entropy_projs:
                eproj.entropy_profile.set_phase(self.phase)
            self.lambda_profile.set_phase(self.phase)


        obs = torch.tensor(samples['obs'], device=self.device)
        act = torch.tensor(samples['actions'], dtype=torch.long, device=self.device).unsqueeze(1)
        old_probs = torch.tensor(samples['probs'], device=self.device)
        v_targ = torch.tensor(v_targ, device=self.device).unsqueeze(1)
        a_targ = torch.tensor(a_targ, device=self.device).unsqueeze(1)

        def tv_proj_probs(p, q, scale, cst_fc):
            hx = cst_fc(p, q)
            if hx > 0:
                eta = 1 / (1 + hx / scale)
                return eta * p + (1 - eta) * q, eta, hx
            else:
                return p, torch.tensor(1., device=self.device), hx

        init_params = self.model.get_state_dict_clone()


        for epoch in range(self.nb_epochs):
            for batch_idx in next_batch_idx(self.sgd_minibatch_size, len(samples['rewards'])):
                self.optim.zero_grad()
                probs = self.model.forward(obs[batch_idx])
                # clipped_probs = torch.min(torch.max(probs, old_probs[batch_idx] - self.tv_max), old_probs[batch_idx] + self.tv_max)
                # loss_p_proj = torch.min(torch.sum(clipped_probs * self.model._adv.detach(), dim=1), torch.sum(probs * self.model._adv.detach(), dim=1))
                projected_probs, eta, tv_cst = tv_proj_probs(probs, old_probs[batch_idx], self.cst_fct.tv_max, self.cst_fct)
                # loss_p_proj = torch.sum(projected_probs * self.model._adv.detach(), dim=1)
                loss_p_proj = torch.sum(probs * self.model._adv.detach(), dim=1)
                # loss_p_proj = torch.min(torch.sum(projected_probs * self.model._adv.detach(), dim=1), torch.sum(probs * self.model._adv.detach(), dim=1))
                # loss_p_unproj = torch.sum(self.model._probs * self.model._adv.detach(), dim=1)
                # loss_p = -torch.mean(torch.min(loss_p_proj, loss_p_unproj)) / eta.detach()
                loss_p = (-torch.mean(loss_p_proj) / self.rwd_scale + max(tv_cst, 0) / self.lr_scaling)
                loss_a = self.lossf_v(self.model._adv.gather(dim=1, index=act[batch_idx]), a_targ[batch_idx]) / self.rwd_scale
                loss_v = self.lossf_v(self.model._value, v_targ[batch_idx]) / self.rwd_scale
                (loss_p + loss_v + loss_a).backward()
                # loss_p = -torch.mean(projected_probs.gather(dim=1, index=act[batch_idx]) * a_targ[batch_idx] / old_probs[batch_idx]) #/ eta.detach()
                # loss_v = self.lossf_v(self.model._value, v_targ[batch_idx]) #/ eta.detach()
                # (loss_p + loss_v).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
                self.optim.step()

        ls_idx = np.random.choice(len(samples['rewards']), min(len(samples['rewards']), 500), replace=False)
        with torch.no_grad():
            self.model.forward(obs[ls_idx])
            adv_values_new = self.model._adv.clone()

        def loss_fc_ls(model):
            new_probs = model.forward(obs[ls_idx])
            return -torch.mean(torch.sum(new_probs * adv_values_new, dim=1))

        def cst_fct_model(model):
            new_probs = model.forward(obs[ls_idx])
            return self.cst_fct(new_probs, old_probs[ls_idx])

        # check constraint and do line search eventually
        with torch.no_grad():
            best_stepsize = 1.
            best_avg_tv_val = np.inf
            best_loss = np.inf
            lower_bound = 0.
            upper_bound = 2.
            upper_init = False
            optim_param = self.model.get_state_dict_clone()
            found_valid = False
            for _ in range(10):
                stepsize = (upper_bound + lower_bound) / 2
                self.model.soft_weight_set(optim_param, init_params, stepsize)
                avg_tv_val = cst_fct_model(self.model) + self.tv_max
                if avg_tv_val <= 1.5 * self.tv_max:
                    found_valid = True
                    new_loss = loss_fc_ls(self.model)
                    if new_loss <= best_loss:
                        best_loss = new_loss
                        best_stepsize = stepsize
                        best_avg_tv_val = avg_tv_val
                    # if not upper_init:
                    #     upper_bound *= 2
                    lower_bound = stepsize
                else:
                    upper_bound = stepsize
                    upper_init = True
            if found_valid:
                self.model.soft_weight_set(optim_param, init_params, best_stepsize)
            else:
                best_stepsize = 0.
                self.model.load_state_dict(init_params)

        self.soft_stepsize += .5 * (best_stepsize - self.soft_stepsize)
        if self.soft_stepsize < .85:
            self.lr_scaling *= .7
        elif best_avg_tv_val < .9 * self.tv_max:
            # self.lr_scaling *= 1.1
            self.lr_scaling *= 1.5
        for pg in self.optim.param_groups:
            pg['lr'] = self.lr * min(max(self.lr_scaling, 1e-3), 1.)

        print('sz {:3.2f} lrs {} lr {:3.6f} tv {:3.2f}; target_lam {:3.2f} target_entropy {}; ts {}'.
              format(best_stepsize, self.lr_scaling, self.optim.param_groups[0]['lr'], best_avg_tv_val, self.lambda_profile.get_target(),
                     [eproj.entropy_profile.get_target() for eproj in self.model.entropy_projs], self.nb_learning_samples))
        return {}

    def get_weights(self):
        return {"model_state_dict": self.model.state_dict(), "phase": self.phase}

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model_state_dict'])
        self.phase = weights['phase']
        for eproj in self.model.entropy_projs:
            eproj.entropy_profile.set_phase(self.phase)
        self.lambda_profile.set_phase(self.phase)


