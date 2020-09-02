import torch
import torch.nn as nn
from models.impala_tvrl_cnn_torch import init_entropy_projs


class MLP(nn.Module):
    def __init__(self, size_list, activation_list=None, preproc=None, postproc=None):
        super().__init__()
        if activation_list is None:
            activation_list = []
            for k in range(len(size_list) - 2):
                activation_list.append(torch.nn.ReLU())
            activation_list.append(None)

        layers = []
        if preproc is not None:
            layers.append(preproc)

        for k, kp, activ in zip(size_list[:-1], size_list[1:], activation_list):
            layers.append(nn.Linear(k, kp))
            if activ is not None:
                nn.init.xavier_normal_(layers[-1].weight, nn.init.calculate_gain(activ._get_name().lower()))
                layers.append(activ)
            else:
                nn.init.xavier_normal_(layers[-1].weight)

        if postproc is not None:
            layers.append(postproc)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, num_outputs, config):
        nn.Module.__init__(self)
        self.model = MLP([obs_space.shape[0]] + 2 * [64])  # replace with config at some point
        self.logits_fc = nn.Linear(in_features=64, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=64, out_features=1)
        self.temp_fc = nn.Linear(in_features=64, out_features=1)
        self.temp_mult = nn.Parameter(torch.tensor(0.))
        self.entropy_projs = init_entropy_projs(num_outputs, config)
        self._features, self._adv, self._value, self._probs = (None,) * 4

    def compute_features(self, x):
        self._features = self.model.forward(x)

    def forward(self, x):
        self.compute_features(x)
        self._adv = self.logits_fc(self._features)
        self._value = self.get_v_from_features()
        # logits_unproj = self.temp_mult * (self.temp_fc(self._features) ** 2) * self._adv
        logits_unproj = torch.exp(self.temp_mult) * self._adv
        probs = torch.softmax(logits_unproj, dim=-1)
        for eproj in self.entropy_projs:
            probs = eproj.project(probs)
        self._probs = probs
        return probs

    def get_v_from_features(self):
        return self.value_fc(self._features)

    def get_v_from_obs(self, x):
        self.compute_features(x)
        return self.get_v_from_features()

    def get_state_dict_clone(self):
        w = self.state_dict()
        for k, v in w.items():
            w[k] = v.clone().detach()
        return w

    def soft_weight_set(self, a_dic, b_dic, mix):
        model_dic = self.state_dict()
        for k, v in a_dic.items():
            model_dic[k].copy_(mix * v + (1 - mix) * b_dic[k])
