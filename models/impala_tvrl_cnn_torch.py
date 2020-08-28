import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class LinearProfile:
    def __init__(self, y0, y1, x1):
        self._get_target = lambda x: y0 + (x / x1) * (y1 - y0)
        self._phase = 0.
        self._target = self._get_target(self._phase)

    def set_phase(self, x):
        self._phase = x
        self._target = self._get_target(self._phase)

    def get_target(self):
        return self._target


class DiscreteMinprobEntropyProjection:
    def __init__(self, nb_act, init_entrop_ratio, explore_time):
        self.entropy_profile = LinearProfile(y0=init_entrop_ratio / nb_act, y1=0., x1=explore_time)
        self.nb_act = nb_act

    def project(self, probs):
        target_entropy = self.entropy_profile.get_target()
        curr_entrop = self.entropy(probs)
        violating_states = curr_entrop < target_entropy - 1e-6
        if any(violating_states):
            prob_unif = 1. / self.nb_act
            eta = ((prob_unif - target_entropy) / (prob_unif - curr_entrop[violating_states])).unsqueeze_(1)
            probs_new = torch.zeros_like(probs)
            probs_new[violating_states] = eta * probs[violating_states] + (1 - eta) * prob_unif
            probs_new[~violating_states] = probs[~violating_states]
            return probs_new
        else:
            return probs

    @staticmethod
    def entropy(probs):
        return torch.min(probs, dim=-1)[0]


class DiscreteTsallisEntropyProjection:
    def __init__(self, nb_act, init_entrop_ratio, explore_time):
        self.entropy_profile = LinearProfile(y0=init_entrop_ratio, y1=0., x1=explore_time)
        self.nb_act = nb_act

    def project(self, probs):
        target_entropy = self.entropy_profile.get_target()
        curr_entrop = self.entropy(probs)
        violating_states = curr_entrop < target_entropy - 1e-6
        if any(violating_states):
            beta = 1 - target_entropy * (self.nb_act - 1) / self.nb_act
            a = torch.sum((probs[violating_states] - 1 / self.nb_act)**2, dim=-1)
            b = torch.mean(probs[violating_states] - 1 / self.nb_act, dim=-1)
            c = 1 / self.nb_act - beta
            delta = b**2 - a * c
            eta = ((-b + torch.sqrt(delta)) / a).unsqueeze_(1)
            new_probs = torch.zeros_like(probs)
            new_probs[violating_states] = eta * probs[violating_states] + (1-eta) / self.nb_act
            new_probs[~violating_states] = probs[~violating_states]
            return new_probs
        else:
            return probs

    def entropy(self, probs):
        return (1 - torch.sum(probs**2, dim=-1)) * self.nb_act / (self.nb_act - 1)


def init_entropy_projs(num_outputs, config):
    entropy_projs = []
    for etype in config['entropy_cst_type'].split(' '):
        if etype == 'min':
            entropy_projs.append(DiscreteMinprobEntropyProjection(num_outputs, config['entrop_min_init_ratio'],
                                                                  config['exploration_time']))
        elif etype == 'tsallis':
            entropy_projs.append(DiscreteTsallisEntropyProjection(num_outputs, config['entrop_tsallis_init_ratio'],
                                                                  config['exploration_time']))
    return entropy_projs


class ImpalaCNN(nn.Module):
    def __init__(self, obs_space, num_outputs, config):
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.temp_fc = nn.Linear(in_features=256, out_features=1)
        self.temp_mult = nn.Parameter(torch.tensor(0.))
        self.entropy_projs = init_entropy_projs(num_outputs, config)
        self._features, self._adv, self._value, self._probs = (None,) * 4

    def compute_features(self, x):
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        self._features = nn.functional.relu(x)

    def forward(self, x):
        self.compute_features(x)
        self._adv = self.logits_fc(self._features)
        self._value = self.value_fc(self._features)
        logits_unproj = self.temp_mult * (self.temp_fc(self._features) ** 2) * self._adv
        probs = torch.softmax(logits_unproj, dim=-1)
        for eproj in self.entropy_projs:
            probs = eproj.project(probs)
        self._probs = probs
        return probs

    def get_v_from_obs(self, x):
        self.compute_features(x)
        return self.value_fc(self._features)

    def get_state_dict_clone(self):
        w = self.state_dict()
        for k, v in w.items():
            w[k] = v.clone().detach()
        return w

    def soft_weight_set(self, a_dic, b_dic, mix):
        model_dic = self.state_dict()
        for k, v in a_dic.items():
            model_dic[k].copy_(mix * v + (1 - mix) * b_dic[k])
