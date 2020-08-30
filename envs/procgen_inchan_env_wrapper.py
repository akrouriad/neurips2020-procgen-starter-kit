from gym import Wrapper
from envs.procgen_env_wrapper import ProcgenEnvWrapper
import numpy as np
from ray.tune import registry


class DiffInChanWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._lastobs = None
        self._lastlastobs = None

    def reset(self):
        self._lastobs = self.env.reset()
        self._lastlastobs = None
        ret = self._lastobs.copy()
        ret[:, :, 2] = (np.sum(ret, dim=2, keepdims=True) + ret[:, :, 2]) / 4
        ret[:, :, 0:1] = 0
        return ret.astype(np.uint8)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ret = obs.copy()
        ret[:, :, 2] = (np.sum(ret, dim=2, keepdims=True) + ret[:, :, 2]) / 4
        fdiff = obs - self._lastobs
        ret[:, :, 1] = (np.sum(fdiff, dim=2, keepdims=True) + fdiff[:, :, 1]) / 4
        if self._lastlastobs is not None:
            sdiff = self._lastobs - self._lastlastobs
            sdiff = fdiff - sdiff
            ret[:, :, 0] = (np.sum(sdiff, dim=2, keepdims=True) + sdiff[:, :, 0]) / 4
        else:
            ret[:, :, 0] = 0.
        self._lastlastobs = self._lastobs
        self._lastobs = obs
        return ret.astype(np.uint8), rew, done, info


# Register Env in Ray
registry.register_env(
    "procgen_inchan_env_wrapper",
    lambda config: DiffInChanWrapper(ProcgenEnvWrapper(config))
)
