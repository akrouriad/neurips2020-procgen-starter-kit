from gym import Wrapper
from envs.procgen_env_wrapper import ProcgenEnvWrapper
import numpy as np
from ray.tune import registry


class DiffMaxChanWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._lastobs = None
        self._lastlastobs = None

    def reset(self):
        self._lastobs = self.env.reset()
        self._lastlastobs = None
        ret = self._lastobs.copy()
        ret[:, :, 2] = np.mean(ret, axis=2)
        ret[:, :, 0:1] = 0
        return ret.astype(np.uint8)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ret = obs.copy()
        ret[:, :, 2] = np.mean(ret, axis=2)
        fdiff = obs - self._lastobs
        ret[:, :, 1] = (np.max(fdiff, axis=2) + 255) / 2
        if self._lastlastobs is not None:
            sdiff = self._lastobs - self._lastlastobs
            sdiff = fdiff - sdiff
            ret[:, :, 0] = (np.max(sdiff, axis=2) + 255) / 2
        else:
            ret[:, :, 0] = 0.
        self._lastlastobs = self._lastobs
        self._lastobs = obs
        return ret.astype(np.uint8), rew, done, info


# Register Env in Ray
registry.register_env(
    "procgen_maxchan_env_wrapper",
    lambda config: DiffMaxChanWrapper(ProcgenEnvWrapper(config))
)
