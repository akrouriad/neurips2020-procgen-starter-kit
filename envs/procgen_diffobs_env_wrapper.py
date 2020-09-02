from gym import Wrapper
from envs.procgen_env_wrapper import ProcgenEnvWrapper
import numpy as np
from ray.tune import registry


class DiffObsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._scale = 25
        self._lastobs = None

    def reset(self):
        self._lastobs = self.env.reset()
        return (self._lastobs / self._scale).astype(np.uint8)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._done = done
        diffobs = ((obs / self._scale + (obs - self._lastobs) + 255) / (2 + 1 / self._scale + 1e-3)).astype(np.uint8)
        self._lastobs = obs
        return diffobs, rew, done, info


# Register Env in Ray
registry.register_env(
    "procgen_diffobs_env_wrapper",
    lambda config: DiffObsWrapper(ProcgenEnvWrapper(config))
)
