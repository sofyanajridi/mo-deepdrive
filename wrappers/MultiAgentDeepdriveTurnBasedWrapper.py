import gymnasium as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from deepdrive_zero.envs.env import Deepdrive2DEnv


class MultiAgentDeepdriveTurnBasedWrapper(MultiAgentEnv):
    """
    Wrapper that transforms the DeepDrive env to a TurnBased MultiAgent env following the Ray rllib API:
    https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical

    """
    metadata = {'render_modes': ['human']}
    render_mode = 'human'

    def __init__(self, env: Deepdrive2DEnv):
        super().__init__()
        self.deepdrive_env = env
        self.vehicle_1_id = "vehicle1"
        self.vehicle_2_id = "vehicle2"
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.current_vehicle = self.vehicle_1_id

    def reset(self, *, seed=None, options=None):
        obs, info = {}, {}
        self.current_vehicle = self.vehicle_1_id
        vehicle_1_obs, vehicle_1_info = self.deepdrive_env.reset(seed=seed, options=options)
        obs[self.vehicle_1_id] = vehicle_1_obs
        info[self.vehicle_1_id] = vehicle_1_info
        return obs, info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

        action_vehicle = action_dict[self.current_vehicle]

        vehicle_obs, vehicle_rew, vehicle_terminated, vehicle_truncated \
            , vehicle_info = self.deepdrive_env.step(action_vehicle)

        rew[self.current_vehicle] = vehicle_rew
        terminated[self.current_vehicle] = vehicle_terminated
        truncated[self.current_vehicle] = vehicle_truncated
        # info[self.current_vehicle] = vehicle_info

        if terminated[self.current_vehicle]:
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        if self.current_vehicle == self.vehicle_1_id:
            self.current_vehicle = self.vehicle_2_id
        else:
            self.current_vehicle = self.vehicle_1_id

        obs[self.current_vehicle] = vehicle_obs
        info[self.current_vehicle] = vehicle_info

        return obs, rew, terminated, truncated, info

    def render(self):
        self.deepdrive_env.render()
