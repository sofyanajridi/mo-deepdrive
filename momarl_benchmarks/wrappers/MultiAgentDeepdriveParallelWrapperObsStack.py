import gymnasium as gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from deepdrive_2d.deepdrive_zero.envs.env import Deepdrive2DEnv
from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper


class MultiAgentDeepdriveParallelWrapperWithObsStack(MultiAgentDeepdriveParallelWrapper):

    def __init__(self, env: Deepdrive2DEnv, vehicle_1_id="vehicle1", vehicle_2_id="vehicle2", multi_objective=False):
        shape = env.observation_space.shape[0] * 3
        env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float32)
        super().__init__(env, vehicle_1_id="vehicle1", vehicle_2_id="vehicle2", multi_objective=False)

        self.observation_space_dict = {vehicle_1_id: env.observation_space,
                                                                     vehicle_2_id: env.observation_space}

        self.vehicle1_prev_prev_stack = []
        self.vehicle1_prev_stack = []
        self.vehicle1_curr_stack = []

        self.vehicle2_prev_prev_stack = []
        self.vehicle2_prev_stack = []
        self.vehicle2_curr_stack = []

    def reset(self, *, seed=None, options=None):
        obs, info = MultiAgentDeepdriveParallelWrapper.reset(self, seed=None, options=None)

        self.vehicle1_prev_prev_stack = np.zeros(29)
        self.vehicle1_prev_stack = np.zeros(29)
        self.vehicle1_curr_stack = obs[self.vehicle_1_id]

        self.vehicle2_prev_prev_stack = np.zeros(29)
        self.vehicle2_prev_stack = np.zeros(29)
        self.vehicle2_curr_stack = obs[self.vehicle_2_id]

        obs[self.vehicle_1_id] = np.concatenate(
            (self.vehicle1_curr_stack, self.vehicle1_prev_stack, self.vehicle1_prev_prev_stack))
        obs[self.vehicle_2_id] = np.concatenate(
            (self.vehicle2_curr_stack, self.vehicle2_prev_stack, self.vehicle2_prev_prev_stack))

        return obs, info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = MultiAgentDeepdriveParallelWrapper.step(self, action_dict)

        self.vehicle1_prev_prev_stack = self.vehicle1_prev_stack
        self.vehicle1_prev_stack = self.vehicle1_curr_stack
        self.vehicle1_curr_stack = obs[self.vehicle_1_id]

        self.vehicle2_prev_prev_stack = self.vehicle1_prev_stack
        self.vehicle2_prev_stack = self.vehicle1_curr_stack
        self.vehicle2_curr_stack = obs[self.vehicle_2_id]

        obs[self.vehicle_1_id] = np.concatenate(
            (self.vehicle1_curr_stack, self.vehicle1_prev_stack, self.vehicle1_prev_prev_stack))
        obs[self.vehicle_2_id] = np.concatenate(
            (self.vehicle2_curr_stack, self.vehicle2_prev_stack, self.vehicle2_prev_prev_stack))


        print(obs)
        return obs, rew, terminated, truncated, info
