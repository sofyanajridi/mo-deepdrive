from ray.rllib.env.multi_agent_env import MultiAgentEnv
from deepdrive_zero.envs.env import Deepdrive2DEnv


class MultiAgentDeepdriveParallelWrapper(MultiAgentEnv):
    """
    Wrapper that transforms the DeepDrive env to a Parallel MultiAgent env following the Ray rllib API:
    https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical
    """

    metadata = {'render_modes': ['human']}
    render_mode = 'human'

    def __init__(self, env: Deepdrive2DEnv, vehicle_1_id="vehicle1", vehicle_2_id="vehicle2", multi_objective=False):
        super().__init__()
        self.deepdrive_env = env
        self.vehicle_1_id = vehicle_1_id
        self.vehicle_2_id = vehicle_2_id
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.observation_space_dict = {vehicle_1_id: env.observation_space, vehicle_2_id: env.observation_space}
        self.action_space_dict = {vehicle_1_id: env.action_space, vehicle_2_id: env.action_space}

        if multi_objective:
            self.reward_space = env.reward_space

    def reset(self, *, seed=None, options=None):

        obs, info = {}, {}

        first_action = 0 if self.deepdrive_env.discrete_actions else [0, 0, 0]  # Depending if action space is continous or discrete

        blank_obs, blank_info = self.deepdrive_env.reset(seed=seed, options=options)
        vehicle_2_obs, vehicle_2_rew, vehicle_2_terminated, vehicle_2_truncated \
            , vehicle_2_info = self.deepdrive_env.step(first_action) # Let first action just be idle, to get back blank observation for second vehicle

        obs[self.vehicle_1_id] = blank_obs
        obs[self.vehicle_2_id] = vehicle_2_obs

        info[self.vehicle_1_id] = blank_info
        info[self.vehicle_2_id] = vehicle_2_info

        return obs, info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

        action_vehicle_1 = action_dict[self.vehicle_1_id]
        action_vehicle_2 = action_dict[self.vehicle_2_id]

        vehicle_2_obs, vehicle_2_rew, vehicle_2_terminated, vehicle_2_truncated \
            , vehicle_2_info = self.deepdrive_env.step(action_vehicle_1)

        vehicle_1_obs, vehicle_1_rew, vehicle_1_terminated, vehicle_1_truncated \
            , vehicle_1_info = self.deepdrive_env.step(action_vehicle_2)

        obs[self.vehicle_1_id] = vehicle_1_obs
        obs[self.vehicle_2_id] = vehicle_2_obs

        rew[self.vehicle_1_id] = vehicle_1_rew
        rew[self.vehicle_2_id] = vehicle_2_rew

        terminated[self.vehicle_1_id] = vehicle_1_terminated
        terminated[self.vehicle_2_id] = vehicle_2_terminated

        truncated[self.vehicle_1_id] = vehicle_1_truncated
        truncated[self.vehicle_2_id] = vehicle_2_truncated

        info[self.vehicle_1_id] = vehicle_1_info
        info[self.vehicle_2_id] = vehicle_2_info

        if terminated[self.vehicle_1_id] or terminated[self.vehicle_2_id]:
            terminated["__all__"] = True
            truncated["__all__"] = True
        else:
            terminated["__all__"] = False
            truncated["__all__"] = False

        return obs, rew, terminated, truncated, info

    def render(self):
        self.deepdrive_env.render()

    def close(self):
        self.deepdrive_env.close()
