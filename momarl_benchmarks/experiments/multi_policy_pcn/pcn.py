from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from loguru import logger
# logger.stop()

import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pcn.pcn import PCN


import gymnasium as gym
from gymnasium.wrappers import TransformReward

def rewardTo2D(r):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = r



    return [0.50 * distance_reward,win_reward,- 0.03 * gforce,- 3.3e-5 * jerk]



def main():
    env_config = dict(
        env_name='deepdrive-2d-onewaypoint',
        is_intersection_map=False,
        discrete_actions=COMFORTABLE_ACTIONS2,
        add_static_obstacle=False,
        gforce_threshold=1.0,
        expect_normalized_action_deltas=False,
        jerk_threshold=150.0,  # 15g/s
        incent_win=True,
        constrain_controls=False,
        physics_steps_per_observation=12,
        multi_objective=True,
        deterministic=True,
        win_coefficient=1,
    )

    def make_env():
        env = OneWaypointEnv(env_configuration=env_config, render_mode=None)
        env.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        env = TransformReward(env,rewardTo2D)
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1,1,1,1]),
        learning_rate=1e-4,
        batch_size=256,
        project_name="MORL-Baselines",
        experiment_name="PCN",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(1e6),
        ref_point=np.array([0, 0,0, 0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([20, 1, 0,0]),
        num_step_episodes=100
    )


if __name__ == "__main__":
    main()


