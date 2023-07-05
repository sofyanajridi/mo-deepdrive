from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv,StaticObstacleEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_a2c import A2C
from momarl_benchmarks.algorithms.mo_a2c import MOA2C

import gymnasium as gym
from gymnasium.wrappers import TransformReward

from loguru import logger
import torch
import numpy as np
# logger.stop()

def rewardTo2D(r):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = r


    return [distance_reward,win_reward,collision_penalty,gforce,jerk]





# Hyperparameters
BATCH_SIZE = 128  # 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

env_config = dict(
    env_name='deepdrive-2d-staticobstacle',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    add_static_obstacle=True,
    gforce_threshold=1.0,
    expect_normalized_action_deltas=False,
    jerk_threshold=150.0,  # 15g/s
    incent_win=True,
    constrain_controls=False,
    physics_steps_per_observation=12,
    multi_objective=True,

)
env = StaticObstacleEnv(env_configuration=env_config, render_mode=None)

env.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
env = TransformReward(env,rewardTo2D)


def linear_utility_f(vec):
    distance_reward,win_reward,collision_penalty,gforce,jerk = vec

    return (0.30 * distance_reward) + (win_reward) - collision_penalty - (0.03 * gforce) - (3.3e-5 * jerk)



wandb_name = "MO_A2C_StaticOnstacleEnv_Utility_1"

a2c_agent = MOA2C(env, LR, GAMMA, linear_utility_f)
a2c_agent.train(20_000, enable_wandb_logging="disabled", wandb_group_name="MO_A2C_StaticOnstacleEnv_Utility_1", wandb_name=wandb_name)