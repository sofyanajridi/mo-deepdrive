from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_dqn import DQN
from momarl_benchmarks.algorithms.mo_a2c import MOA2C
import gymnasium as gym
from gymnasium.wrappers import TransformReward
import numpy as np

from loguru import logger
#
logger.stop()

import torch

# Hyperparameters
# Hyperparameters
BATCH_SIZE = 256  # 128
GAMMA = 0.99
TAU = 0.001
LR =  1e-4

env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True,
)


def rewardToD(r):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = r


    return [distance_reward, win_reward, gforce, jerk]



env = OneWaypointEnv(env_configuration=env_config, render_mode=None)
env.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
env = TransformReward(env,rewardToD)


def utility_f(vec):
    distance_reward, win_reward, gforce, jerk = vec

    if distance_reward > 0:
        return  distance_reward + (win_reward) - pow((0.1 * gforce), 2) - pow((0.1 * jerk), 2)
    else:
        return (0.50 * distance_reward) + (win_reward)



wandb_name = "SO_DQN_OneWaypointEnv_Utility_2"

dqn_agent = DQN(env, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr=LR, utility_f=utility_f, vectorial_reward=True)
dqn_agent.train(50000, enable_wandb_logging="online", wandb_group_name="SO_DQN_OneWaypointEnv_Utility_2",
                wandb_name=wandb_name)
