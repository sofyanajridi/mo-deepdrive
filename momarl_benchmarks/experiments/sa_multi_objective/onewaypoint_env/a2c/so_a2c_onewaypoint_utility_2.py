from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_a2c import A2C
import gymnasium as gym
from gymnasium.wrappers import TransformReward

from loguru import logger
import numpy as np
import torch
logger.stop()

def rewardTo2D(r):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = r


    return [distance_reward,win_reward,gforce,jerk]





# Hyperparameters
BATCH_SIZE = 128  # 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True,
)

env = OneWaypointEnv(env_configuration=env_config, render_mode=None)

env.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
env = TransformReward(env,rewardTo2D)




def utility_f(vec):
    distance_reward,win_reward,gforce,jerk = vec

    if distance_reward > 2:
        return  distance_reward + (0.4 * win_reward) - (0.1 * gforce) - (0.1 * jerk)
    else:
        return (0.50 * distance_reward) + (win_reward) - pow((0.1 * gforce), 2) - pow((0.1 * jerk), 2)



wandb_name = "A2C_OneWaypointEnv_Utility_2_"

a2c_agent = A2C(env, LR, GAMMA, utility_f,vectorial_reward=True)
a2c_agent.train(5000, enable_wandb_logging="online", wandb_group_name="SO_A2C_OneWaypointEnv_Utility_3", wandb_name=wandb_name)