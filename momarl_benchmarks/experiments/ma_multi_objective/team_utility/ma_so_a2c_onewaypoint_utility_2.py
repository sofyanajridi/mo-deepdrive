from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv, IntersectionWithGsAllowDecelEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_a2c import A2C, multi_agent_train
from momarl_benchmarks.algorithms.mo_a2c import MOA2C
import torch
from loguru import logger
from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper
import torch
logger.stop()

# Hyperparameters
BATCH_SIZE = 128  # 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True,
    jerk_threshold=150.0,  # 15g/s
    incent_yield_to_oncoming_traffic=True,
)




env = MultiAgentDeepdriveParallelWrapper(
    IntersectionWithGsAllowDecelEnv(env_configuration=env_config, render_mode=None))


def linear_utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    return (0.50 * distance_reward) + (10 * win_reward) - (4 * collision_penalty) - (0.03 * gforce) - (0.10 * jerk) - (
                0.02 * lane_penalty)


def utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    if distance_reward > 0:
        return (0.50 * distance_reward) + (10 * win_reward) - (4 * collision_penalty) - pow((0.3 * gforce), 2) - pow((0.1 * jerk), 2) - (
                0.02 * lane_penalty)
    else:
        return pow((0.50 * distance_reward),2) + (10 * win_reward) - (4 * collision_penalty)


from datetime import datetime

now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

wandb_name = "MA_SO_A2C_OneWaypointEnv_Utility_2_" + dt_string

vehicle_1_a2c_agent = A2C(env, LR, GAMMA, utility_f, vectorial_reward=True)
vehicle_2_a2c_agent = A2C(env, LR, GAMMA, utility_f,vectorial_reward=True)

multi_agent_train(vehicle_1_a2c_agent, vehicle_2_a2c_agent, env, 20_000, enable_wandb_logging="online",
                  wandb_group_name="MA_SO_A2C_OneWaypointEnv_Utility_2", wandb_name=wandb_name)
