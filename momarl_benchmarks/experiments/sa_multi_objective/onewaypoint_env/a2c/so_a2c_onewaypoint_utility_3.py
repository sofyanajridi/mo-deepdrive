from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_a2c import A2C

from loguru import logger
import torch

logger.stop()

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


def linear_utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    return (0.50 * distance_reward) + (win_reward) - (0.03 * gforce) - (3.3e-5 * jerk)


def utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    if distance_reward > 0:
        return (0.50 * distance_reward) + (win_reward)
    else:
        return (0.50 * distance_reward) + (win_reward) - (0.03 * gforce) - (3.3e-5 * jerk)

def utility(vec):
    speed_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec
    penalty = (0.50 * speed_reward ) + (100 * win_reward) - (0.006 * 5 * (10 * gforce) - (3.3e-5 * (10 * gforce)))
    return (0.50 * speed_reward ) + (win_reward) - pow((0.03 * gforce),2) - pow((3.3e-5 * jerk),3)


# dqn_agent = MODQN(env, BATCH_SIZE, GAMMA, TAU, LR, utility_f)
# dqn_agent.train(100, enable_wandb_logging="disabled", wandb_group_name=None, wandb_name=None)


from datetime import datetime

now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

wandb_name = "A2C_OneWaypointEnv_Utility_3_" + dt_string

a2c_agent = A2C(env, LR, GAMMA, utility,vectorial_reward=True)
a2c_agent.train(10000, enable_wandb_logging="online", wandb_group_name="SO_A2C_OneWaypointEnv_Utility_3", wandb_name=wandb_name)