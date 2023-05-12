from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.mo_dqn import MODQN
from momarl_benchmarks.algorithms.so_dqn import DQN
from momarl_benchmarks.algorithms.mo_a2c import MOA2C

from loguru import logger

logger.stop()

# Hyperparameters
BATCH_SIZE = 256  # 128
GAMMA = 0.99
TAU = 0.001
LR =  3e-5

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



from datetime import datetime

now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

wandb_name = "MO_DQN_OneWaypointEnv_Utility_1_" + dt_string



dqn_agent = MODQN(env, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr=LR, utility_f=linear_utility_f)
dqn_agent.train(50_000, enable_wandb_logging="online", wandb_group_name="MO_DQN_OneWaypointEnv_Utility_1",
                wandb_name=wandb_name)
