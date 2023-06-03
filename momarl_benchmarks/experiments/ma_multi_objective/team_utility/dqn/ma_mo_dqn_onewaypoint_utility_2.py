from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv, IntersectionWithGsAllowDecelEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2

from momarl_benchmarks.algorithms.mo_dqn import MODQN,multi_agent_train_MODQN

from loguru import logger

from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper

logger.stop()

BATCH_SIZE = 128  # 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-4

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
    gforce_threshold=1.0,
    end_on_lane_violation=True,
    lane_margin=0.2,
    expect_normalized_action_deltas=False,
    jerk_threshold=150.0,  # 15g/s
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
    multi_objective=True,
)


env = MultiAgentDeepdriveParallelWrapper(
    IntersectionWithGsAllowDecelEnv(env_configuration=env_config, render_mode=None), multi_objective=True)





def utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    if distance_reward == 0:
        return -10
    elif distance_reward > 2:
        return (1 * distance_reward) + (win_reward) - (4 * collision_penalty) - pow((0.03 * gforce),2) - pow((3.3e-5 * jerk),2) - (0.06 * lane_penalty)
    else:
        return (0.50 * distance_reward) + (win_reward) - (4 * collision_penalty) -  pow((0.03 * gforce),4) -  pow((0.03 * gforce),4) - (0.06 * lane_penalty)



wandb_name = "MA_MO_DQN_OneWaypointEnv_Utility_2"

vehicle_1_dqn_agent = MODQN(env, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr=LR, utility_f=utility_f)
vehicle_2_dqn_agent = MODQN(env, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr=LR, utility_f=utility_f)

multi_agent_train_MODQN(vehicle_1_dqn_agent, vehicle_2_dqn_agent, env, 100_000, enable_wandb_logging="online",
                  wandb_group_name="MA_MO_DQN_OneWaypointEnv_Utility_2", wandb_name=wandb_name,config=env_config)
