import matplotlib.pyplot as plt
import torch
import numpy as np
from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from momarl_benchmarks.algorithms.in_progress.mo_dqn_without_state_cond import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from loguru import logger
#
logger.stop()



# Hyperparameters
BATCH_SIZE = 256  # 128
GAMMA = 0.99
TAU = 0.001
LR =  3e-5

config = {
"batch_size" : BATCH_SIZE,
"gamma" : GAMMA,
"tau" : TAU,
"learning_rate" : LR,
"utility_function" : "  if distance_reward > 0: return (0.50 * distance_reward) + (win_reward) else: return (0.50 * distance_reward ) + (win_reward) - (0.03 * gforce) - (3.3e-5 * jerk)"
}

env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True
)


import wandb
wandb.init(project="momarl-benchmarks-test",name="MO_DQN_DeepDrive_OneWayPoint_w/o_state_cond", config=config, group="MO_DQN_DeepDrive_OneWayPoint_w/o_state_cond")

env = OneWaypointEnv(env_configuration=env_config, render_mode=None)



# def utility_f(vec):
#     speed_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec
#     # aggresive_penalty = np.power(((3.3e-5 *jerk) + (0.03 * gforce)),3)
#     return (0.50 * speed_reward ) + (win_reward) - (np.power((0.03 * gforce),2) - np.power((3.3e-5 * jerk),3))

def utility_f(vec):
    distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec

    if distance_reward > 0:
        return  distance_reward + (win_reward) - pow((0.1 * gforce), 2) - pow((0.1 * jerk), 2)
    else:
        return (0.50 * distance_reward) + (win_reward)


dqn_agent = DQNAgent(env, BATCH_SIZE, GAMMA, TAU, LR, utility_f)

STATS_EVERY = 1
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
for episode in range(50_000):
    # Initialize the environment and get it's state
    done = False
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    accrued_reward = np.array([0, 0, 0, 0, 0, 0])
    episode_reward = 0

    while not done:
        action = dqn_agent.select_action(obs, accrued_reward)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        speed_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = reward
        episode_reward += utility_f(reward)
        accrued_reward = accrued_reward + reward
        accrued_reward_tensor = torch.tensor(accrued_reward, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = terminated
        done = torch.tensor([done], device=device)

        dqn_agent.memory.push(obs, action, next_obs, reward, accrued_reward_tensor, done)

        obs = next_obs

        dqn_agent.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        dqn_agent.target_nn_soft_weights_update()

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        wandb.log({'ep': episode, 'avg_reward': average_reward})
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}')

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()
