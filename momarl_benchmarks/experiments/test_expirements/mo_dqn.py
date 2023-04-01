import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from loguru import logger
logger.stop()



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','accrued_reward','done'))


class ReplayMemory(object):
    """
    Experience relay memory

    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    """
    DQN network

    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)






# Hyperparameters
BATCH_SIZE = 128 #128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Environment

env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True
)


env = OneWaypointEnv(env_configuration=env_config, render_mode=None)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
n_observations = env.observation_space.shape[0]

n_objectives = env.reward_space.shape[0]

policy_net = DQN(n_observations + n_objectives, n_actions * n_objectives).to(device)
target_net = DQN(n_observations + n_objectives, n_actions * n_objectives).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)





def utility_f(vec):
    speed_reward, win_reward, gforce_penalty, collision_penalty, jerk_penalty, lane_penalty = vec

    return  np.log(1 + (0.50 * speed_reward) + win_reward - (0.006 * 5 * gforce_penalty) - (4 * collision_penalty) - (
                3.3e-5 * jerk_penalty) - (0.02 * lane_penalty))

def calculate_utilities_batch(total_rewards):
    total_rewards = total_rewards.tolist()
    utilities = torch.tensor([[utility_f(j) for j in i]for i in total_rewards],dtype=torch.float32, device=device)
    return utilities

def calculate_utilities(total_reward):
    total_reward = total_reward.tolist()
    utilities = torch.tensor([utility_f(i) for i in total_reward],dtype=torch.float32, device=device)

    return utilities








steps_done = 0

def select_action(state,accrued_reward):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            accrued_reward = torch.tensor(accrued_reward,dtype=torch.float32)
            augmented_state = torch.cat((state,accrued_reward.unsqueeze(0)),dim=1)
            multi_obj_state_action_value = policy_net(augmented_state).view(n_actions,n_objectives)
            total_reward = multi_obj_state_action_value + accrued_reward
            utility = calculate_utilities(total_reward)
            best_action = torch.argmax(utility).unsqueeze(0).unsqueeze(0)
            return best_action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    accrued_reward_batch = torch.cat(batch.accrued_reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    augmented_states_batch = torch.concatenate((state_batch,accrued_reward_batch),dim=1)
    multi_objective_state_values = policy_net(augmented_states_batch).view(-1,n_actions,n_objectives)
    multi_objective_state_action_values = multi_objective_state_values[torch.arange(BATCH_SIZE), action_batch.squeeze()]

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_accrued_rewards = accrued_reward_batch + reward_batch
        next_augmented_states = torch.concatenate((next_state_batch,next_accrued_rewards),dim=1)
        next_state_values = target_net(next_augmented_states).view(-1,n_actions,n_objectives)
        total_rewards = next_accrued_rewards.unsqueeze(1) + GAMMA * next_state_values
        utilities = calculate_utilities_batch(total_rewards)
        best_actions = torch.argmax(utilities, dim=1)
        next_state_values = next_state_values[torch.arange(BATCH_SIZE),best_actions]
        # Compute the expected Q values
        expected_mutli_objective_state_action_values =  reward_batch + (next_state_values * GAMMA) * ~done_batch.unsqueeze(1)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(multi_objective_state_action_values, expected_mutli_objective_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

STATS_EVERY = 10
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
for episode in range(1000):
    # Initialize the environment and get it's state
    done = False
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    accrued_reward = np.array([0, 0, 0, 0, 0, 0])
    episode_reward = 0

    while not done:
        action = select_action(obs,accrued_reward)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += utility_f(reward)
        accrued_reward = accrued_reward + reward
        accrued_reward_tensor = torch.tensor(accrued_reward, dtype=torch.float32, device=device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = terminated
        done = torch.tensor([done], device=device)

        memory.push(obs, action, next_obs, reward,accrued_reward_tensor,done)

        obs = next_obs

        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}')


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()