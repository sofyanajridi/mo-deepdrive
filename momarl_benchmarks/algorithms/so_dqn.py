import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import inspect

from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))


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



class DQNNetwork(nn.Module):
    """
    DQN network

    """

    def __init__(self, n_observations, n_actions):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN:
    """
    DQN Agent

    """

    def __init__(self, env, batch_size, gamma, tau, lr, utility_f=None, vectorial_reward=False):
        self.vectorial_reward = vectorial_reward
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.n_actions = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]
        self.policy_nn = DQNNetwork(self.n_observations, self.n_actions).to(
            device)
        self.target_nn = DQNNetwork(self.n_observations, self.n_actions).to(
            device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.optimizer = optim.AdamW(self.policy_nn.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.eps_decay = 4000
        self.utility_f = utility_f
        self.steps_done = 0

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_nn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # print(non_final_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        test_action_values = self.policy_nn(state_batch)
        state_action_values = test_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = self.target_nn(next_state_batch).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (next_state_values * self.gamma) * ~done_batch
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_nn.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_nn.state_dict()
        policy_net_state_dict = self.policy_nn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_nn.load_state_dict(target_net_state_dict)
        
    def train(self,nr_episodes, enable_wandb_logging="disabled",wandb_group_name=None,wandb_name=None):
        print("Starting training...")

        config = {
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "learning_rate": self.lr,
            "utility_function": inspect.getsource(self.utility_f)
        }
        print(f"Hyperparameters: {config}")
        import wandb
        wandb.init(project="momarl-benchmarks-final", name=wandb_name, config=config,
                   group=wandb_group_name, mode=enable_wandb_logging)

        STATS_EVERY = 1
        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        
        
        for episode in range(nr_episodes):
            # Initialize the environment and get it's state
            done = False
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            timestep = 0
            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated


                if self.vectorial_reward:
                    reward = self.utility_f(reward)
                episode_reward += reward
                reward = torch.tensor([reward], device=device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                done = torch.tensor([done], device=device)


                # Store the transition in memory
                self.memory.push(obs, action, next_obs, reward, done)



                # Move to the next state
                obs = next_obs

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′

                timestep += 1
                
            wandb.log({'episode': episode, 'reward_per_ep': episode_reward,
                       })
            ep_rewards.append(episode_reward)
            if not episode % STATS_EVERY:
                average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
                aggr_ep_rewards['ep'].append(episode)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
                wandb.log({'ep': episode, 'avg_reward': average_reward, 'max_reward': max(ep_rewards[-STATS_EVERY:]),
                           'min_reward': min(ep_rewards[-STATS_EVERY:])})
                print(f'Episode: {episode:>5d}, average reward: {average_reward}')
        