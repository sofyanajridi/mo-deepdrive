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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'accrued_reward', 'done'))


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

    def __init__(self, n_inputs, n_outputs):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(n_inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """
    Single DQN Agent

    """

    def __init__(self, env, batch_size, gamma, tau, lr, utility_f):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.shape[0]
        self.n_objectives = env.reward_space.shape[0]
        self.policy_nn = DQNNetwork(self.n_observations, self.n_actions * self.n_objectives).to(
            device)
        self.target_nn = DQNNetwork(self.n_observations, self.n_actions * self.n_objectives).to(
            device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.optimizer = optim.AdamW(self.policy_nn.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.eps_decay = 1000
        self.utility_f = utility_f
        self.steps_done = 0



    def calculate_utilities_batch(self, total_rewards):
        total_rewards = total_rewards.tolist()
        utilities = torch.tensor([[self.utility_f(j) for j in i] for i in total_rewards], dtype=torch.float32, device=device)
        return utilities

    def calculate_utilities(self, total_reward):
        total_reward = total_reward.tolist()
        utilities = torch.tensor([self.utility_f(i) for i in total_reward], dtype=torch.float32, device=device)

        return utilities

    def select_action(self, state, accrued_reward):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                accrued_reward = torch.tensor(accrued_reward, dtype=torch.float32)
                multi_obj_state_action_value = self.policy_nn(state).view(self.n_actions, self.n_objectives)
                total_reward = multi_obj_state_action_value + accrued_reward
                utility = self.calculate_utilities(total_reward)
                best_action = torch.argmax(utility).unsqueeze(0).unsqueeze(0)
                return best_action
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
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
        multi_objective_state_values = self.policy_nn(state_batch).view(-1, self.n_actions, self.n_objectives)
        multi_objective_state_action_values = multi_objective_state_values[
            torch.arange(self.batch_size), action_batch.squeeze()]

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_accrued_rewards = accrued_reward_batch + reward_batch
            next_state_values = self.target_nn(next_state_batch).view(-1, self.n_actions, self.n_objectives)
            total_rewards = next_accrued_rewards.unsqueeze(1) + self.gamma * next_state_values
            utilities = self.calculate_utilities_batch(total_rewards)
            best_actions = torch.argmax(utilities, dim=1)
            next_state_values = next_state_values[torch.arange(self.batch_size), best_actions]
            # Compute the expected Q values
            expected_mutli_objective_state_action_values = reward_batch + (
                        next_state_values * self.gamma) * ~done_batch.unsqueeze(1)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(multi_objective_state_action_values, expected_mutli_objective_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_nn.parameters(), 100)
        self.optimizer.step()

    def target_nn_soft_weights_update(self):
        target_net_state_dict = self.target_nn.state_dict()
        policy_net_state_dict = self.policy_nn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_nn.load_state_dict(target_net_state_dict)