from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2

import numpy as np
import torch
import gymnasium
from torch import nn
import matplotlib.pyplot as plt

# Code from:
# https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b
# https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/cartpole_a2c_online.ipynb


''' One step Actor critic '''


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, state_dim, reward_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, reward_dim)
        )

    def forward(self, X):
        return self.model(X)


env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    is_intersection_map=False,
    discrete_actions=COMFORTABLE_ACTIONS2,
    incent_win=True,
    multi_objective=True
)

env = OneWaypointEnv(env_configuration=env_config, render_mode=None)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
reward_dim = env.reward_space.shape[0]

print(state_dim)
print(n_actions)
print(reward_dim)

actor = Actor(state_dim, n_actions)
critic = Critic(state_dim, reward_dim)

adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99


def utility(vec):
    speed_reward, win_reward, gforce_penalty, collision_penalty, jerk_penalty, lane_penalty = vec
    return (0.50 * speed_reward) + win_reward - (0.006 * 5 * gforce_penalty) - (4 * collision_penalty) - (3.3e-5 * jerk_penalty) - (0.02 * lane_penalty)


episode_rewards = []

for i in range(500):
    done = False
    total_reward = 0
    obs, info = env.reset()
    accrued_reward = [0, 0, 0, 0, 0, 0]
    timestep = 0

    while not done:
        probs = actor(torch.from_numpy(obs).float())
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        next_obs, reward, terminated, truncated, info = env.step(action.detach().data.numpy())
        done = terminated

        sum2 = utility(
            torch.Tensor(accrued_reward) + pow(gamma, timestep) * critic(
                torch.from_numpy(obs).float()))

        sum1 = utility(torch.Tensor(accrued_reward) + pow(gamma, timestep) * (
                torch.Tensor(reward) + gamma * critic(torch.from_numpy(next_obs).float())))

        advantage = sum1 - sum2

        total_reward += utility(reward)

        accrued_reward += pow(gamma, timestep) * reward
        obs = next_obs

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = -dist.log_prob(action) * advantage.detach()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()
        timestep += 1

    episode_rewards.append(total_reward)
