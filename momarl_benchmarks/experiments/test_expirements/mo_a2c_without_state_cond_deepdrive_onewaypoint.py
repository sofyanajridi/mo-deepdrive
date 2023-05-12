from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2

import numpy as np
import torch
import gymnasium
from torch import nn
import matplotlib.pyplot as plt
from loguru import logger
logger.stop()

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

actor = Actor(state_dim , n_actions)
critic = Critic(state_dim , reward_dim)

adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99


config = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "utility_function":"(0.50 * speed_reward ) + (win_reward) - (torch.pow((0.03 * gforce),2) - torch.pow((3.3e-5 * jerk),3))"

}
import wandb
wandb.init(project="momarl-benchmarks",name="MO_A2C_DeepDrive_OneWayPoint_w/o_state_cond", config=config, group="MO_A2C_DeepDrive_OneWayPoint_w/o_state_cond")


# def utility(vec):
#     speed_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec
#     aggresive_penalty = torch.pow(((3.3e-5 *jerk) + (0.03 * gforce)),3)
#     return torch.pow(speed_reward,2) + 5 * win_reward - torch.log(1 + aggresive_penalty)


def utility(vec):
    speed_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec
    penalty = (0.50 * speed_reward ) + (100 * win_reward) - (0.006 * 5 * (10 * gforce) - (3.3e-5 * (10 * gforce)))
    return (0.50 * speed_reward ) + (win_reward) - (torch.pow((0.03 * gforce),2) - torch.pow((3.3e-5 * jerk),3))

# def utility(vec):
#     distance_reward, win_reward, gforce, collision_penalty, jerk, lane_penalty = vec
#
#     return (0.50 * distance_reward) + (win_reward) - (0.03 * gforce) - (3.3e-5 * jerk)



STATS_EVERY = 5
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

for episode in range(5000):
    done = False
    total_reward = 0
    episode_reward = 0
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
                torch.Tensor(reward) + gamma * critic(
            torch.from_numpy(next_obs).float())))

        advantage = sum1 - sum2


        episode_reward += utility(torch.Tensor(reward))


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

