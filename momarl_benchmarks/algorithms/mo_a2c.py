import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import inspect



# Code inspired from:
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


class MOA2C:

    def __init__(self, env, lr, gamma, utility_f):
        self.utility_f = utility_f
        self.gamma = gamma
        self.lr = lr
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_objectives = env.reward_space.shape[0]
        self.actor_nn = Actor(self.n_observations + self.n_objectives, self.n_actions)
        self.critic_nn = Critic(self.n_observations + self.n_objectives,  self.n_objectives)
        self.actor_optimizer = torch.optim.Adam(self.actor_nn.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_nn.parameters(), lr=self.lr)


    def train(self, nr_episodes, enable_wandb_logging="disabled",wandb_group_name=None,wandb_name=None, enable_plot=False):
        print("Starting training...")
        config = {
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "utility_function": inspect.getsource(self.utility_f)
        }
        print(f"Hyperparameters: {config}")
        import wandb
        wandb.init(project="momarl-benchmarks", name=wandb_name, config=config,
                   group=wandb_group_name, mode=enable_wandb_logging)

        STATS_EVERY = 5
        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

        for episode in range(nr_episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            accrued_reward = np.zeros(self.n_objectives)
            timestep = 0

            while not done:
                probs = self.actor_nn(torch.from_numpy(np.concatenate((obs, accrued_reward))).float())
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action.detach().data.numpy())
                done = terminated

                sum2 = self.utility_f(
                    torch.Tensor(accrued_reward) + pow(self.gamma, timestep) * self.critic_nn(
                        torch.from_numpy(np.concatenate((obs, accrued_reward))).float()))

                sum1 = self.utility_f(torch.Tensor(accrued_reward) + pow(self.gamma, timestep) * (
                        torch.Tensor(reward) + self.gamma * self.critic_nn(
                    torch.from_numpy(
                        np.concatenate((next_obs, (accrued_reward + (pow(self.gamma, timestep) * reward))))).float())))

                advantage = sum1 - sum2

                episode_reward += self.utility_f(torch.Tensor(reward))

                accrued_reward += pow(self.gamma, timestep) * reward
                obs = next_obs

                critic_loss = advantage.pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_loss = -dist.log_prob(action) * advantage.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                timestep += 1

            ep_rewards.append(episode_reward)
            if not episode % STATS_EVERY:
                average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
                aggr_ep_rewards['ep'].append(episode)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
                wandb.log({'ep': episode, 'avg_reward': average_reward, 'max_reward': max(ep_rewards[-STATS_EVERY:]),
                           'min_reward': min(ep_rewards[-STATS_EVERY:])})
                print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}')
        if enable_plot:
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
            plt.legend(loc=4)
            plt.show()




