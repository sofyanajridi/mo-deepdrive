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
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)


class A2C:

    def __init__(self, env, lr, gamma, utility_f=None, vectorial_reward=False):
        self.vectorial_reward = vectorial_reward
        self.utility_f = utility_f
        self.gamma = gamma
        self.lr = lr
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.actor_nn = Actor(self.n_observations, self.n_actions)
        self.critic_nn = Critic(self.n_observations)
        self.actor_optimizer = torch.optim.Adam(self.actor_nn.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_nn.parameters(), lr=self.lr)

    def train(self, nr_episodes, enable_wandb_logging="disabled", wandb_group_name=None, wandb_name=None,
              enable_plot=False):
        print("Starting training...")
        config = {
            "gamma": self.gamma,
            "learning_rate": self.lr,
            "utility_function": inspect.getsource(self.utility_f),
            "env_config" : self.env.env_config
        }
        print(f"Hyperparameters: {config}")
        import wandb
        wandb.init(project="momarl-benchmarks-final", name=wandb_name, config=config,
                   group=wandb_group_name, mode=enable_wandb_logging)

        STATS_EVERY = 5
        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        timestep = 0
        avg_reward_per_timestep = []

        for episode in range(nr_episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()

            while not done:
                probs = self.actor_nn(torch.from_numpy(obs).float())
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action.detach().data.numpy())

                if self.vectorial_reward:  # take direct utility over rewards (This is a wrong approach, only used for baseline comparison)
                    reward = self.utility_f(reward)
                done = terminated

                episode_reward += reward
                wandb.log({'timestep': timestep, 'reward_per_timestep': reward
                           })

                advantage = reward + (1 - done) * self.gamma * self.critic_nn(
                    torch.from_numpy(next_obs).float()) - self.critic_nn(
                    torch.from_numpy(obs).float())

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
                print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}')
        if enable_plot:
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
            plt.legend(loc=4)
            plt.show()


def multi_agent_train(vehicle_1: A2C, vehicle_2: A2C, env, nr_episodes, enable_wandb_logging="disabled",
                      wandb_group_name=None,
                      wandb_name=None,config=None):
    print("Starting training...")
    config = {
        "vehicle_1_gamma": vehicle_1.gamma,
        "vehicle_1_learning_rate": vehicle_1.lr,
        "vehicle_1_utility_function": inspect.getsource(vehicle_1.utility_f),
        "vehicle_2_gamma": vehicle_2.gamma,
        "vehicle_2_learning_rate": vehicle_2.lr,
        "vehicle_2_utility_function": inspect.getsource(vehicle_2.utility_f),
        "env_config": config
    }
    print(f"Hyperparameters: {config}")
    import wandb
    wandb.init(project="momarl-benchmarks-final", name=wandb_name, config=config,
               group=wandb_group_name, mode=enable_wandb_logging)

    STATS_EVERY = 5
    ep_rewards_vehicle_1 = []
    aggr_ep_rewards_vehicle_1 = {'ep': [], 'avg': [], 'max': [], 'min': []}
    ep_rewards_vehicle_2 = []
    aggr_ep_rewards_vehicle_2 = {'ep': [], 'avg': [], 'max': [], 'min': []}
    ep_rewards_total = []
    aggr_ep_rewards_total = {'ep': [], 'avg': [], 'max': [], 'min': []}
    timestep = 0


    for episode in range(nr_episodes):
        done = False
        episode_reward_vehicle_1 = 0
        episode_reward_vehicle_2 = 0
        total_episode_reward = 0
        obs, info = env.reset()

        while not done:
            probs_vehicle_1 = vehicle_1.actor_nn(torch.from_numpy(obs['vehicle1']).float())
            dist_vehicle_1 = torch.distributions.Categorical(probs=probs_vehicle_1)
            action_vehicle_1 = dist_vehicle_1.sample()

            probs_vehicle_2 = vehicle_2.actor_nn(torch.from_numpy(obs['vehicle2']).float())
            dist_vehicle_2 = torch.distributions.Categorical(probs=probs_vehicle_2)
            action_vehicle_2 = dist_vehicle_2.sample()

            next_obs, reward, terminated, truncated, info = env.step(
                {'vehicle1': action_vehicle_1.detach().data.numpy(),
                 'vehicle2': action_vehicle_2.detach().data.numpy()})

            # print(next_obs)
            # vehicle_1_reaches_destination.append(reward['vehicle1'][1])
            # vehicle_2_reaches_destination.append(reward['vehicle2'][1])
            # wa


            if vehicle_1.vectorial_reward:  # take direct utility over rewards (This is a wrong approach, only used for baseline comparison)
                reward['vehicle1'] = vehicle_1.utility_f(reward['vehicle1'])
                reward['vehicle2'] = vehicle_2.utility_f(reward['vehicle2'])

            done = terminated

            episode_reward_vehicle_1 += reward['vehicle1']
            episode_reward_vehicle_2 += reward['vehicle2']
            total_episode_reward += reward['vehicle1'] + reward['vehicle2']
            wandb.log({'timestep': timestep, 'reward_per_timestep_vehicle_1': reward['vehicle1'],
                       'reward_per_timestep_vehicle_2': reward['vehicle2']})

            advantage_vehicle_1 = reward['vehicle1'] + (1 - done['vehicle1']) * vehicle_1.gamma * vehicle_1.critic_nn(
                torch.from_numpy(next_obs['vehicle1']).float()) - vehicle_1.critic_nn(
                torch.from_numpy(obs['vehicle1']).float())

            advantage_vehicle_2 = reward['vehicle2'] + (1 - done['vehicle2']) * vehicle_2.gamma * vehicle_2.critic_nn(
                torch.from_numpy(next_obs['vehicle2']).float()) - vehicle_2.critic_nn(
                torch.from_numpy(obs['vehicle2']).float())

            obs = next_obs

            critic_loss_vehicle_1 = advantage_vehicle_1.pow(2).mean()
            vehicle_1.critic_optimizer.zero_grad()
            critic_loss_vehicle_1.backward()
            vehicle_1.critic_optimizer.step()

            critic_loss_vehicle_2 = advantage_vehicle_2.pow(2).mean()
            vehicle_2.critic_optimizer.zero_grad()
            critic_loss_vehicle_2.backward()
            vehicle_2.critic_optimizer.step()

            actor_loss_vehicle_1 = -dist_vehicle_1.log_prob(action_vehicle_1) * advantage_vehicle_1.detach()
            vehicle_1.actor_optimizer.zero_grad()
            actor_loss_vehicle_1.backward()
            vehicle_1.actor_optimizer.step()

            actor_loss_vehicle_2 = -dist_vehicle_2.log_prob(action_vehicle_2) * advantage_vehicle_2.detach()
            vehicle_2.actor_optimizer.zero_grad()
            actor_loss_vehicle_2.backward()
            vehicle_2.actor_optimizer.step()

            if done["vehicle1"] and done["vehicle1"]:
                done = True
                wandb.log({'ep_done': episode, 'won_vehicle_1': info['vehicle1']['stats']['done_only']['won'],
                          'won_vehicle_2': info['vehicle2']['stats']['done_only']['won'], 'info_vehicle_1': info['vehicle1'],'info_vehicle_2': info['vehicle2'] })

            else:
                done = False



            timestep += 1

        wandb.log({'episode': episode, 'total_reward_per_ep': total_episode_reward,
                   'reward_per_ep_vehicle_1': episode_reward_vehicle_1,
                   'reward_per_ep_vehicle_2': episode_reward_vehicle_2
                   })
        ep_rewards_vehicle_1.append(episode_reward_vehicle_1)
        ep_rewards_vehicle_2.append(episode_reward_vehicle_2)
        ep_rewards_total.append(total_episode_reward)

        if not episode % STATS_EVERY:
            average_reward_vehicle_1 = sum(ep_rewards_vehicle_1[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_rewards_vehicle_1['ep'].append(episode)
            aggr_ep_rewards_vehicle_1['avg'].append(average_reward_vehicle_1)
            aggr_ep_rewards_vehicle_1['max'].append(max(ep_rewards_vehicle_1[-STATS_EVERY:]))
            aggr_ep_rewards_vehicle_1['min'].append(min(ep_rewards_vehicle_1[-STATS_EVERY:]))

            average_reward_vehicle_2 = sum(ep_rewards_vehicle_2[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_rewards_vehicle_2['ep'].append(episode)
            aggr_ep_rewards_vehicle_2['avg'].append(average_reward_vehicle_2)
            aggr_ep_rewards_vehicle_2['max'].append(max(ep_rewards_vehicle_2[-STATS_EVERY:]))
            aggr_ep_rewards_vehicle_2['min'].append(min(ep_rewards_vehicle_2[-STATS_EVERY:]))

            average_reward_total = sum(ep_rewards_total[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_rewards_total['ep'].append(episode)
            aggr_ep_rewards_total['avg'].append(average_reward_total)
            aggr_ep_rewards_total['max'].append(max(ep_rewards_total[-STATS_EVERY:]))
            aggr_ep_rewards_total['min'].append(min(ep_rewards_total[-STATS_EVERY:]))

            wandb.log({'ep': episode, 'avg_reward_vehicle_1': average_reward_vehicle_1,
                       'max_reward_vehicle_1': max(ep_rewards_vehicle_1[-STATS_EVERY:]),
                       'min_reward_vehicle_1': min(ep_rewards_vehicle_1[-STATS_EVERY:]),
                       'avg_reward_vehicle_2': average_reward_vehicle_2,
                       'max_reward_vehicle_2': max(ep_rewards_vehicle_2[-STATS_EVERY:]),
                       'min_reward_vehicle_2': min(ep_rewards_vehicle_2[-STATS_EVERY:]),
                       'avg_reward_total': average_reward_total,
                       'max_reward_total': max(ep_rewards_total[-STATS_EVERY:]),
                       'min_reward_total': min(ep_rewards_total[-STATS_EVERY:])})

            print(
                f'Episode: {episode:>5d}, average reward_vehicle_1: {average_reward_vehicle_1:>4.1f}, average reward_vehicle_2: {average_reward_vehicle_2:>4.1f}, average reward_total: {average_reward_total:>4.1f} ')
