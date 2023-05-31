import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import inspect



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'accrued_reward', 'done'))

# Code inspired from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
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
        self.layer1 = nn.Linear(n_inputs, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class MODQN:
    """
    MO DQN Agent

    """

    def __init__(self, env, batch_size, gamma, tau, lr, utility_f):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.n_actions = self.env.action_space.n
        self.n_observations = self.env.observation_space.shape[0]
        self.n_objectives = self.env.reward_space.shape[0]
        self.policy_nn = DQNNetwork(self.n_observations + self.n_objectives, self.n_actions * self.n_objectives).to(
            device)
        self.target_nn = DQNNetwork(self.n_observations + self.n_objectives, self.n_actions * self.n_objectives).to(
            device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.optimizer = optim.AdamW(self.policy_nn.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.eps_decay = 4000
        self.utility_f = utility_f
        self.steps_done = 0
        self.num_param_updates = 0


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
                accrued_reward = torch.tensor(accrued_reward, dtype=torch.float32,device=device)
                augmented_state = torch.cat((state, accrued_reward.unsqueeze(0)), dim=1)
                multi_obj_state_action_value = self.policy_nn(augmented_state).view(self.n_actions, self.n_objectives)
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
        augmented_states_batch = torch.concatenate((state_batch, accrued_reward_batch), dim=1)
        multi_objective_state_values = self.policy_nn(augmented_states_batch).view(-1, self.n_actions, self.n_objectives)
        multi_objective_state_action_values = multi_objective_state_values[
            torch.arange(self.batch_size), action_batch.squeeze()]

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_accrued_rewards = accrued_reward_batch + reward_batch
            next_augmented_states = torch.concatenate((next_state_batch, next_accrued_rewards), dim=1)
            next_state_values = self.target_nn(next_augmented_states).view(-1, self.n_actions, self.n_objectives)
            total_rewards = next_accrued_rewards.unsqueeze(1) + self.gamma * next_state_values
            utilities = self.calculate_utilities_batch(total_rewards)
            best_actions = torch.argmax(utilities, dim=1)
            next_state_values = next_state_values[torch.arange(self.batch_size), best_actions]
            # Compute the expected Q values
            expected_mutli_objective_state_action_values = reward_batch + (
                        next_state_values * self.gamma) * ~done_batch.unsqueeze(1)

        # loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(multi_objective_state_action_values, expected_mutli_objective_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # for param in self.policy_nn.parameters():
        #     param.grad.data.clamp_(-1, 1)


        torch.nn.utils.clip_grad_value_(self.policy_nn.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_nn.state_dict()
        policy_net_state_dict = self.policy_nn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_nn.load_state_dict(target_net_state_dict)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′



    def train(self, nr_episodes, enable_wandb_logging="disabled",wandb_group_name=None,wandb_name=None, enable_plot=False):
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


        STATS_EVERY = 5
        ep_rewards = []
        aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        for episode in range(nr_episodes):
            # Initialize the environment and get it's state
            done = False
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            accrued_reward = np.zeros(self.n_objectives)
            episode_reward = np.zeros(self.n_objectives)

            while not done:
                action = self.select_action(obs, accrued_reward)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                accrued_reward_tensor = torch.tensor(accrued_reward, dtype=torch.float32, device=device).unsqueeze(0)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                done = terminated
                done = torch.tensor([done], device=device)

                self.memory.push(obs, action, next_obs, reward_tensor, accrued_reward_tensor, done)


                obs = next_obs
                accrued_reward = accrued_reward + reward


                self.optimize_model()




            ep_rewards.append(self.utility_f(episode_reward))
            if not episode % STATS_EVERY:
                average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
                aggr_ep_rewards['ep'].append(episode)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
                wandb.log({'ep': episode, 'avg_reward': average_reward, 'max_reward': max(ep_rewards[-STATS_EVERY:]), 'min_reward': min(ep_rewards[-STATS_EVERY:])})
                print(f'Episode: {episode:>5d}, average reward: {average_reward}')
        if enable_plot:
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
            plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
            plt.legend(loc=4)
            plt.show()

