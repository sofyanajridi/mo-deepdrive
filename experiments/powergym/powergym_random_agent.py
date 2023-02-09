from powergym.powergym.env_register import make_env
import matplotlib.pyplot as plt

NR_EPISODES = 1000
env = make_env('13Bus', multi_objective=False)

r_per_episode = [0] * (NR_EPISODES)
for i in range(1, NR_EPISODES):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


