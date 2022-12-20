from powergym.powergym.env_register import make_env
import matplotlib.pyplot as plt

NR_EPISODES = 1000
env = make_env('13Bus')

r_per_episode = [0] * (NR_EPISODES)
for i in range(1, NR_EPISODES):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r_per_episode[i] += reward


cum_r_per_episode = [item / (idx + 1) for idx,item in enumerate(r_per_episode)]
print(cum_r_per_episode)
plt.plot(r_per_episode)
plt.show()
