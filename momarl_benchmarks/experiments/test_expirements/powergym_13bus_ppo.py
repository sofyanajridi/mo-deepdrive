from powergym.powergym.env_register import make_env
from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
import matplotlib.pyplot as plt
import gymnasium as gym
import sys
sys.modules["gym"] = gym
from stable_baselines3 import PPO

NR_EPISODES = 1000
env = make_env('13Bus')
env = V26toV21Wrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

r_per_episode = [0] * (NR_EPISODES)

for i in range(0, NR_EPISODES):
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        r_per_episode[i] += reward


plt.plot(r_per_episode)
plt.show()