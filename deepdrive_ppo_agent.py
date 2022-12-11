
from stable_baselines3 import PPO

import gym

import deepdrive_zero

env = gym.make('deepdrive_2d-one-waypoint-v0')
env_config = dict(
    env_name='deepdrive_2d-one-waypoint-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=3.3e-5,
    gforce_penalty_coeff=0.006 * 5,
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.02,
    speed_reward_coeff=0.50,
    gforce_threshold=None,
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
)

env.configure_env(env_config)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

env.close()