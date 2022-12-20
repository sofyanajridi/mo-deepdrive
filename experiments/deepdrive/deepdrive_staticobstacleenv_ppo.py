from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
import gym
import deepdrive_zero



env = gym.make('deepdrive_2d-static-obstacle-60-fps-v0')
env_config = dict(
    env_name='deepdrive_2d-static-obstacle-60-fps-v0',
    is_intersection_map=False,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    collision_penalty_coeff=1,
)
env.configure_env(env_config)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
# model.save("ppo_static-obstacle")

# model = PPO.load('saved_models/ppo_static-obstacle.zip',env=env)


vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

