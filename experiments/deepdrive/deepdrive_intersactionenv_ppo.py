import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
import deepdrive_zero

from deepdrive_zero.v26_To_V21Wrapper import V26toV21Wrapper



import sys
sys.modules["gym"] = gym
from stable_baselines3 import PPO

env_config = dict(
    env_name='deepdrive_2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    collision_penalty_coeff=1,
)

env = gym.make('deepdrive_2d-intersection-w-gs-allow-decel-v0', env_configuration=env_config,render_mode="human")
# env = StepAPICompatibility(env, output_truncation_bool=False)
env = V26toV21Wrapper(env)

obs = env.reset()



# env.configure_env(env_config)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)
# model.save("ppo_intersection")

# model = PPO.load('../../saved_models/ppo_intersection.zip', env=env)
# model.learn(total_timesteps=10)


vec_env = env
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

