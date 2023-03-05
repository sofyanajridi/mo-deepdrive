import gymnasium as gym
from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
from deepdrive_2d.deepdrive_zero.envs.variants import IntersectionWithGsAllowDecelEnv
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

env = IntersectionWithGsAllowDecelEnv(env_configuration=env_config)
env = V26toV21Wrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000)
# model.save("ppo_intersection")

vec_env = env
obs = vec_env.reset()

for i in range(10):
    done = False
    obs = vec_env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print("final obs")
        print(obs)
        vec_env.render()

