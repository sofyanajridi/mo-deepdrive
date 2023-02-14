import gymnasium as gym
from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
import sys
sys.modules["gym"] = gym
from stable_baselines3 import PPO


env_config = dict(
    env_name='deepdrive_2d-static-obstacle-v0',
    is_intersection_map=False,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    collision_penalty_coeff=1,
    add_static_obstacle=True,
    multi_objective=False
)


env = gym.make('deepdrive_2d-static-obstacle-v0', env_configuration=env_config, render_mode=None)
env = V26toV21Wrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)


vec_env = env
obs = vec_env.reset()

for i in range(10):
    done = False
    obs = vec_env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

