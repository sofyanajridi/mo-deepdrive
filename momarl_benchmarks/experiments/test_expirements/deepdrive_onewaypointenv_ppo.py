import gymnasium as gym
from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
import sys
sys.modules["gym"] = gym
from stable_baselines3 import A2C

env_config = dict(
    env_name='deepdrive-2d-onewaypoint',
    discrete_actions=COMFORTABLE_ACTIONS2,
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
env = OneWaypointEnv(env_configuration=env_config, render_mode=None)
env = V26toV21Wrapper(env)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)


vec_env = env
obs = vec_env.reset()

for i in range(10):
    done = False
    obs = vec_env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

