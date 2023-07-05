from deepdrive_zero.envs.variants import OneWaypointEnv, StaticObstacleEnv, IntersectionEnv
from deepdrive_zero.discrete_actions.comfortable_actions2 import COMFORTABLE_ACTIONS2

config = {"multi_objective": True,
          'discrete_actions': COMFORTABLE_ACTIONS2
          }
env = OneWaypointEnv(env_configuration=config, render_mode=None)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
