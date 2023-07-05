import gymnasium as gym
from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
from deepdrive_2d.deepdrive_zero.envs.variants import OneWaypointEnv,StaticObstacleEnv

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

env = StaticObstacleEnv(env_configuration=env_config)



obs = env.reset()

print(env.observation_space.shape[0])


