from deepdrive_zero.envs import OneWaypointSteerOnlyEnv, \
    OneWaypointEnv, IncentArrivalEnv, StaticObstacleEnv, \
    NoGforcePenaltyEnv, SixtyFpsEnv, IntersectionEnv, IntersectionWithGsEnv, \
    IntersectionWithGsAllowDecelEnv
from deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS

env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    discrete_actions=COMFORTABLE_ACTIONS

)

env = IntersectionEnv()
env.configure_env(env_config)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
