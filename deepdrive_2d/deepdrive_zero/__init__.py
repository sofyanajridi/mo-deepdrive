from gymnasium.envs.registration import register

# TODO: Rename 2d here and in RunConfigurations to zero

register(
    id='deepdrive_2d-v0',
    entry_point='deepdrive_zero.envs:Deepdrive2DEnv',
)

register(
    id='deepdrive_2d-one-waypoint-steer-only-v0',
    entry_point='deepdrive_zero.envs:OneWaypointSteerOnlyEnv',
)

register(
    id='deepdrive_2d-one-waypoint-v0',
    entry_point='deepdrive_zero.envs:OneWaypointEnv',
)

register(
    id='deepdrive_2d-incent-arrival-v0',
    entry_point='deepdrive_zero.envs:IncentArrivalEnv',
)

register(
    id='deepdrive_2d-static-obstacle-v0',
    entry_point='deepdrive_zero.envs:StaticObstacleEnv',
)

register(
    id='deepdrive_2d-static-obstacle-no-g-pen-v0',
    entry_point='deepdrive_zero.envs:NoGforcePenaltyEnv',
)

register(
    id='deepdrive_2d-static-obstacle-60-fps-v0',
    entry_point='deepdrive_zero.envs:SixtyFpsEnv',
)

register(
    id='deepdrive_2d-intersection-v0',
    entry_point='deepdrive_zero.envs:IntersectionEnv',
)

register(
    id='deepdrive_2d-intersection-w-gs-v0',
    entry_point='deepdrive_zero.envs:IntersectionWithGsEnv',
)

register(
    id='deepdrive_2d-intersection-w-gs-allow-decel-v0',
    entry_point='deepdrive_zero.envs:IntersectionWithGsAllowDecelEnv',
)