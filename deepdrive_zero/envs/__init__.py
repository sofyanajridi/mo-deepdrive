from gymnasium.envs.registration import register

# TODO: Rename 2d here and in RunConfigurations to zero

register(
    id='deepdrive_2d-v0',
    entry_point='deepdrive_zero.envs:Deepdrive2DEnv',
)

register(
    id='deepdrive_2d-one-waypoint-v0',
    entry_point='deepdrive_zero.envs:OneWaypointEnv',
)

register(
    id='deepdrive_2d-static-obstacle-v0',
    entry_point='deepdrive_zero.envs:StaticObstacleEnv',
)

register(
    id='deepdrive_2d-intersection-v0',
    entry_point='deepdrive_zero.envs:IntersectionEnv',
)
