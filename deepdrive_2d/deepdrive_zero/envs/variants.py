from deepdrive_2d.deepdrive_zero.envs import Deepdrive2DEnv


class OneWaypointSteerOnlyEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class OneWaypointEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False, env_configuration=env_configuration,
                         render_mode=render_mode)


class IncentArrivalEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class StaticObstacleEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class NoGforcePenaltyEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True,
                         disable_gforce_penalty=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class SixtyFpsEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True,
                         disable_gforce_penalty=True,
                         physics_steps_per_observation=1, env_configuration=env_configuration,
                         render_mode=render_mode)


class IntersectionEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, disable_gforce_penalty=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class IntersectionWithGsEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, env_configuration=env_configuration,
                         render_mode=render_mode)


class IntersectionWithGsAllowDecelEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration, render_mode=None):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, forbid_deceleration=False, env_configuration=env_configuration,
                         render_mode=render_mode)
