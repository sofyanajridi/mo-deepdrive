from deepdrive_zero.envs.env import Deepdrive2DEnv
class OneWaypointEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False, env_configuration=env_configuration,
                         render_mode=render_mode,env_variation='OneWayPoint')

class StaticObstacleEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True, env_configuration=env_configuration,
                         render_mode=render_mode,env_variation='StaticObstacle')

class IntersectionEnv(Deepdrive2DEnv):
    def __init__(self, env_configuration,  render_mode=None):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, disable_gforce_penalty=True, env_configuration=env_configuration,
                         render_mode=render_mode,env_variation='InterSection')