import gym
import deepdrive_zero

env = gym.make('deepdrive_2d-one-waypoint-v0')
env_config = dict(
    env_name='deepdrive_2d-one-waypoint-v0',
    is_intersection_map=True,
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

env.configure_env(env_config)
env.reset()
env.render()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render()


