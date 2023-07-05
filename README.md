# MO-DeepDrive



## Getting started


## Deepdrive benchmark

### Environment variations
![IntersactionEnv](/momarl_benchmarks/images/IntersactionEnv.png "IntersactionEnv")
![IntersactionEnv](/momarl_benchmarks/images/StaticObstacleEnv.png "IntersactionEnv")
![IntersactionEnv](/momarl_benchmarks/images/OneWayPointEnv.png "IntersactionEnv")


1. Multi Agent Intersaction Env
2. Single Agent StaticObstacle Env (Black line is a bike)
3. Single Agent OneWayPoint Env

Orange point is the final state which changes position every episode for the OneWayPointEnv and StaticObstacleEnv.

Each environment can be further configured on the following properties:
 - jerk_penalty_coeff              (default 0)
 - gforce_penalty_coeff            (default 0)
 - lane_penalty_coeff              (default 0.02)
 - collision_penalty_coeff         (default 0.31)
 - speed_reward_coeff              (default 0.5)
 - win_coefficient                 (default 1)
 - gforce_threshold                (default 1)
 - jerk_threshold                  (default None)
 - constrain_controls              (default False)
 - ignore_brake                    (default False)
 - forbid_deceleration             (default False)
 - expect_normalized_action_deltas (default False)
 - discrete_actions
 - incent_win                      (default True)
 - dummy_accel_agent_indices       (default None)
 - wait_for_action                 (default False)
 - incent_yield_to_oncoming_traffic (default True)
 - physics_steps_per_observation   (default 6)
 - end_on_lane_violation           (default False)
 - lane_margin                     (default 0)
 - is_intersection_map             (default True)
 - end_on_harmful_gs               (default False)

### Single agent vs Multi agent

The env.step function works the same for the single agent and the multi agent env variation, however for the multi agent variation we are looping through agents:
1. agent 1 obs = env.reset() (Get a blank observation, i.e. just zeroes)
2. agent_1_action = model(agent_1_obs)
3. agent_2_obs = env.step(agent_1_action) (step 1: agent_2_obs is just blank)
4. agent_2_action = model(agent_2_obs)
5. agent_1_obs = env.step(agent_2_ action) (step 2: where agent_1_obs was from step 1 above)
6. agent_1_action = model(agent_1_obs)
7. agent_2_obs = env.step(agent_1_action) (step 3)
8. ...


### Action space
You can choose between a discrete or continous action space.
- Discrete actions: (Between 0 and 15)
    - 0: 'IDLE'
    - 1: 'DECAY_STEERING_MAINTAIN_SPEED'
    - 2: 'DECAY_STEERING_DECREASE_SPEED'
    - 3: 'DECAY_STEERING_INCREASE_SPEED'
    - 4: 'SMALL_STEER_LEFT_MAINTAIN_SPEED'
    - 5: 'SMALL_STEER_LEFT_DECREASE_SPEED'
    - 6: 'SMALL_STEER_LEFT_INCREASE_SPEED'
    - 7: 'SMALL_STEER_RIGHT_MAINTAIN_SPEED'
    - 8: 'SMALL_STEER_RIGHT_DECREASE_SPEED'
    - 9: 'SMALL_STEER_RIGHT_INCREASE_SPEED'
    - 10: 'LARGE_STEER_LEFT_MAINTAIN_SPEED'
    - 11: 'LARGE_STEER_LEFT_DECREASE_SPEED'
    - 12: 'LARGE_STEER_LEFT_INCREASE_SPEED'
    - 13: 'LARGE_STEER_RIGHT_MAINTAIN_SPEED'
    - 14: 'LARGE_STEER_RIGHT_DECREASE_SPEED'
    - 15: 'LARGE_STEER_RIGHT_INCREASE_SPEED'

- Continous actions: [steer, accel, brake] continous value between -1 and 1 for each action
    - steer: Heading angle of the ego
    - accel: m/s/s of the ego, positive for forward, negative for reverse
    - brake: From 0g at -1 to 1g at 1 of brake force

    |**Action Space** | |
    | ------------- | ------------- |
    | **Variable**| **Range**|
    | steer     | {-1, 1} |
    | accel | {-1, 1} |
    | brake | {-1, 1} |



### Multi Objective

The environment supports multi objective RL and can be configured to return a vectorized reward instead of a scalar reward.
This can be done by configuring the env_config function of env after calling env.make() and adding the following
property ``` multi_objective=True ```




### Setup

### Bugs and issues

- env.render() does not work anymore, again this is because they are calling a non existing function from a library (pyglet) that has now been updated
    - The function that is causing this issue can be found in file _env.py_  line 261:
        ```
        pyglet.app.event_loop.has_exit = False
        pyglet.app.event_loop._legacy_setup() # This line is causing the issue. 
        pyglet.app.event_loop.run()
        pyglet.app.platform_event_loop.start()
        pyglet.app.event_loop.dispatch_event('on_enter')
        pyglet.app.event_loop.is_running = True
        ```


    
## Authors and acknowledgment

- Deepdrive Zero: https://github.com/deepdrive/deepdrive-zero



## License


## Project status

