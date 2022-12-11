# MOMARL-Benchmarks



## Getting started


## Deepdrive benchmark

### Environment variations
![IntersactionEnv](/images/IntersactionEnv.png "IntersactionEnv")
![IntersactionEnv](/images/StaticObstacleEnv.png "IntersactionEnv")
![IntersactionEnv](/images/OneWayPointEnv.png "IntersactionEnv")


1. Multi Agent Intersaction Env
2. Single Agent StaticObstacle Env (Grey line is a bike)
3. Single Agent OneWayPoint Env

Orange point is the final state

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


### Observation space

### Reward function

Nowhere mentioned so derived from looking at the code:

Reward = speed_reward + win_reward - gforce_penalty - collision_penalty - jerk_penalty - lane_penalty)

where:
- speed_reward = I think this is depended on how close the car is to the destination.
- win_reward = reached destination 
- gforce_penalty = g_force above 0.5
- collision_penalty = collided with other car or bike
- jerk_penalty = jerk treshold reached (looking at jerk between two frames)
- lane_penalty = how far the car is outside of the lanes



### Setup

### Bugs and issues

- When cloning from the original repository and following the install instructions, the benchmark will not work. To fix, go to file _agent.py_ on line 163.
Remove following line:
`self.observation_space = env.observation_space`

- Existing expirements configurations of the authors do not work anymore, this is because they are calling a non existing function from a library (mpi4py) that has now been updated
    - Tried solving this by going back to older versions of that specific library, however this caused other issues. 

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
        I have to tried to fix this by manually editing the pyglet library and adding the missing piece of function from a random github repository (https://github.com/jpaalasm/pyglet/blob/bf1d1f209ca3e702fd4b6611377257f0e2767282/pyglet/app/base.py#L211) that still  had the original 'legacy_setup' function in their pyglet library folder, sadly this did not work. 

        Going back to older versions of the pyglet library introduced many other issues.
    
    --> **Issue has now been fixed, use pyglet==1.5.15, arcade==2.4.2 and pymunk==5.7.0 on Python 3.9 with Rosetta Emulation for M1 Mac**

## Powergym benchmark

### Environment variations


The implemented circuit systems are summerized as follows.
| **System**| **# Caps**| **# Regs**| **# Bats**|
| ------------- | ------------- |------- |------- |
| 13Bus     | 2 | 3 | 1 |
| 34Bus | 4 | 6 | 2 |
| 123Bus | 4 | 7 | 4 |
| 8500Node | 10 | 12 | 10 |

### Action space
{} denotes a finite set and [] denote a continuous interval.

Discrete battery has discretized choices on the discharge power (e.g., choose from {0,...,32}) and continuous battery chooses the normalized discharge power from the interval [-1,1]

|**Action Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| Discharge power (disc.) | {0, ..., 32} |
| Discharge power (cont.) | [-1, 1]  |

This means that for e.g for 13bus discrete an action consists of an array of length 6 (2 caps + 3 regs + 1 bats): 

`[Capacitor status 1, capacitor status 2, Regulator tap number 1, Regulator tap number 2, Regulator tap number 3, Discharge power 1 (disc.) ]`

### Observation space

{} denotes a finite set and [] denote a continuous interval.

|**Observation Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Bus voltage     | [0.8, 1.2] |
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| State-of-charge (soc) | [0, 1] |
| Discharge power  | [-1, 1]  |

### Reward function

The reward function is a combination of three losses: voltage violation, control error, and power loss. The control error is further decomposed into capacitor's & regulator's switching cost and battery's discharge loss & soc loss. The weights among these losses depends on the circuit system and is listed in the Appendix of our paper.

### Setup

### Bugs and issues
- Render function is not implemented


## Authors and acknowledgment

- Deepdrive zero: https://github.com/deepdrive/deepdrive-zero
- PowerGym: https://github.com/siemens/powergym 


## License


## Project status

