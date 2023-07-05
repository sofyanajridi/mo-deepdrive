# mo-deepdrive


Self driving car benchmark based on the original DeepDrive simulator with added support for multi-objective RL.

## Motivation
To the best of our knowledge, the development for both the DeepDrive and DeepDrive zero
benchmarks has been abandoned. There has been no update since June 2020 for the DeepDrive benchmark and November 2021 for the DeepDrive Zero benchmark. Both benchmarks
fail during installation due to outdated libraries and API calls. However, the DeepDrive zero
benchmark is still a very interesting simulator for the development and introduction to the field
of single-agent, multi-agent, single-objective and multi-objective RL. Moreover, looking
closer at the objective in the DeepDrive benchmark reveals that it is constructed of multiple
objectives where those objectives are combined into a scalar reward. As such, we deconstruct
this benchmark, reveal its underlying multi-objective nature, and update, extend, and modify it
to serve as a novel benchmark for multi-objective (multi-agent) reinforcement learning literature.



## Enviroment variation

### OneWayPoint (single-agent)

```python
from deepdrive_zero.envs.variants import OneWaypointEnv
config = {"multi_objective": True}
env = OneWaypointEnv(env_configuration=config, render_mode=None)
```

The OneWayPoint environment is the simplest environment variation in
the benchmark. In this scenario, a single vehicle must drive to the end destination, highlighted
as an orange circle. The placement of the orange circle is random and changes every episode.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

### StaticObstacle (single-agent)

```python
from deepdrive_zero.envs.variants import StaticObstacleEnv
config = {"multi_objective": True}
env = StaticObstacleEnv(env_configuration=config, render_mode=None)
```

The StaticObstacle environment is the second most challenging environment.
This scenario includes a static obstacle which is represented as a bike. Here,
a single vehicle must drive to the end destination without causing a collision with
the bike. Both the placement of the obstacle and the end destination are random.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

### IntersectionEnv (multi-agent)

```python
from deepdrive_zero.envs.variants import IntersectionEnv
config = {"multi_objective": True}
env = IntersectionEnv(env_configuration=config, render_mode=None)
```

The Intersection environment is the most difficult environment and the only multi-agent one. This environment represents the unprotected left scenario, one of
the most difficult scenarios for both autonomous vehicles and human drivers. In this scenario, two vehicles need to cross an intersection to reach two different
destinations without colliding with each other. Both vehicles should also not deviate from their
lane. Vehicle one (the vehicle at the bottom right of the intersection) has the easiest trajectory
as it only needs to go straight until it reaches the orange circle. Vehicle two has the most difficult
trajectory as it needs to go straight, turn to the left and reach the green circle at the left side
of the intersection. A collision or not reaching the destination after a certain timestep ends the
episode.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

## Action Space
You can choose between a discrete or continous action space.
- Discrete actions: (Between 0 and 21)
    - 0: 'IDLE'
    - 1: 'DECAY_STEERING_MAINTAIN_SPEED'
    - ...
    - 20: 'LARGE_STEER_RIGHT_DECREASE_SPEED',
    - 21: 'LARGE_STEER_RIGHT_INCREASE_SPEED',

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

## Multi-Objective

This benchmarj supports multi-objective RL and can be configured to return a vectorized reward instead of a scalar reward.
This can be done by configuring the env_config function of env after calling env.make() and adding the following
property ``` multi_objective=True ```.

Each environment has the following objectives:

**OneWayPoint:**

$$  \begin{bmatrix}
    distance \\
    destination \\
    g−force \\
    jerk \\
    \end{bmatrix} 
    $$ 
    
**StaticObstacle:**

$$  \begin{bmatrix}
    distance \\
    destination \\
    g−force \\
    collision \\
    jerk \\
    \end{bmatrix} 
    $$ 

**Intersection:**

$$  \begin{bmatrix}
    distance \\
    destination \\
    g−force \\
    collision \\
    jerk \\
    lane violation \\
    \end{bmatrix} 
    $$ 

The distance objective measures how close the vehicle is to the end destination and the
destination objective will either return one if the destination was reached and 0 if not. Instead
of enforcing a penalty for g-force, jerk and lane violation, we return the actual g-force and jerk
value and the distance margin of lane violation. As a general rule, we want to maximise the
distance and destination objectives and minimise the g-force, jerk, collision and lane violation
objectives

## Installation

```
git clone https://github.com/sofyanajridi/mo-deepdrive
pip install -e .
```

## Usage

```python
from deepdrive_zero.envs.variants import OneWaypointEnv, StaticObstacleEnv, IntersectionEnv
from deepdrive_zero.discrete_actions.comfortable_actions2 import COMFORTABLE_ACTIONS2

config = {"multi_objective": True,
          'discrete_actions': COMFORTABLE_ACTIONS2}
env = OneWaypointEnv(env_configuration=config, render_mode=None)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, vectorial_reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
```

## Documentation

Read the [documentation online]().

## Bugs and issues

- env.render() does not work anymore, this is because the original benchmark is calling a non-existing function from a library (pyglet) that has now been updated
    - The function that is causing this issue can be found in file _env.py_  line 261:
        ```
        pyglet.app.event_loop.has_exit = False
        pyglet.app.event_loop._legacy_setup() # This line is causing the issue. 
        pyglet.app.event_loop.run()
        pyglet.app.platform_event_loop.start()
        pyglet.app.event_loop.dispatch_event('on_enter')
        pyglet.app.event_loop.is_running = True
        ```
## Environment Parameters
Each environment can be further configured on the following properties:

<details><summary><a>Variables</a></summary>

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

</details>
 

## Acknowledgment
Deepdrive Zero: https://github.com/deepdrive/deepdrive-zero
