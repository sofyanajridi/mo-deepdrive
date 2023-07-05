# mo-deepdrive


Self driving car benchmark based on the original DeepDrive simulator with added support for multi-objective RL.



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

## Acknowledgment
Deepdrive Zero: https://github.com/deepdrive/deepdrive-zero
