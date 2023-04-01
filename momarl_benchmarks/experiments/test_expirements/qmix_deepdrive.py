import gymnasium as gym
import ray
from ray.rllib.policy.policy import PolicySpec
from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper
from deepdrive_2d.deepdrive_zero.envs.variants import IntersectionWithGsAllowDecelEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
import numpy as np
from ray.rllib.algorithms.qmix import QMixConfig

from ray.tune.registry import register_env
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
)


def env_creator(env_config):
    return MultiAgentDeepdriveParallelWrapper(IntersectionWithGsAllowDecelEnv(env_config),vehicle_1_id=0,vehicle_2_id=1)


register_env("IntersectionEnv", env_creator)

obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
action_discrete_space = gym.spaces.Discrete(22)  # 16 or 22
action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)


def agent_to_policy_map(agent_id, episode, worker, **kwargs):
    if agent_id == 0:
        return "vehicle1_policy"
    elif agent_id == 1:
        return "vehicle2_policy"
    else:
        raise ValueError("Unknown agent type: ", agent_id)


ray.init()

config = (
    QMixConfig()
    .environment("IntersectionEnv", disable_env_checking=True, observation_space=obs_space,
                 action_space=action_discrete_space, render_env=True, env_config=env_config)
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=4)
    .framework("torch")
)

stop = {
    "timesteps_total": 2_000_000,
}

tuner = tune.Tuner(
    "QMIX",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1

                             ))

results = tuner.fit()
