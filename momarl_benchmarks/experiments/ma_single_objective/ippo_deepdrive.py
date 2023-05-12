import gymnasium as gym
import ray
from ray.rllib.policy.policy import PolicySpec

from momarl_benchmarks.wrappers.v26ToV21Wrapper import V26toV21Wrapper
from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper
from deepdrive_2d.deepdrive_zero.envs.variants import IntersectionWithGsAllowDecelEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from deepdrive_2d.deepdrive_zero.constants import COMFORTABLE_STEERING_ACTIONS, \
    COMFORTABLE_ACTIONS
# import sys
# sys.modules["gym"] = gym
# #
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
    jerk_penalty_coeff=0.10,
    gforce_penalty_coeff=0.031,
    lane_penalty_coeff=0.02,
    collision_penalty_coeff=0.31,
    speed_reward_coeff=0.50,
    incent_win=True
)


def env_creator(env_config):
    return MultiAgentDeepdriveParallelWrapper(IntersectionWithGsAllowDecelEnv(env_config))


register_env("IntersectionEnv", env_creator)

obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
action_discrete_space = gym.spaces.Discrete(22)  # 16 or 22
action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)


def agent_to_policy_map(agent_id, episode, worker, **kwargs):
    if agent_id == "vehicle1":
        return "vehicle1_policy"
    elif agent_id == "vehicle2":
        return "vehicle2_policy"
    else:
        raise ValueError("Unknown agent type: ", agent_id)


ray.init()

config = (
    PPOConfig()
    .environment("IntersectionEnv", disable_env_checking=True, observation_space=obs_space,
                 action_space=action_discrete_space, env_config=env_config)
    .framework("torch")
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=4, enable_connectors=False)
    .multi_agent(
        policies=
        {"vehicle1_policy": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=obs_space,  # infer automatically from env
            action_space=action_discrete_space,
            # infer automatically from env  # use main config plus <- this override here
        ),
            "vehicle2_policy": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=obs_space,  # infer automatically from env
                action_space=action_discrete_space,  # infer automatically from env
            )
        }, policy_mapping_fn=agent_to_policy_map)
)

stop = {
    "timesteps_total": 2_000_000,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1

                              , callbacks=[WandbLoggerCallback(project="momarl-benchmarks")]
                             , name="IPPO_DeepDrive_IntersectionEnv"),

)

results = tuner.fit()
