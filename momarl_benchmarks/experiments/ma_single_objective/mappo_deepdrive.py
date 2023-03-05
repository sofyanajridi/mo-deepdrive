import gymnasium as gym
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict, Discrete
import argparse
import os
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
)

def env_creator(env_config):
    return MultiAgentDeepdriveParallelWrapper(IntersectionWithGsAllowDecelEnv(env_config), vehicle_1_id=0,
                                              vehicle_2_id=1)


register_env("IntersectionEnv", env_creator)


class YetAnotherTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.
    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        print(obs_space)
        print(action_space)
        self.action_model = TorchFC(
            Box(low=-np.inf, high=np.inf, shape=(29,)),  # one-hot encoded Discrete(6)
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )

        print(self.action_model)

        self.value_model = TorchFC(
            obs_space, action_space, 1, model_config, name + "_vf"
        )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # Store model-input for possible `value_function()` call.
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2]
        )
        return torch.reshape(value_out, [-1])






class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""


    def on_postprocess_trajectory(
            self,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(22))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]

        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, -22:] = opponent_actions


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": 0,  # filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": 0,  # filled in by FillInActions
        },
    }


    return new_obs


ModelCatalog.register_custom_model(
    "cc_model",
    YetAnotherTorchCentralizedCriticModel
)

obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
action_space = gym.spaces.Discrete(22)  # 16 or 22

observer_space = Dict(
        {
            "own_obs": obs_space,
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": obs_space,
            "opponent_action": action_space,
        }
    )




def agent_to_policy_map(agent_id, episode, worker, **kwargs):
    if agent_id == 0:
        return "vehicle1_policy"
    elif agent_id == 1:
        return "vehicle2_policy"
    else:
        raise ValueError("Unknown agent type: ", agent_id)


ray.init()

config = (
    PPOConfig()
    .environment("IntersectionEnv", disable_env_checking=True, observation_space=obs_space,
                 action_space=action_space, env_config=env_config)
    .framework("torch")
    # Parallelize environment rollouts.
    .rollouts(enable_connectors=False, batch_mode="complete_episodes",
            num_rollout_workers=0,)
    .callbacks(FillInActions)
    .training(model={"custom_model": "cc_model"})
    .multi_agent(
        policies=
        {"vehicle1_policy": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=observer_space,  # infer automatically from env
            action_space=action_space,
            # infer automatically from env  # use main config plus <- this override here
        ),
            "vehicle2_policy": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=observer_space,  # infer automatically from env
                action_space=action_space,  # infer automatically from env
            )
        },
        policy_mapping_fn=agent_to_policy_map,
        observation_fn=central_critic_observer)
)

stop = {
    "timesteps_total": 20_000_000,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1

                             # ,callbacks=[WandbLoggerCallback(project="momarl-benchmarks")]
                             , name="MAPPO_DeepDrive_IntersectionEnv"),

)

results = tuner.fit()
