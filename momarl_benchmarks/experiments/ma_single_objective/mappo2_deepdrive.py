import gymnasium as gym
import ray
from ray.rllib.policy.policy import PolicySpec

from momarl_benchmarks.wrappers.MultiAgentDeepdriveParallelWrapper import MultiAgentDeepdriveParallelWrapper
from deepdrive_2d.deepdrive_zero.envs.variants import IntersectionWithGsAllowDecelEnv
from deepdrive_2d.deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2

import numpy as np

from ray.tune.registry import register_env
from ray import air, tune

from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.air.integrations.wandb import WandbLoggerCallback

torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 29 + 29 + 22  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 80, activation_fn=nn.Tanh),
            SlimFC(80, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        input_ = torch.cat(
            [
                obs,
                opponent_obs,
                torch.nn.functional.one_hot(opponent_actions.long(), 22).float(),
            ],
            1,
        )
        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
        policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
            not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF

        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device
                ),
                convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_ACTION], policy.device
                ),
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            # Use super() to get to the base PPO policy.
            # This special loss function utilizes a shared
            # value function defined on self, and the loss function
            # defined on PPO policies.
            return loss_with_central_critic(
                self, super(), model, dist_class, train_batch
            )

        @override(base)
        def postprocess_trajectory(
                self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def stats_fn(self, train_batch: SampleBatch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTFPolicy


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )


class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CCPPOTorchPolicy


ModelCatalog.register_custom_model(
    "cc_model",
    TorchCentralizedCriticModel
)

env_config = dict(
    env_name='deepdrive-2d-intersection',
    is_intersection_map=True,
    discrete_actions=COMFORTABLE_ACTIONS2,
)


def env_creator(env_config):
    return MultiAgentDeepdriveParallelWrapper(IntersectionWithGsAllowDecelEnv(env_config), vehicle_1_id=0,
                                              vehicle_2_id=1)


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
    PPOConfig()
    .environment("IntersectionEnv", disable_env_checking=True, observation_space=obs_space,
                 action_space=action_discrete_space, env_config=env_config,render_env=True)
    .framework("torch")
    # Parallelize environment rollouts.
    .rollouts(batch_mode="complete_episodes",
              num_rollout_workers=0)
    .training(model={"custom_model": "cc_model"})
    .multi_agent(
        policies=
        {"vehicle1_policy": PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=obs_space,  # infer automatically from env
            action_space=action_discrete_space,
            config=PPOConfig.overrides(framework_str="torch")
            # infer automatically from env  # use main config plus <- this override here
        ),
            "vehicle2_policy": PolicySpec(
                policy_class=None,  # infer automatically from Algorithm
                observation_space=obs_space,  # infer automatically from env
                action_space=action_discrete_space,
                config=PPOConfig.overrides(framework_str="torch")
            )
        },
        policy_mapping_fn=agent_to_policy_map
    )
)

stop = {
    "timesteps_total": 2_000_000,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1

                             ,callbacks=[WandbLoggerCallback(project="momarl-benchmarks")]
                             , name="MAPPO_DeepDrive_IntersectionEnv"),

)

results = tuner.fit()
