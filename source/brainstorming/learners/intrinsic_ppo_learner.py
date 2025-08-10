from typing import Any, Dict

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.torch.torch_differentiable_learner import (
    TorchDifferentiableLearner,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, NamedParamDict, TensorType

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID
from source.brainstorming.learners.CustomPPOLearner import CustomPPOLearner
from source.brainstorming.learners.IntrinsicRewardLearnerConnector import (
    remove_gae_from_learner_connectors,
)

torch, nn = try_import_torch()


class IntrinsicPPOLearner(
    TorchDifferentiableLearner, CustomPPOLearner
):  # This has to be in that order such that differentiableLearner is more important
    """PPO learner that incorporates intrinsic rewards"""

    custom_use_ppo_torch_learner: bool = False

    @override(TorchDifferentiableLearner)
    def build(self, device) -> None:
        # Initialize the base learner
        super().build(device=device)
        PPOTorchLearner.build(
            self
        )  # as rllib forgot to call super.build() in Differential learner
        remove_gae_from_learner_connectors(self)
        print(f"PPO Learner {self._learner_connector=}")
        self._custom_with_one_ts_to_episode = bool(
            "AddOneTsToEpisodesAndTruncate"
            in [str(x) for x in self._learner_connector.connectors]
        )

    # @override(TorchDifferentiableLearner)
    # def update(
    #     self,
    #     params: Dict[ModuleID, NamedParamDict],
    #     training_data: TrainingData
    # ) -> Dict[str, Any]:
    #     """
    #     Main update method that handles intrinsic rewards before PPO update.

    #     Args:
    #         batch: The training batch to learn from.

    #     Returns:
    #         Dictionary with update metrics.
    #     """
    #     # Track the original rewards for later use
    #     self._original_rewards = batch[DEFAULT_POLICY_ID][Columns.REWARDS].clone()

    #     # Process the batch to compute intrinsic rewards (if intrinsic module exists)
    #     # Get intrinsic rewards for the batch
    #     intrinsic_output = self.module[
    #         INTRINSIC_REWARD_MODULE_ID
    #     ].forward_train(  # TODO: LUNA+PHILIPP : HAS TO BE NON-UPDATE CALL
    #         batch
    #     )
    #     intrinsic_rewards = intrinsic_output[Columns.INTRINSIC_REWARDS]

    #     # Add intrinsic rewards to the extrinsic rewards
    #     intrinsic_coeff = self.config.learner_config_dict["intrinsic_reward_coeff"]
    #     batch[DEFAULT_POLICY_ID][Columns.REWARDS] += intrinsic_coeff * intrinsic_rewards

    #     # Run standard PPO update with the augmented rewards
    #     results = super().update(
    #         params=batch
    #     )  # TODO: Is this correct there? does this actually just update the policy??

    #     # hallucinate update,

    #     # Restore original rewards
    #     batch[DEFAULT_POLICY_ID][Columns.REWARDS] = self._original_rewards
    #     # TODO: We guess there is the pseudo update missing here

    #     return results
    @override(TorchDifferentiableLearner)
    def compute_gradients(
        self,
        loss_per_module: Dict[ModuleID, TensorType],
        params: Dict[ModuleID, NamedParamDict],
        **kwargs,
    ) -> Dict[ModuleID, NamedParamDict]:
        # TODO (simon): Add grad scalers later.
        total_loss = sum(loss_per_module.values())

        grads = torch.autograd.grad(
            total_loss,
            sum((list(param.values()) for mid, param in params.items()), []),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        # Map all gradients to their keys.
        grads_list = list(grads)
        offset = 0
        named_grads = {}
        for module_id, module_params in params.items():
            n = len(module_params)
            module_slice = grads_list[offset : offset + n]
            named_grads[module_id] = {
                name: g for (name, _), g in zip(module_params.items(), module_slice)
            }
            offset += n

        return named_grads

    @override(TorchDifferentiableLearner)
    def compute_losses(
        self, *, fwd_out: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        loss = self.compute_ppo_loss(
            config=self.config.get_config_for_module(PPO_AGENT_POLICY_ID),
            batch=batch,
            fwd_out=fwd_out,
            use_intrinsic_rewards=True,
            with_one_ts_to_episode=self._custom_with_one_ts_to_episode,
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )

        return {
            PPO_AGENT_POLICY_ID: loss,
            INTRINSIC_REWARD_MODULE_ID: torch.tensor(0.0, device=loss.device),
        }

    @override(TorchDifferentiableLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: "AlgorithmConfig",
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        raise NotImplementedError()
