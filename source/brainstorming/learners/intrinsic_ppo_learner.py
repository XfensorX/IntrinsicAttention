from typing import Any, Dict

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.torch.torch_differentiable_learner import (
    TorchDifferentiableLearner,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID


class IntrinsicPPOLearner(
    TorchDifferentiableLearner, PPOTorchLearner
):  # This has to be in that order such that differentiableLearner is more important
    """PPO learner that incorporates intrinsic rewards"""

    @override(TorchDifferentiableLearner)
    def build(self, device) -> None:
        # Initialize the base learner
        super().build(device=device)
        PPOTorchLearner.build(
            self
        )  # as rllib forgot to call super.build() in Differential learner

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
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: AlgorithmConfig,
        batch: Dict[ModuleID, Any],
        fwd_out: Dict[ModuleID, Any],
    ) -> TensorType:
        if module_id == PPO_AGENT_POLICY_ID:
            return PPOTorchLearner.compute_loss_for_module(
                self, module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )
        elif module_id == INTRINSIC_REWARD_MODULE_ID:
            return 0  # TODO: do we need a loss here actually? And if yes, which one? (PPO LOss does not make sense for intrinsic reward network)
        else:
            raise NotImplementedError("Do not know the used module_id: ", module_id)

    # @override(TorchDifferentiableLearner)
    # def _update(
    #     self,
    #     batch: Dict[ModuleID, Any],
    #     params: Dict[ModuleID, Dict[ModuleID, Any | Any]],
    # ) -> Tuple[Any, Dict[str, Dict[str, Any]], Any, Any]:
    #     print(f"IntrinsicPPOLearner _update: {batch=}")
    #
    #     return super()._update(batch, params)

    # @override(DifferentiableLearner)
    # def update(
    #     self,
    #     params: Dict[ModuleID, Dict[ModuleID, Any | Any]],
    #     training_data: TrainingData,
    #     *,
    #     _no_metrics_reduce: bool = False,
    #     **kwargs,
    # ) -> Tuple[Dict[ModuleID, Dict[ModuleID, Any | Any]] | Dict]:
    #     print(f"IntrinsicPPOLearner update: {training_data=}")
    #
    #     return super().update(
    #         params, training_data, _no_metrics_reduce=_no_metrics_reduce, **kwargs
    #     )


# TODO: we can postprocess the gradients here
