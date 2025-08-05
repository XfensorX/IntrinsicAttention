from typing import Any, Dict

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import DEFAULT_POLICY_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_differentiable_learner import (
    TorchDifferentiableLearner,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID


class IntrinsicPPOLearner(
    TorchDifferentiableLearner, PPOTorchLearner
):  # This has to be in that order such that differentiableLearner is more important
    """PPO learner that incorporates intrinsic rewards"""

    @override(TorchDifferentiableLearner)
    def update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main update method that handles intrinsic rewards before PPO update.

        Args:
            batch: The training batch to learn from.

        Returns:
            Dictionary with update metrics.
        """
        # Track the original rewards for later use
        self._original_rewards = batch[DEFAULT_POLICY_ID][Columns.REWARDS].clone()

        # Process the batch to compute intrinsic rewards (if intrinsic module exists)
        # Get intrinsic rewards for the batch
        intrinsic_output = self.module[
            INTRINSIC_REWARD_MODULE_ID
        ].forward_train(  # TODO: LUNA+PHILIPP : HAS TO BE NON-UPDATE CALL
            batch
        )
        intrinsic_rewards = intrinsic_output[Columns.INTRINSIC_REWARDS]

        # Add intrinsic rewards to the extrinsic rewards
        intrinsic_coeff = self.config.learner_config_dict["intrinsic_reward_coeff"]
        batch[DEFAULT_POLICY_ID][Columns.REWARDS] += intrinsic_coeff * intrinsic_rewards

        # Run standard PPO update with the augmented rewards
        results = super().update(
            params=batch
        )  # TODO: Is this correct there? does this actually just update the policy??

        # hallucinate update,

        # Restore original rewards
        batch[DEFAULT_POLICY_ID][Columns.REWARDS] = self._original_rewards
        # TODO: We guess there is the pseudo update missing here

        return results

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self, *, module_id: ModuleID, batch: Dict[str, Any], fwd_out: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute the standard PPO loss (intrinsic rewards have already been added)"""
        return super().compute_loss_for_module(
            module_id=module_id, batch=batch, fwd_out=fwd_out
        )


# TODO: we can postprocess the gradients here
