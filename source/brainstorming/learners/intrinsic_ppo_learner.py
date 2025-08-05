from typing import Any, Dict

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_differentiable_learner import (
    TorchDifferentiableLearner,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID


class IntrinsicPPOLearner(PPOTorchLearner, TorchDifferentiableLearner):
    """PPO learner that incorporates intrinsic rewards"""

    @override(PPOTorchLearner)
    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main update method that handles intrinsic rewards before PPO update.

        Args:
            batch: The training batch to learn from.

        Returns:
            Dictionary with update metrics.
        """
        # Track the original rewards for later use
        self._original_rewards = batch["default_policy"][Columns.REWARDS].clone()

        # Process the batch to compute intrinsic rewards (if intrinsic module exists)
        if "intrinsic_reward_module" in self.module:
            # Get intrinsic rewards for the batch
            intrinsic_output = self.module["intrinsic_reward_module"].forward_train(
                batch
            )
            intrinsic_rewards = intrinsic_output[Columns.INTRINSIC_REWARDS]

            # Add intrinsic rewards to the extrinsic rewards
            intrinsic_coeff = self.config.learner_config_dict["intrinsic_reward_coeff"]
            batch["default_policy"][Columns.REWARDS] += (
                intrinsic_coeff * intrinsic_rewards
            )
        else:
            raise NotImplementedError("No intrinsic reward module available")

        # Run standard PPO update with the augmented rewards
        results = super().update(batch)

        # Restore original rewards
        batch["default_policy"][Columns.REWARDS] = self._original_rewards

        return results

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self, *, module_id: ModuleID, batch: Dict[str, Any], fwd_out: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute the standard PPO loss (intrinsic rewards have already been added)"""
        return super().compute_loss_for_module(
            module_id=module_id, batch=batch, fwd_out=fwd_out
        )
