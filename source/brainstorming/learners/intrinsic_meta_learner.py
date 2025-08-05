from typing import Any, Dict, List, Optional

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_meta_learner import TorchMetaLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID


class IntrinsicAttentionMetaLearner(TorchMetaLearner):
    """Meta-learner for updating the intrinsic reward network"""

    @override(TorchMetaLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
        others_loss_per_module: Optional[List[Dict[ModuleID, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Compute meta-loss for optimizing the intrinsic reward network.

        The meta-objective is to minimize the PPO policy's loss, which indicates
        that the intrinsic rewards are helping the agent learn better.
        """
        # Get the PPO policy's performance metrics
        if others_loss_per_module is None or not others_loss_per_module:
            return torch.tensor(0.0, device=self._device)

        # Extract PPO policy's performance - typically the first module's loss
        ppo_loss = others_loss_per_module[0].get(
            "default_policy", torch.tensor(0.0, device=self._device)
        )

        # Meta-objective: minimize PPO loss
        meta_loss = ppo_loss

        # Optional: Add regularization terms

        # 1. Sparsity regularization to keep intrinsic rewards small
        intrinsic_rewards = fwd_out.get(Columns.INTRINSIC_REWARDS, None)
        if intrinsic_rewards is not None:
            sparsity_reg = torch.mean(torch.abs(intrinsic_rewards))
            sparsity_weight = self.config.learner_config_dict.get(
                "sparsity_weight", 0.01
            )
            meta_loss = meta_loss + sparsity_weight * sparsity_reg

        # 2. Entropy regularization on attention weights
        attention_weights = fwd_out.get("attention_weights", None)
        if attention_weights is not None:
            # Calculate entropy of attention weights
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10), dim=-1
            ).mean()
            entropy_weight = self.config.learner_config_dict.get(
                "entropy_weight", 0.001
            )
            # We want to maximize entropy (diversity), so subtract
            meta_loss = meta_loss - entropy_weight * entropy

        return meta_loss
