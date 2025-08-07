from typing import Any, Dict, List

import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import DEFAULT_POLICY_ID
from ray.rllib.core.learner.torch.torch_meta_learner import TorchMetaLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType

from brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID


class IntrinsicAttentionMetaLearner(TorchMetaLearner, PPOTorchLearner):
    """Meta-learner for updating the intrinsic reward network"""

    @override(TorchMetaLearner)
    def build(self) -> None:
        """Build the meta-learner with a proper connector pipeline."""
        # Initialize the base learner
        super().build()

    @override(TorchMetaLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: AlgorithmConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
        others_loss_per_module: List[Dict[ModuleID, TensorType]] = None,
    ) -> TensorType:
        """
        Compute meta-loss for optimizing the intrinsic reward network.

        The meta-objective is to minimize the PPO policy's loss, which indicates
        that the intrinsic rewards are helping the agent learn better.
        """
        if module_id == PPO_AGENT_POLICY_ID:
            return PPOTorchLearner.compute_loss_for_module(
                self, module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )
        elif module_id == INTRINSIC_REWARD_MODULE_ID:
            return 0  # TODO: do we need a loss here actually? And if yes, which one? (PPO LOss does not make sense for intrinsic reward network)
        else:
            raise NotImplementedError("Do not know the used module_id: ", module_id)

        # TODO: LUAN+PHILIPP: !!! HERE CALCULATE LOSS ONLY WITH EXTRINSIC REWARDS => This happens automatically, becuse the rewards are reset to original during the differential Learner update function
        # But this is the wrong loss, we need to calculate it from fwd_out
        # Get the PPO policy's performance metrics
        if others_loss_per_module is None or not others_loss_per_module:
            return torch.tensor(0.0, device=self._device)

        # TODO: This is prprbably supposed to be calculated again
        # Extract PPO policy's performance - typically the first module's loss
        ppo_loss = others_loss_per_module[0].get(
            DEFAULT_POLICY_ID, torch.tensor(0.0, device=self._device)
        )

        # Meta-objective: minimize PPO loss
        meta_loss = ppo_loss  # !! TODO: Eventually just use extrinsic reward here
        raise NotImplementedError()

        # TODO:  maybe we need this later:

        # Optional: Add regularization terms

        # # 1. Sparsity regularization to keep intrinsic rewards small
        # intrinsic_rewards = fwd_out.get(Columns.INTRINSIC_REWARDS, None)
        # if intrinsic_rewards is not None:
        #     sparsity_reg = torch.mean(torch.abs(intrinsic_rewards))
        #     sparsity_weight = self.config.learner_config_dict.get(
        #         "sparsity_weight", 0.01
        #     )
        #     meta_loss = meta_loss + sparsity_weight * sparsity_reg

        # # 2. Entropy regularization on attention weights
        # attention_weights = fwd_out.get("attention_weights", None)
        # if attention_weights is not None:
        #     # Calculate entropy of attention weights
        #     entropy = -torch.sum(
        #         attention_weights * torch.log(attention_weights + 1e-10), dim=-1
        #     ).mean()
        #     entropy_weight = self.config.learner_config_dict.get(
        #         "entropy_weight", 0.001
        #     )
        #     # We want to maximize entropy (diversity), so subtract
        #     meta_loss = meta_loss - entropy_weight * entropy

        return meta_loss
