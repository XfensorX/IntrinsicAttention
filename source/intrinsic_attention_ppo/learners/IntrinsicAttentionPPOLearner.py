"""
Learner implementation for PPO with intrinsic rewards on top of RLlib’s Torch stack.

IntrinsicAttentionPPOLearner:
        •	Computes gradients with torch.autograd.grad to support higher-order objectives.
        •	Integrates intrinsic rewards into PPO loss computation.
        •	Removes GAE from learner connectors when required and supports a one-timestep episode
augmentation used by certain connector pipelines.

"""

from typing import Any, Dict

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.torch.torch_differentiable_learner import (
    TorchDifferentiableLearner,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, NamedParamDict, TensorType

from source.intrinsic_attention_ppo.config import (
    INTRINSIC_REWARD_MODULE_ID,
    PPO_AGENT_POLICY_ID,
)
from source.intrinsic_attention_ppo.learners.CustomPPOLearner import CustomPPOLearner
from source.intrinsic_attention_ppo.learners.learner_utils.remove_gae_from_learner_connector import (
    remove_gae_from_learner_connectors,
)

torch, nn = try_import_torch()


class IntrinsicAttentionPPOLearner(
    TorchDifferentiableLearner, CustomPPOLearner
):  # This has to be in that order such that differentiableLearner is more important
    """
    Differentiable PPO learner with intrinsic rewards.

    This class combines TorchDifferentiableLearner and a project-specific CustomPPOLearner.
    The MRO is intentional: TorchDifferentiableLearner must precede CustomPPOLearner so
    differentiable capabilities (e.g., higher-order grads) take precedence.

    Notes
        •	Uses compute_losses (module-level loss dict) instead of compute_loss_for_module.
        •	Detects whether an “AddOneTsToEpisodesAndTruncate” connector is present and adapts loss
    computation accordingly.
    """

    @override(TorchDifferentiableLearner)
    def build(self, device) -> None:
        """
            Side Effects
            •	Calls PPOTorchLearner.build(self) to compensate for a missing super().build() in
        RLlib’s differentiable learner.
            •	Removes GAE from learner connectors
        """
        # Initialize the base learner
        super().build(device=device)
        PPOTorchLearner.build(
            self
        )  # as rllib forgot to call super.build() in Differential learner
        remove_gae_from_learner_connectors(self)
        self._custom_with_one_ts_to_episode = bool(
            "AddOneTsToEpisodesAndTruncate"
            in [str(x) for x in self._learner_connector.connectors]
        )

    @override(TorchDifferentiableLearner)
    def compute_gradients(
        self,
        loss_per_module: Dict[ModuleID, TensorType],
        params: Dict[ModuleID, NamedParamDict],
        **kwargs,
    ) -> Dict[ModuleID, NamedParamDict]:
        """
        Compute per-module named gradients for higher-order optimization.
        """
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
        """
        Compute PPO losses with intrinsic rewards and connector-aware options.
        """
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

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: AlgorithmConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        """
        This is an abstract method and has to be overridden, but we use "compute_losses" instead
        """
        raise NotImplementedError()
