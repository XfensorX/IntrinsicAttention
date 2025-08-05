from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from source.brainstorming.base_models.IntrinsicAttention import IntrinsicAttention

torch, nn = try_import_torch()


class IntrinsicAttentionModule(TorchRLModule):
    """Differentiable module for intrinsic attention rewards"""

    @override(TorchRLModule)
    def setup(self):
        # Extract configuration parameters
        input_dim = self.model_config.get("input_dim", 64)
        v_dim = self.model_config.get("attention_v_dim", 32)
        qk_dim = self.model_config.get("attention_qk_dim", 32)

        # Create the intrinsic attention module
        self.intrinsic_attention = IntrinsicAttention(
            input_dim=input_dim,
            v_dim=v_dim,
            qk_dim=qk_dim,
        )

        # Feature extraction for observations
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU(),
        )

    def compute_intrinsic_rewards(
        self, observations: TensorType
    ) -> Tuple[TensorType, TensorType]:
        """Compute intrinsic rewards using the attention mechanism"""
        # Encode observations
        encoded_obs = self.obs_encoder(observations)

        # Get intrinsic rewards from attention mechanism
        intrinsic_rewards, attention_weights = self.intrinsic_attention(encoded_obs)

        return intrinsic_rewards, attention_weights

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Get observations
        observations = batch[Columns.OBS]

        # Compute intrinsic rewards
        intrinsic_rewards, attention_weights = self.compute_intrinsic_rewards(
            observations
        )

        # Return results
        return {
            Columns.INTRINSIC_REWARDS: intrinsic_rewards,
            "attention_weights": attention_weights,
        }
