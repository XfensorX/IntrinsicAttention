from typing import Any, Dict, Optional

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from source.brainstorming.base_models.IntrinsicAttention import IntrinsicAttention

torch, nn = try_import_torch()


# TODO: Actually Use Attention correctly here
class IntrinsicAttentionModule(TorchRLModule, ValueFunctionAPI):
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

        # self.compute_extrinsic_values = ReluMlp(
        #     [input_dim, input_dim // 2, 1], output_layer=None
        # )

        # Feature extraction for observations
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU(),
        )

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Get observations

        encoded_obs = self.obs_encoder(batch[Columns.OBS])

        # Get intrinsic rewards from attention mechanism
        intrinsic_rewards, attention_weights = self.intrinsic_attention(encoded_obs)

        # extrinsic_values = self.compute_extrinsic_values(encoded_obs)

        return {
            Columns.INTRINSIC_REWARDS: intrinsic_rewards,
            # COL_EX_VF_PREDS: extrinsic_values,
            "attention_weights": attention_weights,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        raise ValueError("This cannot be called !")

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError(
            "`IntrinsicCuriosityModel` should only be used for training! "
            "Only calls to `forward_train()` supported."
        )
