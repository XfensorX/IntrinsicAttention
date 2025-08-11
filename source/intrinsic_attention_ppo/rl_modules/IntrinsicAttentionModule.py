from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch.nn.functional as F
from pydantic import BaseModel, model_validator
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

# You can keep your existing ReluMlp util for convenience.
from intrinsic_attention_ppo.rl_modules.ReluMlp import ReluMlp
from source.intrinsic_attention_ppo.config import COL_ATTENTION_WEIGHTS, COL_EX_VF_PREDS

torch, nn = try_import_torch()


class BlockConfig(BaseModel):
    type: Literal["attention", "mlp"]
    hidden_sizes: list[int] | None = None

    @model_validator(mode="after")
    def check_hidden_sizes_present(self) -> "BlockConfig":
        if self.type == "mlp" and self.hidden_sizes is None:
            raise ValueError("Hidden Sizes must be provided for mlp")

        return self


class IntrinsicRewardNetworkConfig(BaseModel):
    encoder_hidden_sizes: list[int] | None
    encoding_dim: int
    num_heads: int
    layers: list[BlockConfig]
    head_hidden_sizes: list[int] | None


class IntrinsicAttentionModuleConfig(BaseModel):
    intrinsic_reward_network: IntrinsicRewardNetworkConfig
    extrinsic_value_hidden_layers: list[int]
    max_seq_len: int  # do make the episodes full
    vf_share_layers: bool


class _AttentionBlock(nn.Module):
    """Self-attention with residual + LayerNorm. Returns attn maps (B, H, T, T)."""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        B, T, _ = x.shape
        attn_out, attn_w = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        # attn_w can be (B, H, T, T) (new) or (B*H, T, T) (older torch). Normalize shape:
        if attn_w.dim() == 3 and attn_w.shape[0] == B * self.num_heads:
            attn_w = attn_w.view(B, self.num_heads, T, T)
        # Residual + LN
        x = self.ln(x + attn_out)
        return x, attn_w


class IntrinsicAttentionModule(TorchRLModule, ValueFunctionAPI):
    """
    Inputs: Columns.OBS, Columns.ACTIONS, Columns.REWARDS.
    """

    @override(TorchRLModule)
    def setup(self):
        cfg = IntrinsicAttentionModuleConfig.model_validate(self.model_config)
        irn = cfg.intrinsic_reward_network
        num_heads = irn.num_heads

        obs_dim = int(self.observation_space.shape[0])
        self.act_dim = self.action_space.n

        rew_dim = 1  # scalar reward per step
        self.concat_dim = obs_dim + self.act_dim + rew_dim
        if irn.encoder_hidden_sizes is None:
            self.encoder = None
            self.model_dim = self.concat_dim
        else:
            self.model_dim = int(irn.encoding_dim)
            self.encoder = ReluMlp(
                hidden_layers=irn.encoder_hidden_sizes,
                input_size=self.concat_dim,
                output_size=self.model_dim,
                output_layer=None,
            )

        if self.model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({num_heads}). "
                f"Either adjust encoder_hidden_sizes (last hidden) or num_heads."
            )

        self.layers = nn.ModuleList()
        self.attn_indices: List[int] = []
        for i, block in enumerate(irn.layers):
            if block.type == "attention":
                blk = _AttentionBlock(self.model_dim, num_heads)
                self.attn_indices.append(len(self.layers))
                self.layers.append(blk)
            elif block.type == "mlp":
                blk = ReluMlp(
                    input_size=self.model_dim,
                    output_size=self.model_dim,
                    hidden_layers=block.hidden_sizes,
                )
                self.layers.append(blk)

        if irn.head_hidden_sizes is None:
            if self.model_dim != 1:
                raise ValueError(
                    f"The last attention block will output in size {self.model_dim} But you did"
                    f" not specify a intrinsic_reward_head. We need output size 1."
                    f"If you want a direct linear layer from model_did/encoding_dim to 1, just use an empty list as hidden_sizes for "
                    f"config.intrinsic_reward_network.head_hidden_sizes"
                )
            else:
                self.intrinsic_reward_head = None
        else:
            self.intrinsic_reward_head = ReluMlp(
                hidden_layers=irn.head_hidden_sizes,
                input_size=self.model_dim,
                output_size=1,
                output_layer=None,
            )

        self.extrinsic_value_network = ReluMlp(
            hidden_layers=cfg.extrinsic_value_hidden_layers,
            input_size=obs_dim,
            output_size=1,
            output_layer=None,
        )

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        # Keep full episodes
        return {
            "h": np.zeros(
                1,
            )
        }

    # ------- helpers -------
    def _ensure_bt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # (B, D) -> (B, D, 1)
            x = x.unsqueeze(2)
        return x

    def _stack_inputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs = self._ensure_bt(batch[Columns.OBS].float())  # (B, T, obs_dim)

        acts = F.one_hot(
            batch[Columns.ACTIONS].long(), num_classes=self.act_dim
        ).float()
        acts = self._ensure_bt(acts)
        rews = self._ensure_bt(batch[Columns.REWARDS].float())

        x = torch.cat([obs, acts, rews], dim=-1)  # (B, T, concat_dim)
        return x

    def _run_layers(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, attn_stack) with attn_stack shape (B, L_attn, H, T, T)."""
        attn_maps: List[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, _AttentionBlock):
                x, w = layer(x)  # w: (B, H, T, T)
                attn_maps.append(w)
            else:
                x = layer(x)

        if attn_maps:
            attn_stack = torch.stack(attn_maps, dim=1)  # (B, L_attn, H, T, T)
        else:
            B, T, _ = x.shape
            attn_stack = x.new_zeros(
                (x.shape[0], 0, 0, T, T)
            )  # empty if no attention layers
        return x, attn_stack

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        stacked_x = self._stack_inputs(batch)

        embedding = self.encoder(stacked_x) if self.encoder is not None else stacked_x

        attention_out, attn_stack = self._run_layers(embedding)

        if self.intrinsic_reward_head is not None:
            intrinsic_rewards = self.intrinsic_reward_head(attention_out).squeeze(-1)
        else:
            intrinsic_rewards = attention_out.squeeze(-1)

        extrinsic_values = self.extrinsic_value_network(
            self._ensure_bt(batch[Columns.OBS].float())
        ).squeeze(-1)
        return {
            Columns.INTRINSIC_REWARDS: intrinsic_rewards,  # (B, T)
            COL_EX_VF_PREDS: extrinsic_values,  # (B, T)
            COL_ATTENTION_WEIGHTS: attn_stack,  # (B, L_attn, H, T, T)
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError("Training-only module. Use `_forward_train()`.")

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        raise ValueError("This cannot be called!")
