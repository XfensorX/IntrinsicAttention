"""
Torch/RLlib module for intrinsic-attention PPO-style training.

The module consumes per-timestep tuples of observation, action, and reward,
encodes them (optionally via an MLP), applies a configurable stack of
self-attention and MLP blocks, and produces:
• intrinsic reward predictions (B, T),
• an extrinsic value function baseline from observations (B, T), and
• attention maps from all attention layers (B, L_attn, H, T, T).

Configuration is defined via Pydantic models:
• BlockConfig — selects “attention” or “mlp” blocks and their sizes,
• IntrinsicRewardNetworkConfig — encoder, attention heads, stack layout, head,
• IntrinsicAttentionModuleConfig — top-level module wiring.

This is a training-only module; inference APIs that assume policy/value heads are
not supported.
"""

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

from source.intrinsic_attention_ppo.config import COL_ATTENTION_WEIGHTS, COL_EX_VF_PREDS

# You can keep your existing ReluMlp util for convenience.
from source.intrinsic_attention_ppo.rl_modules.ReluMlp import ReluMlp

torch, nn = try_import_torch()


class BlockConfig(BaseModel):
    """
    Configuration for a single network block.

    Attributes:
    type: Either “attention” or “mlp”.
    hidden_sizes: Hidden layer sizes for an MLP block. Must be provided
    when type == “mlp”; unused for attention blocks.
    """

    type: Literal["attention", "mlp"]
    hidden_sizes: list[int] | None = None

    @model_validator(mode="after")
    def check_hidden_sizes_present(self) -> "BlockConfig":
        """
        Validate that MLP blocks define hidden_sizes.

        Raises:
        ValueError: If type is “mlp” and hidden_sizes is None.

        Returns:
        The validated BlockConfig instance (self).
        """
        if self.type == "mlp" and self.hidden_sizes is None:
            raise ValueError("Hidden Sizes must be provided for mlp")

        return self


class IntrinsicRewardNetworkConfig(BaseModel):
    """
    Configuration for the intrinsic reward subnetwork.

    Attributes:
    encoder_hidden_sizes: Hidden sizes for an optional encoder MLP that
    projects concatenated (obs, one-hot action, reward) into encoding_dim.
    If None, the concatenated input is used directly.

    encoding_dim: Output feature size of the encoder MLP (model dimension).

    num_heads: Number of attention heads used in attention blocks.

    layers: Ordered list of network blocks to apply (“attention” or “mlp”).

    head_hidden_sizes: Hidden sizes for the intrinsic reward head MLP that

    maps the model dimension to a scalar. If None, the model dimension
    must already be 1 and no head is applied.
    """

    encoder_hidden_sizes: list[int] | None
    encoding_dim: int
    num_heads: int
    layers: list[BlockConfig]
    head_hidden_sizes: list[int] | None


class IntrinsicAttentionModuleConfig(BaseModel):
    """
    Top-level module configuration.

    Attributes:
    intrinsic_reward_network: Configuration of the intrinsic reward network.
    extrinsic_value_hidden_layers: Hidden sizes for the extrinsic value
    network that maps observations to scalar values.
    max_seq_len: Maximum sequence length; used by callers to keep full episodes.
    vf_share_layers: If True, share layers between policy and value; unused here
    but kept for interface parity with other RLlib modules.
    """

    intrinsic_reward_network: IntrinsicRewardNetworkConfig
    extrinsic_value_hidden_layers: list[int]
    max_seq_len: int  # do make the episodes full
    vf_share_layers: bool


class _AttentionBlock(nn.Module):
    """
    Multi-head self-attention block with residual connection and LayerNorm.

    Applies nn.MultiheadAttention in batch-first mode over (B, T, D) inputs and
    returns the transformed sequence and attention weights normalized to shape
    (B, H, T-out, T-in) across PyTorch versions.
    """

    def __init__(self, model_dim: int, num_heads: int):
        """
        Args:
        model_dim: Feature dimension D of the input and output.
        num_heads: Number of attention heads H.

        Initializes MultiheadAttention (batch_first=True) and a LayerNorm over D.
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run self-attention with residual + LayerNorm.

        Args:
        x: Input tensor of shape (B, T, D).

        Returns:
        A tuple (y, attn_w):
        y: Output tensor of shape (B, T, D).
        attn_w: Attention weights of shape (B, H, T, T).

        Notes:
        Handles PyTorch variants that return attention weights as (B*H, T, T) by
        reshaping to (B, H, T, T).
        """
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
    RLlib Torch module that predicts intrinsic rewards from sequences and an
    extrinsic value baseline from observations.

    Inputs (by column):
    Columns.OBS: Observations with shape (B, T, obs_dim) or (B, obs_dim).
    Columns.ACTIONS: Discrete actions with shape (B, T) or (B,) (will be one-hot).
    Columns.REWARDS: Scalar rewards with shape (B, T) or (B,).

    Outputs (in training forward):
    Columns.INTRINSIC_REWARDS: Predicted intrinsic rewards (B, T).
    COL_EX_VF_PREDS: Extrinsic value predictions (B, T).
    COL_ATTENTION_WEIGHTS: Stacked attention maps (B, L_attn, H, T, T).

    Notes:
    • The intrinsic pipeline optionally encodes inputs to encoding_dim, runs a
    user-defined stack of attention/MLP blocks, and applies an optional head.
    • model_dim must be divisible by num_heads.
    • This module is training-only; _forward and compute_values are disabled.
    """

    @override(TorchRLModule)
    def setup(self):
        """
        Build submodules from configuration.

        Parses IntrinsicAttentionModuleConfig and constructs:
        • optional encoder MLP (or identity if encoder_hidden_sizes is None),
        • a sequence of attention/MLP blocks,
        • an intrinsic reward head (optional) that outputs a scalar,
        • an extrinsic value network over observations.

        Raises:
        ValueError: If model dimension is not divisible by the number of heads,
        or if no intrinsic head is provided while model_dim != 1.
        """
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
        """
        This is only needed to prevent RLlib running into an error while using stateful episodes without stateful models.
        """
        # Keep full episodes
        return {
            "h": np.zeros(
                1,
            )
        }

    # ------- helpers -------
    def _ensure_bt(self, x: torch.Tensor) -> torch.Tensor:
        """
        ensures time dimension exists.
        """
        if x.dim() == 2:  # (B, D) -> (B, D, 1)
            x = x.unsqueeze(2)
        return x

    def _stack_inputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Assemble the intrinsic model input from batch columns.

        Steps:
        1) Cast observations to float and ensure a time dimension.
        2) One-hot encode discrete actions to size act_dim and ensure a time dim.
        3) Cast rewards to float and ensure a time dim.
        4) Concatenate along the feature axis to shape (B, T, obs_dim + act_dim + 1).

        Returns:
        Tensor of shape (B, T, concat_dim).
        """
        obs = self._ensure_bt(batch[Columns.OBS].float())  # (B, T, obs_dim)

        acts = F.one_hot(
            batch[Columns.ACTIONS].long(), num_classes=self.act_dim
        ).float()
        acts = self._ensure_bt(acts)
        rews = self._ensure_bt(batch[Columns.REWARDS].float())

        x = torch.cat([obs, acts, rews], dim=-1)  # (B, T, concat_dim)
        return x

    def _run_layers(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x.shape: (B, T, D).

        Returns (x, attn_stack) with attn_stack shape (B, L_attn, H, T, T)."""
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
        """
        Returns Dict with:
        Columns.INTRINSIC_REWARDS: Tensor (B, T).
        COL_EX_VF_PREDS: Tensor (B, T).
        COL_ATTENTION_WEIGHTS: Tensor (B, L_attn, H, T, T).th:
        """
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
        """
        Cannot be used, as the compute_values calls from rllib are not functional and
        therefore destroy differentiability for second order gradients
        """
        raise ValueError("This cannot be called!")
