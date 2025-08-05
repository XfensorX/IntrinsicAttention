from typing import Any, Dict, Optional

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from source.brainstorming.models.GRUBase import GRUBase
from source.brainstorming.models.ReluMlp import ReluMlp

torch, nn = try_import_torch()


class DifferentiableIntrinsicAttentionPPOModel(TorchRLModule, ValueFunctionAPI):
    """PPO model with GRU for temporal context processing"""

    @override(TorchRLModule)
    def setup(self):
        # Extract model configuration
        obs_embed_dim = self.model_config.get("obs_embed_dim", 64)
        hidden_size = self.model_config.get("gru_hidden_size", 256)
        num_layers = self.model_config.get("gru_num_layers", 2)
        pre_head_dim = self.model_config.get("pre_head_embedding_dim", 256)

        # Observation embedding
        input_dim = self.observation_space.shape[0]
        self.obs_embedding = ReluMlp([input_dim, obs_embed_dim * 2, obs_embed_dim])

        # GRU core
        self.gru = GRUBase(
            input_dim=obs_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=pre_head_dim,
        )

        # Policy head (action distribution)
        action_shape = (
            self.action_space.n
            if hasattr(self.action_space, "n")
            else self.action_space.shape[0]
        )
        self.policy_head = ReluMlp(
            [pre_head_dim, pre_head_dim // 2, action_shape], output_layer=None
        )

        # Value function head
        self.value_head = ReluMlp(
            [pre_head_dim, pre_head_dim // 2, 1], output_layer=None
        )

        # Initial state
        self.register_state(
            "hidden_state", lambda: torch.zeros(num_layers, 1, hidden_size)
        )

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return self.get_state("hidden_state")

    def _process_observations(self, observations, state, seq_lens=None):
        # Embed observations
        obs_embeds = self.obs_embedding(observations)

        # Process through GRU
        gru_out, new_state = self.gru(
            obs_embeds.unsqueeze(1) if len(obs_embeds.shape) == 2 else obs_embeds, state
        )

        return gru_out.squeeze(1) if len(gru_out.shape) == 3 and gru_out.shape[
            1
        ] == 1 else gru_out, new_state

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Extract observations and state
        observations = batch[Columns.OBS]
        state = batch.get(Columns.STATE, self.get_state("hidden_state"))

        # Process through GRU
        gru_out, new_state = self._process_observations(observations, state)

        # Get action logits and values
        action_logits = self.policy_head(gru_out)
        values = self.value_head(gru_out).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: values,
            Columns.STATE_OUT: new_state,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        forward_out = self.forward_train(batch)
        return forward_out[Columns.VF_PREDS]
