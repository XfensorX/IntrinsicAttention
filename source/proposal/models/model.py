from typing import Any, Dict, Optional

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

# FIXME: torch, nn = try_import_torch()
import torch
import torch.nn as nn

from source.proposal.models.GRUBase import GRUBase
from source.proposal.models.IntrinsicAttention import IntrinsicAttention
from source.proposal.models.ReluMlp import ReluMlp


class IntrinsicAttentionPPOModel(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    Tips:
    - use Columns.???? instead of SAMPLEBATCH.??? because this is old API (ChatGPT os often wrong)
    """

    # This class contains significant contributions from:
    # https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py
    @override(TorchRLModule)
    def setup(self):  # TODO: Setup models
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.model_config`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).

        example:
        - in_size = self.observation_space.shape[0]
        """

        action_dim = self.action_space.shape[0]

        obs_dim = self.model_config.get("obs_dim")
        obs_embed_dim = self.model_config.get("obs_embed_dim")
        pre_head_embedding_dim = self.model_config.get("pre_head_embedding_dim")

        self._gru_hidden_size = self.model_config.get("gru_hidden_size")

        self.observation_embedding_layer = ReluMlp(
            [obs_dim, int((obs_dim + obs_embed_dim) / 2), obs_embed_dim]
        )

        self.intrinsic_reward_network = IntrinsicAttention(
            input_dim=action_dim + obs_embed_dim,
            v_dim=self.model_config.get("attention_v_dim"),
            qk_dim=self.model_config.get("attention_qk_dim"),
        )

        self.gru_base_network = GRUBase(
            input_dim=action_dim + obs_embed_dim,
            hidden_size=self._gru_hidden_size,
            num_layers=self.model_config.get("gru_num_layers"),
            output_size=pre_head_embedding_dim,
        )

        self.value_head = ReluMlp(
            [pre_head_embedding_dim, pre_head_embedding_dim, 1], output_layer=None
        )

        self.policy_head = ReluMlp(
            [pre_head_embedding_dim, pre_head_embedding_dim, action_dim],
            output_layer=None,
        )

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {"h": np.zeros(shape=(self._gru_hidden_size,), dtype=np.float32)}

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):

        embeddings, state_outs = self._compute_gru_embeddings_and_state_outs(batch)
        action_logits = self.policy_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,  # this is the distribution, argmax is done automatically
            Columns.STATE_OUT: state_outs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):

        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings, state_outs = self._compute_gru_embeddings_and_state_outs(batch)

        action_logits = self.policy_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_gru_embeddings_and_state_outs(batch)
        values = self.value_head(embeddings).squeeze(-1)
        return values

    def compute_obs_action_embedding(self, batch):
        obs = batch[Columns.OBS]
        actions = batch[Columns.ACTIONS]

        obs_embed = self.observation_embedding_layer(obs)
        return torch.cat([obs_embed, actions], dim=-1)

    def _compute_gru_embeddings_and_state_outs(self, batch):

        obs_action_embedding = self.compute_obs_action_embedding(batch)

        state_in = batch[Columns.STATE_IN]
        h = state_in["h"]
        embeddings, h = self.gru(obs_action_embedding, h.unsqueeze(0))
        embeddings = self.gru_output_net(embeddings)
        return embeddings, {"h": h.squeeze(0)}

    def compute_intrinsic_rewards(self, batch, need_weights: bool = False):
        obs_action_embedding = self.compute_obs_action_embedding(batch)
        return self.intrinsic_reward_network(obs_action_embedding, need_weights)
