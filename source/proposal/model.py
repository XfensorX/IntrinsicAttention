from tkinter import N
from typing import Any, Dict, Optional

from gymnasium.envs.tabular.blackjack import obs
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

# For better Interpretabilty, could be more if it is not working
NUM_HEADS = 1


class IntrinsicAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        v_dim: int,
        qk_dim: int,
    ):
        super().__init__()

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=input_dim,
            kdim=qk_dim,
            vdim=v_dim,
            num_heads=NUM_HEADS,
            batch_first=True,
        )

        self.reward_layer = nn.Sequential(
            nn.Linear(v_dim, v_dim // 2),
            nn.ReLU(),
            nn.Linear(v_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        need_weights: bool = False,
    ):
        """
        !! Observations have to be padded to same length already
        input_dim should be some kind of mixture of actions and observations

        Args:
            inputs (torch.Tensor): shape BATCH x max_trajectory_length x input_dim
            actions (torch.Tensor): shape BATCH x max_trajectory_length x action_dim

        Return:
            rewards:
                shape: BATCH x max_trajectory_length x 1
            attn_weights: from the attn layer add the dimensions
                shape: BATCH x max_trajectory_length x max_trajectory_length
                 IF NEED_WEIGHTS is TRUE!!, else None
        """

        attn_out, attn_weights = self.attention_layer(inputs, need_weights=need_weights)
        rewards = self.reward_layer(attn_out).squeeze()

        return rewards, attn_weights


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

        obs_dim = self.model_config.get("attention_obs_dim")
        action_dim = self.model_config.get("action_dim")
        obs_embed_dim = self.model_config.get("attention_obs_embed_dim")
        self._gru_hidden_size = self.model_config.get("gru_hidden_size")

        obs_embed_layer_hidden_size = int((obs_dim + obs_embed_dim) / 2)
        self.observation_embedding_layer = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_layer_hidden_size),
            nn.ReLU(),
            nn.Linear(obs_embed_layer_hidden_size, obs_embed_dim),
            nn.ReLU(),
        )

        input_dim = action_dim + obs_embed_dim

        self.intrinsic_reward_network = IntrinsicAttention(
            input_dim=input_dim,
            v_dim=self.model_config.get("attention_v_dim"),
            qk_dim=self.model_config.get("attention_qk_dim"),
        )

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self._gru_hidden_size[0],
            num_layers=self.model_config.get("gru_num_layers"),
            batch_first=True,
        )

        self.gru_output_net = NotImplemented
        self.value_head = NotImplemented
        self.policy_head = NotImplemented

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
