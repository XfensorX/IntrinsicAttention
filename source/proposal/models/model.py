from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from source.proposal.models.GRUBase import GRUBase
from source.proposal.models.IntrinsicAttention import IntrinsicAttention
from source.proposal.models.ReluMlp import ReluMlp

torch, nn = try_import_torch()


class IntrinsicAttentionPPOModel(TorchRLModule, ValueFunctionAPI):
    """
    Tips:
    - use Columns.???? instead of SAMPLEBATCH.??? because this is old API (ChatGPT os often wrong)
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    # This class contains significant contributions from:
    # https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py
    @override(TorchRLModule)
    def setup(self):  # TODO: Setup models
        # FIXME: This is a hack to make the env work with RLlib
        if len(self.action_space.shape) == 0:
            action_dim = 1
        else:
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
            input_dim=obs_embed_dim,
            hidden_size=self._gru_hidden_size,
            num_layers=self.model_config.get("gru_num_layers"),
            output_size=pre_head_embedding_dim,
        )

        self.value_head = ReluMlp(
            [pre_head_embedding_dim, pre_head_embedding_dim, 1], output_layer=None
        )

        self.policy_head = ReluMlp(
            [pre_head_embedding_dim, pre_head_embedding_dim, self.action_space.n],
            output_layer=None,
        )

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(
                shape=(
                    self.model_config["gru_num_layers"],
                    self._gru_hidden_size,
                ),
                dtype=np.float32,
            )
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embeddings, state_outs = self._compute_gru_embeddings_and_state_outs(
            batch[Columns.STATE_IN]["h"],
            self.compute_obs_action_embedding(batch, training=False),
        )
        action_logits = self.policy_head(embeddings)
        # TODO: Is squeezing correct?
        return {
            Columns.ACTION_DIST_INPUTS: action_logits.squeeze(
                -1
            ),  # this is the distribution, argmax is done automatically
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # episode_ids = batch[Columns.EPS_ID]

        obs_action_embedding = self.compute_obs_action_embedding(batch, training=False)

        gru_embeddings, state_out = self._compute_gru_embeddings_and_state_outs(
            batch[Columns.STATE_IN]["h"], obs_action_embedding
        )

        action_logits = self.policy_head(gru_embeddings)
        vf_preds = self.value_head(gru_embeddings).squeeze(-1)

        # TODO: Is squeezing correct?
        return {
            Columns.ACTION_DIST_INPUTS: action_logits.squeeze(-1),
            # Columns.INTRINSIC_REWARDS: intrinsic_rewards,
            Columns.STATE_OUT: state_out,
            Columns.EMBEDDINGS: gru_embeddings,
            Columns.VF_PREDS: vf_preds,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        # TODO: This error was raised
        # raise NotImplementedError()
        if embeddings is None:
            embeddings = batch.get(Columns.EMBEDDINGS)
        values = self.value_head(embeddings).squeeze(-1)
        return values

    def compute_obs_action_embedding(self, batch, training: bool = True) -> TensorType:
        obs = batch[Columns.OBS].squeeze(-1)  # [batch, obs_dim]
        obs_embed = self.observation_embedding_layer(obs.float())
        # FIXME: Action concationation only for attention layer!
        # TODO: Determine correct dimensions: squeeze obs or unsqueeze actions?
        # TODO: Use One-Hot encoding for observations?
        if training:
            actions = batch[Columns.ACTIONS]
            return torch.cat([obs_embed, actions.float()], dim=-1)
        else:
            return obs_embed

    def _compute_gru_embeddings_and_state_outs(self, h_in, obs_action_embedding):
        # TODO: Is this necessary? Rllib [batch, num_layers, hidden_size], GRU needs [num_layers, batch, hidden_size]
        h_in = h_in.permute(1, 0, 2)
        embeddings, h = self.gru_base_network(obs_action_embedding, h_in)
        return embeddings, {"h": h.permute(1, 0, 2)}  # [batch, num_layer, hidden_size]
