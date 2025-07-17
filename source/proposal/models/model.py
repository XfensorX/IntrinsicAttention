from typing import Any, Dict, Optional

import time

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
    def setup(self):
        self.action_dim = self.action_space.n

        obs_dim = self.observation_space.shape[0]
        obs_embed_dim = self.model_config.get("obs_embed_dim")
        pre_head_embedding_dim = self.model_config.get("pre_head_embedding_dim")

        self._gru_hidden_size = self.model_config.get("gru_hidden_size")

        self.observation_embedding_layer = ReluMlp(
            [obs_dim, int((obs_dim + obs_embed_dim) / 2), obs_embed_dim]
        )

        self.intrinsic_reward_network = IntrinsicAttention(
            input_dim=self.action_dim + obs_embed_dim,
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
            [pre_head_embedding_dim, pre_head_embedding_dim, self.action_dim],
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
    @torch.no_grad()
    def _forward(self, batch, **kwargs):
        """
        :param batch: Description
            batch[Columns.OBS].shape == (batch, time, obs_dim)
        """

        embeddings, state_outs = self._compute_gru_embeddings_and_state_outs(
            batch[Columns.STATE_IN]["h"],
            self.observation_embedding_layer(batch[Columns.OBS]),
        )

        action_logits = self.policy_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,  # this is the distribution, action sampling is done automatically
            Columns.STATE_OUT: {"h": state_outs},
            Columns.EMBEDDINGS: embeddings,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        if not hasattr(self, "_last_print_time"):
            self._last_print_time = 0
        now = time.time()
        if now - self._last_print_time > 2:
            print(
                f"{batch[Columns.OBS].shape=}, {batch[Columns.SEQ_LENS]=}, {torch.unique(batch[Columns.EPS_ID])=}"
            )
            self._last_print_time = now

        # Ensure, that all steps are from one episode
        if 0 in batch[Columns.EPS_ID]:
            assert (
                len(torch.unique(batch[Columns.EPS_ID]))
                == batch[Columns.OBS].shape[0] + 1
            )
        else:
            assert (
                len(torch.unique(batch[Columns.EPS_ID])) == batch[Columns.OBS].shape[0]
            )

        gru_embeddings, state_out = self._compute_gru_embeddings_and_state_outs(
            batch[Columns.STATE_IN]["h"],
            self.observation_embedding_layer(batch[Columns.OBS]),
        )
        action_logits = self.policy_head(gru_embeddings)

        vf_preds = self.value_head(gru_embeddings).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            # Columns.INTRINSIC_REWARDS: intrinsic_rewards,
            Columns.STATE_OUT: {"h": state_out},
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
        return self.value_head(embeddings).squeeze(-1)

    def _compute_gru_embeddings_and_state_outs(self, h_in, obs_action_embedding):
        # TODO: Is this necessary? Rllib [batch, num_layers, hidden_size], GRU needs [num_layers, batch, hidden_size]
        h_in = h_in.permute(1, 0, 2)
        embeddings, h = self.gru_base_network(obs_action_embedding, h_in)
        return embeddings, h.permute(1, 0, 2)  # [batch, num_layer, hidden_size]
