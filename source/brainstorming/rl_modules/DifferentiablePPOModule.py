from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from source.brainstorming.base_models.GRUBase import GRUBase
from source.brainstorming.base_models.ReluMlp import ReluMlp
from source.brainstorming.config import COL_EX_IN_VF_PREDS

torch, nn = try_import_torch()


class DifferentiablePPOModule(TorchRLModule, ValueFunctionAPI):
    """
    Tips:
    - use Columns.???? instead of SAMPLEBATCH.??? because this is old API (ChatGPT os often wrong)
    """

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

        self.gru_base_network = GRUBase(
            input_dim=obs_embed_dim,
            hidden_size=self._gru_hidden_size,
            num_layers=self.model_config.get("gru_num_layers"),
            output_size=pre_head_embedding_dim,
        )

        self.value_head = ReluMlp(
            [pre_head_embedding_dim, pre_head_embedding_dim, 1], output_layer=None
        )

        # self.extrinsic_value_head = ReluMlp(
        #     [pre_head_embedding_dim, pre_head_embedding_dim, 1], output_layer=None
        # )

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
            Columns.ACTION_DIST_INPUTS: action_logits,
            # this is the distribution, action sampling is done automatically
            Columns.STATE_OUT: {"h": state_outs},
            Columns.EMBEDDINGS: embeddings,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        gru_embeddings, state_out = self._compute_gru_embeddings_and_state_outs(
            batch[Columns.STATE_IN]["h"],
            self.observation_embedding_layer(batch[Columns.OBS]),
        )
        action_logits = self.policy_head(gru_embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.STATE_OUT: {"h": state_out},
            # Columns.EMBEDDINGS: gru_embeddings,  # TODO: remove this, as we should not use it outside
            COL_EX_IN_VF_PREDS: self.value_head(gru_embeddings).squeeze(-1),
            # COL_EX_VF_PREDS: self.extrinsic_value_head(gru_embeddings).squeeze(-1),
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        raise ValueError("This cannot be called !")

    def _compute_gru_embeddings_and_state_outs(self, h_in, obs_action_embedding):
        # TODO: Is this necessary? Rllib [batch, num_layers, hidden_size], GRU needs [num_layers, batch, hidden_size]
        h_in = h_in.permute(1, 0, 2)
        embeddings, h = self.gru_base_network(obs_action_embedding, h_in)
        return embeddings, h.permute(1, 0, 2)  # [batch, num_layer, hidden_size]
