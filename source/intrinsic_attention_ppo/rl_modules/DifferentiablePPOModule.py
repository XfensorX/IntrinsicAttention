from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from intrinsic_attention_ppo.rl_modules.ReluMlp import (
    ReluMlp,
)
from source.intrinsic_attention_ppo.config import COL_EX_IN_VF_PREDS

torch, nn = try_import_torch()


class DifferentiablePPOModuleConfig(BaseModel):
    embedding_dim: int
    max_seq_len: int  # to make sure full episodes are used everywhere
    embedding_hidden_sizes: list[int]
    policy_head_hidden_sizes: list[int]
    value_head_hidden_sizes: list[int]
    vf_share_layers: bool


class DifferentiablePPOModule(TorchRLModule, ValueFunctionAPI):
    """
    Tips:
    - use Columns.???? instead of SAMPLEBATCH.??? because this is old API (ChatGPT os often wrong)
    """

    # This class contains significant contributions from:
    # https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py
    @override(TorchRLModule)
    def setup(self):
        config = DifferentiablePPOModuleConfig.model_validate(self.model_config)
        action_dim = self.action_space.n
        obs_dim = self.observation_space.shape[0]

        self.embedding_layer = ReluMlp(
            hidden_layers=config.embedding_hidden_sizes,
            input_size=obs_dim,
            output_size=config.embedding_dim,
        )
        self.value_head = ReluMlp(
            hidden_layers=config.value_head_hidden_sizes,
            input_size=config.embedding_dim,
            output_size=1,
            output_layer=None,
        )
        self.policy_head = ReluMlp(
            hidden_layers=config.policy_head_hidden_sizes,
            input_size=config.embedding_dim,
            output_size=action_dim,
            output_layer=None,
        )

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(
                1,
            )
        }  # This is needed to keep episodes

    @override(TorchRLModule)
    @torch.no_grad()
    def _forward(self, batch, **kwargs):
        """
        :param batch: Description
            batch[Columns.OBS].shape == (batch, time, obs_dim)
        """
        embeddings = self.embedding_layer(batch[Columns.OBS])
        action_logits = self.policy_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.EMBEDDINGS: embeddings,
            # this is the distribution, action sampling is done automatically
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings = self.embedding_layer(batch[Columns.OBS])
        action_logits = self.policy_head(embeddings)
        value_predictions = self.value_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.EMBEDDINGS: embeddings,
            COL_EX_IN_VF_PREDS: value_predictions.squeeze(-1),
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        raise ValueError("This cannot be called !")
