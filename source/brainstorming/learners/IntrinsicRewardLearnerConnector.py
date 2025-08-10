from typing import Any, Dict, List

from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner import Learner
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID


class IntrinsicRewardCalculation(ConnectorV2):
    """Learner ConnectorV2 piece computing intrinsic rewards that can be later used for loss computation."""

    def __init__(self, intrinsic_reward_coeff: float):
        super().__init__()
        self.intrinsic_reward_coeff = intrinsic_reward_coeff

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        assert (
            len(rl_module) == 2
            and PPO_AGENT_POLICY_ID in rl_module
            and INTRINSIC_REWARD_MODULE_ID in rl_module
        )

        assert PPO_AGENT_POLICY_ID in batch and INTRINSIC_REWARD_MODULE_ID not in batch
        assert Columns.REWARDS in batch[PPO_AGENT_POLICY_ID]

        fwd_out = rl_module[INTRINSIC_REWARD_MODULE_ID].forward_train(
            batch[PPO_AGENT_POLICY_ID]
        )

        batch[PPO_AGENT_POLICY_ID][Columns.INTRINSIC_REWARDS] = fwd_out[
            Columns.INTRINSIC_REWARDS
        ]
        batch[PPO_AGENT_POLICY_ID][Columns.REWARDS] += (
            self.intrinsic_reward_coeff * fwd_out[Columns.INTRINSIC_REWARDS]
        )

        return batch


def add_intrinsic_reward_connector(learner: Learner):
    raise NotImplementedError
    learner_config_dict = learner.config.learner_config_dict

    # Assert, we are only training one policy (RLModule) and we have the ICM
    # in our MultiRLModule.

    assert "intrinsic_reward_coeff" in learner_config_dict

    if learner.config.add_default_connectors_to_learner_pipeline:
        learner._learner_connector.insert_after(
            NumpyToTensor,
            IntrinsicRewardCalculation(learner_config_dict["intrinsic_reward_coeff"]),
        )


def remove_gae_from_learner_connectors(learner: Learner):
    learner._learner_connector.remove(GeneralAdvantageEstimation)
