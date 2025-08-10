# from typing import Any, List, Dict
#
# from ray.rllib.connectors.connector_v2 import ConnectorV2
# from ray.rllib.connectors.learner import GeneralAdvantageEstimation
# from ray.rllib.core.learner import Learner
# from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.typing import EpisodeType
#
# from brainstorming.config import PPO_AGENT_POLICY_ID, INTRINSIC_REWARD_MODULE_ID
#
#
# class AddIntrinsicRewardBatchData(ConnectorV2):
#     """Learner ConnectorV2 piece computing intrinsic rewards that can be later used for loss computation."""
#
#     @override(ConnectorV2)
#     def __call__(
#             self,
#             *,
#             rl_module: MultiRLModule,
#             episodes: List[EpisodeType],
#             batch: Dict[str, Any],
#             **kwargs,
#     ):
#         assert (
#                 len(rl_module) == 2
#                 and PPO_AGENT_POLICY_ID in rl_module
#                 and INTRINSIC_REWARD_MODULE_ID in rl_module
#         )
#
#         assert PPO_AGENT_POLICY_ID in batch and INTRINSIC_REWARD_MODULE_ID not in batch
#         batch[INTRINSIC_REWARD_MODULE_ID] = batch[PPO_AGENT_POLICY_ID]
#
#         return batch
#
#
# def add_intrinsic_reward_batch_data_connector(learner: Learner):
#     if learner.config.add_default_connectors_to_learner_pipeline:
#         learner._learner_connector.insert_before(
#             GeneralAdvantageEstimation,
#             AddIntrinsicRewardBatchData(),
#         )
