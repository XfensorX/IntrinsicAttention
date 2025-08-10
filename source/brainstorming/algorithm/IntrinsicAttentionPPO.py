import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.training_data import TrainingData
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID

META_LEARNER_RESULTS = "meta_learner_results"
META_LEARNER_UPDATE_TIMER = "meta_learner_update_timer"


class IntrinsicAttentionPPO(PPO):
    """PPO implementation with intrinsic attention rewards and meta-gradient learning"""

    # Register the algorithm with Ray's registry

    # Register the algorithm

    @override(PPO)
    def training_step(self) -> None:
        """
        Perform one training iteration with meta-gradient learning.

        Steps:
        1. Collect episodes from environment
        2. Inner loop: Update PPO policy with intrinsic + extrinsic rewards
        3. Outer loop: Update intrinsic reward network via meta-gradient

        Returns:
            Dict with training metrics
        """
        batch = self.custom_sample_batch()
        learner_results = self.custom_meta_gradient_update(batch)
        self.custom_sync_weights(learner_results)

    def custom_sync_weights(self, learner_results):
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}

            print(f"DEBUG 1: {modules_to_update=}")
            self.env_runner_group.sync_weights(
                # Sync weights from learner_group to all EnvRunners.
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                inference_only=True,
            )

    def custom_sample_episodes(self):
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            # Sample in parallel from the workers.
            episodes, env_runner_results = synchronous_parallel_sample(
                worker_set=self.env_runner_group,
                max_env_steps=self.config.total_train_batch_size,
                sample_timeout_s=self.config.sample_timeout_s,
                _uses_new_env_runners=(self.config.enable_env_runner_and_connector_v2),
                _return_metrics=True,
            )

        self.metrics.aggregate(env_runner_results, key=ENV_RUNNER_RESULTS)

        return episodes

    def custom_sample_batch(self):
        episodes = self.custom_sample_episodes()
        print(f"DEBUG 2: {self.learner_group._learner._learner_connector.connectors=}")
        batch = self.learner_group._learner._learner_connector(
            rl_module=self.learner_group._learner.module,
            episodes=episodes,
            metrics=self.metrics,
        )
        return batch

    def custom_meta_gradient_update(self, batch):
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update(  # THis is actully only the MetaLearner
                batch=MultiAgentBatch(
                    policy_batches={
                        INTRINSIC_REWARD_MODULE_ID: SampleBatch(
                            **batch[PPO_AGENT_POLICY_ID],
                            _max_seq_len=251,  # TODO: load from config
                            _zero_padded=True,
                        ),
                        PPO_AGENT_POLICY_ID: SampleBatch(
                            **batch[PPO_AGENT_POLICY_ID],
                            _max_seq_len=251,  # TODO: load from config
                            _zero_padded=True,
                        ),
                    },
                    env_steps=torch.prod(
                        torch.tensor(batch[PPO_AGENT_POLICY_ID][Columns.OBS].shape[:-1])
                    ),
                ),
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(
                            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)
                        )
                    ),
                },
                num_epochs=self.config.num_epochs,
                minibatch_size=self.config.minibatch_size,
                shuffle_batch_per_epoch=self.config.shuffle_batch_per_epoch,
                others_training_data=[
                    TrainingData(
                        batch=MultiAgentBatch(
                            policy_batches={
                                PPO_AGENT_POLICY_ID: SampleBatch(
                                    **batch[PPO_AGENT_POLICY_ID],
                                    _max_seq_len=251,  # TODO: load from config
                                    _zero_padded=True,
                                ),
                                # TODO: think about if this is correct: (validate that the intrinsic rewards are not used for loss calculation)
                                INTRINSIC_REWARD_MODULE_ID: SampleBatch(
                                    **batch[PPO_AGENT_POLICY_ID],
                                    _max_seq_len=251,  # TODO: load from config
                                    _zero_padded=True,
                                ),
                            },
                            env_steps=torch.prod(
                                torch.tensor(
                                    batch[PPO_AGENT_POLICY_ID][Columns.OBS].shape[:-1]
                                )
                            ),
                        ),
                    )
                ],
            )
        self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)
        return learner_results
