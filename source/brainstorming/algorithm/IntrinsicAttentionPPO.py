from typing import Any, Dict

import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.columns import Columns

# Fix the import here
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    TIMERS,
)

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID

META_LEARNER_RESULTS = "meta_learner_results"
META_LEARNER_UPDATE_TIMER = "meta_learner_update_timer"

# from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, MultiRLModule


class IntrinsicAttentionPPO(PPO):
    """PPO implementation with intrinsic attention rewards and meta-gradient learning"""

    # Register the algorithm with Ray's registry

    # Register the algorithm

    @override(PPO)
    def training_step(self) -> Dict[str, Any]:
        """
        Perform one training iteration with meta-gradient learning.

        Steps:
        1. Collect episodes from environment
        2. Inner loop: Update PPO policy with intrinsic + extrinsic rewards
        3. Outer loop: Update intrinsic reward network via meta-gradient

        Returns:
            Dict with training metrics
        """
        # 1. Sample episodes from environment
        episodes, env_runner_results = self._sample_episodes()

        # Store original episodes for meta-learning
        original_episodes = {
            k: {
                kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
            for k, v in episodes.items()
        }

        # 2. Compute intrinsic rewards for the episodes
        with self._timers.timeit("intrinsic_reward_time"):
            if INTRINSIC_REWARD_MODULE_ID in self.learner_group.modules:
                intrinsic_fwd = self.learner_group.modules[
                    INTRINSIC_REWARD_MODULE_ID
                ].forward_train(episodes)
                intrinsic_rewards = intrinsic_fwd[Columns.INTRINSIC_REWARDS]

                # Add intrinsic rewards to episodes for PPO training
                intrinsic_coeff = self.config.learner_config_dict[
                    "intrinsic_reward_coeff"
                ]
                if ALL_MODULES in episodes and Columns.REWARDS in episodes[ALL_MODULES]:
                    episodes[ALL_MODULES][Columns.REWARDS] += (
                        intrinsic_coeff * intrinsic_rewards
                    )

        # 3. Inner loop: Update the PPO policy using episodes with intrinsic rewards
        with self._timers.timeit(TIMERS.LEARNER_UPDATE):
            learner_results = self._learn_on_episodes(episodes)

        # 4. Outer loop: Update the intrinsic reward network via meta-gradient
        with self._timers.timeit((TIMERS, META_LEARNER_UPDATE_TIMER)):
            # Use original episodes (without added intrinsic rewards)
            meta_results = self.learner_group.meta_learner.update(
                original_episodes,
                others_training_data=[learner_results],
            )

        # 5. Combine and return results
        results = {
            ENV_RUNNER_RESULTS: env_runner_results,
            LEARNER_RESULTS: learner_results,
            META_LEARNER_RESULTS: meta_results,
        }

        return results
