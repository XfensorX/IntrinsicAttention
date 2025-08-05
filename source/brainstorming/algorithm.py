from typing import Any, Dict

import torch
from ray.rllib.algorithms.algorithm_config import (
    DifferentiableAlgorithmConfig,
)
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.columns import Columns

# Fix the import here
from ray.rllib.core.learner.differentiable_learner_config import (
    DifferentiableLearnerConfig,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    TIMERS,
)

META_LEARNER_RESULTS = "meta_learner_results"
META_LEARNER_UPDATE_TIMER = "meta_learner_update_timer"

from source.brainstorming.learners.intrinsic_meta_learner import (
    IntrinsicAttentionMetaLearner,
)
from source.brainstorming.learners.intrinsic_ppo_learner import IntrinsicPPOLearner
from source.brainstorming.models.DifferentiableIntrinsicAttention import (
    DifferentiableIntrinsicAttentionModule,
)
from source.brainstorming.models.model import IntrinsicAttentionPPOModel


class IntrinsicAttentionPPOConfig(PPOConfig, DifferentiableAlgorithmConfig):
    """Configuration for PPO with intrinsic attention rewards"""

    def __init__(self, algo_class=None):
        PPOConfig.__init__(self, algo_class=algo_class or IntrinsicAttentionPPO)

        # Make sure we're using the new API stack
        self.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

        # Set PyTorch as framework
        self.framework("torch")

        # Set to collect complete episodes
        self.env_runners(
            batch_mode="complete_episodes",
        )

        # Configure learners
        self.learner_class = IntrinsicPPOLearner
        self.meta_learner_class = IntrinsicAttentionMetaLearner

        # Configure main PPO model
        self.rl_module_spec = IntrinsicAttentionPPOModel

        # Configure the differentiable module for intrinsic rewards
        self.differentiable_learner_configs = [
            DifferentiableLearnerConfig(
                module_id="intrinsic_reward_module",
                learner_class=None,  # Will use default differentiable learner
                module_spec=DifferentiableIntrinsicAttentionModule,
                optimizer_config={
                    "_default": {
                        "type": "adam",
                        "lr": 0.0001,
                    },
                },
            ),
        ]

        # Configure intrinsic reward coefficient and other meta-learning parameters
        self.learner_config_dict = {
            # Coefficient for intrinsic rewards
            "intrinsic_reward_coeff": 0.01,
            # Regularization weights
            "sparsity_weight": 0.01,
            "entropy_weight": 0.001,
        }


class IntrinsicAttentionPPO(PPO):
    """PPO implementation with intrinsic attention rewards and meta-gradient learning"""

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
            intrinsic_module_id = "intrinsic_reward_module"
            if intrinsic_module_id in self.learner_group.modules:
                intrinsic_fwd = self.learner_group.modules[
                    intrinsic_module_id
                ].forward_train(episodes)
                intrinsic_rewards = intrinsic_fwd.get(Columns.INTRINSIC_REWARDS)

                # Add intrinsic rewards to episodes for PPO training
                intrinsic_coeff = self.config.learner_config_dict.get(
                    "intrinsic_reward_coeff", 0.01
                )
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
