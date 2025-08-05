from ray.rllib.algorithms.algorithm_config import DifferentiableAlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.learner.differentiable_learner_config import (
    DifferentiableLearnerConfig,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from source.brainstorming.algorithm.IntrinsicAttentionPPO import IntrinsicAttentionPPO
from source.brainstorming.learners.intrinsic_ppo_learner import IntrinsicPPOLearner
from source.brainstorming.rl_modules.DifferentiablePPOModule import (
    DifferentiablePPOModel,
)
from source.brainstorming.train import create_env


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

        # TODO: Set Hyperparameters
        self.learners(
            differentiable_learner_configs=[
                DifferentiableLearnerConfig(learner_class=IntrinsicPPOLearner),
            ]
        )

        # Configure main PPO model
        self.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=DifferentiablePPOModel,
                model_config={
                    "obs_embed_dim": 64,
                    "pre_head_embedding_dim": 256,
                    "gru_hidden_size": 256,
                    "gru_num_layers": 2,
                    "attention_v_dim": 32,
                    "attention_qk_dim": 32,
                    "max_seq_len": 251,
                },
                action_space=create_env(None).action_space,
                observation_space=create_env(None).observation_space,
            ),
        )

        # Configure intrinsic reward coefficient and other meta-learning parameters
        self.learner_config_dict = {
            # Coefficient for intrinsic rewards
            "intrinsic_reward_coeff": 0.01,
            # Regularization weights
            "sparsity_weight": 0.01,
            "entropy_weight": 0.001,
        }
