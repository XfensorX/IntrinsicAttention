from ray.rllib.algorithms.algorithm_config import (
    AlgorithmConfig,
    DifferentiableAlgorithmConfig,
)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.learner.differentiable_learner_config import (
    DifferentiableLearnerConfig,
)
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from source.intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPO import (
    IntrinsicAttentionPPO,
)
from source.intrinsic_attention_ppo.config import (
    INTRINSIC_REWARD_MODULE_ID,
    PPO_AGENT_POLICY_ID,
)
from source.intrinsic_attention_ppo.learners.IntrinsicAttentionMetaLearner import (
    IntrinsicAttentionMetaLearner,
)
from source.intrinsic_attention_ppo.learners.IntrinsicAttentionPPOLearner import (
    IntrinsicAttentionPPOLearner,
)
from source.intrinsic_attention_ppo.rl_modules.DifferentiablePPOModule import (
    DifferentiablePPOModule,
)
from source.intrinsic_attention_ppo.rl_modules.IntrinsicAttentionModule import (
    IntrinsicAttentionModule,
)


class IntrinsicAttentionPPOConfig(DifferentiableAlgorithmConfig, PPOConfig):
    """Configuration for PPO with intrinsic attention rewards"""

    # TODO: Noch etwas lost hier alles

    def __init__(self, algo_class=None, environment=None):
        PPOConfig.__init__(
            self, algo_class=algo_class or IntrinsicAttentionPPO
        )  # FIXME: was intrinsicattention PPO

        DifferentiableAlgorithmConfig.__init__(
            self,
            algo_class=algo_class or IntrinsicAttentionPPO,
        )

        if environment is not None:
            self.environment(environment)

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
            rollout_fragment_length="auto",
            num_envs_per_env_runner=1,
            num_env_runners=0,
        )
        diff_learner_config = DifferentiableLearnerConfig(
            learner_class=IntrinsicAttentionPPOLearner,
            # minibatch_size=251, # Learning: Do NOT set this
            lr=0.01,
            add_default_connectors_to_learner_pipeline=True,
            policies_to_update=[PPO_AGENT_POLICY_ID],
            minibatch_size=500,
            # num_total_minibatches=1,
            # num_epochs=1,
        )

        self.learners(
            differentiable_learner_configs=[diff_learner_config], num_learners=0
        )

        # for Params Ranges, e.g. have a look at
        # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        self.training(
            minibatch_size=500,  # meta learner mini train betch size
            train_batch_size_per_learner=2000,
            # sgd_minibatch_size=128,
            # num_sgd_iter=10,
            num_epochs=10,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=0.5,
            entropy_coeff=0.01,
            use_critic=True,
            vf_share_layers=True,
            use_kl_loss=True,
            kl_target=0.005,
            kl_coeff=0.5,
            use_gae=True,
            vf_loss_coeff=0.75,
            learner_class=IntrinsicAttentionMetaLearner,
        )

        module_spec = RLModuleSpec(
            module_class=DifferentiablePPOModule,
            model_config={
                "obs_embed_dim": 2,
                "pre_head_embedding_dim": 5,
                "gru_hidden_size": 3,
                "gru_num_layers": 1,
                "attention_v_dim": 15,
                "attention_qk_dim": 17,
                "vf_share_layers": True,
                "max_seq_len": 500,
            },
            action_space=self.action_space,
            observation_space=self.observation_space,
        )
        intrinsic_reward_module_spec = RLModuleSpec(
            module_class=IntrinsicAttentionModule,
            observation_space=self.observation_space,
            action_space=self.action_space,
            learner_only=True,
            model_config={
                "vf_share_layers": True,
            },
        )

        # This is a bug in rllib, if you do not set self.grad_clip, but log_gradients is True (default)
        # Then no gradients will be sent through to the model updates in Learner:
        # ray/rllib/core/learner/learner.py:559
        self.log_gradients = False

        # self.multi_agent(policies=[PPO_AGENT_POLICY_ID, INTRINSIC_REWARD_MODULE_ID])
        # self.policies = {
        #     PPO_AGENT_POLICY_ID: PolicySpec(),
        #     INTRINSIC_REWARD_MODULE_ID: PolicySpec(),
        # }
        self.rl_module(
            rl_module_spec=MultiRLModuleSpec(
                # this HAS to stay in this order, ass otherwise gradient computation of the meta learner is wrong (bug in rllib)
                # ray.rllib.core.learner.torch.torch_differentiable_learner.TorchDifferentiableLearner.compute_gradients
                rl_module_specs={
                    PPO_AGENT_POLICY_ID: module_spec,
                    INTRINSIC_REWARD_MODULE_ID: intrinsic_reward_module_spec,
                }
            ),
            algorithm_config_overrides_per_module={
                INTRINSIC_REWARD_MODULE_ID: AlgorithmConfig.overrides(lr=0.0005)
                # own learning rate for intrinsic reward
            },
        )

        # Configure intrinsic reward coefficient and other meta-learning parameters
        self.learner_config_dict.update(
            {
                # Coefficient for intrinsic rewards
                "intrinsic_reward_coeff": 1.0,  # FIXME: change
                # Regularization weights
                "sparsity_weight": 0.01,
                "entropy_weight": 0.001,
            }
        )

    @override(PPOConfig)
    def get_default_learner_class(self) -> type[Learner] | str:
        return IntrinsicAttentionMetaLearner

    # @override(DifferentiableAlgorithmConfig)
    # def get_differentiable_learner_classes(
    #     self,
    # ) -> List[Union[Type[DifferentiableLearner], str]]:
    #     return [IntrinsicPPOLearner]
