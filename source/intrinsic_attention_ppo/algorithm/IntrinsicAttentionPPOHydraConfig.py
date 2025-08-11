from omegaconf import DictConfig
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


class IntrinsicAttentionPPOHydraConfig(DifferentiableAlgorithmConfig, PPOConfig):
    """Configuration for PPO with intrinsic attention rewards using a hydra config."""

    def __init__(self, algo_class=None, cfg: DictConfig | None = None):
        PPOConfig.__init__(self, algo_class=algo_class or IntrinsicAttentionPPO)

        DifferentiableAlgorithmConfig.__init__(
            self,
            algo_class=algo_class or IntrinsicAttentionPPO,
        )

        self.environment(cfg.env.name)

        # Make sure we're using the new API stack
        self.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

        # Set PyTorch as framework
        self.framework("torch")

        # Set to collect complete episodes
        self.env_runners(
            num_env_runners=cfg.env_runners.num_env_runners,
            num_cpus_per_env_runner=cfg.env_runners.num_cpus_per_env_runner,
            num_envs_per_env_runner=cfg.env_runners.num_envs_per_env_runner,
            gym_env_vectorize_mode=cfg.env_runners.gym_env_vectorize_mode,
            rollout_fragment_length=cfg.env_runners.rollout_fragment_length,
            batch_mode=cfg.env_runners.batch_mode,
        )
        diff_learner_config = DifferentiableLearnerConfig(
            learner_class=IntrinsicAttentionPPOLearner,
            add_default_connectors_to_learner_pipeline=True,
            policies_to_update=[PPO_AGENT_POLICY_ID],
            # Minibatch Size has to be larger than max_seq_length
            minibatch_size=cfg.ppo_learner.minibatch_size,
            lr=cfg.ppo_learner.lr,
            num_epochs=cfg.ppo_learner.num_epochs,
            shuffle_batch_per_epoch=cfg.ppo_learner.shuffle_batch_per_epoch,
        )

        self.learners(
            differentiable_learner_configs=[diff_learner_config],
            num_learners=cfg.ppo_learner.num_learners,
        )

        # for Params Ranges, e.g. have a look at
        # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        self.training(
            minibatch_size=cfg.intrinsic_learner.minibatch_size,
            shuffle_batch_per_epoch=cfg.intrinsic_learner.shuffle_batch_per_epoch,
            num_epochs=cfg.intrinsic_learner.num_epochs,
            lr=cfg.intrinsic_learner.lr,
            ## Training ##
            gamma=cfg.training.gamma,
            train_batch_size_per_learner=cfg.training.train_batch_size_per_learner,
            grad_clip=cfg.training.grad_clip,
            ## PPO ##
            use_critic=cfg.ppo.use_critic,
            use_gae=cfg.ppo.use_gae,
            use_kl_loss=cfg.ppo.use_kl_loss,
            lambda_=cfg.ppo.lambda_,
            kl_coeff=cfg.ppo.kl_coeff,
            kl_target=cfg.ppo.kl_target,
            vf_loss_coeff=cfg.ppo.vf_loss_coeff,
            entropy_coeff=cfg.ppo.entropy_coeff,
            clip_param=cfg.ppo.clip_param,
            vf_clip_param=cfg.ppo.vf_clip_param,
            ## Fixed ##
            vf_share_layers=True,
            learner_class=IntrinsicAttentionMetaLearner,
        )

        module_spec = RLModuleSpec(
            module_class=DifferentiablePPOModule,
            model_config={
                "vf_share_layers": True,
                "embedding_dim": cfg.ppo.embedding_dim,
                "max_seq_len": cfg.env.length + 1,
                "policy_head_hidden_sizes": [cfg.ppo.policy_head_hidden_size],
                "value_head_hidden_sizes": [cfg.ppo.value_head_hidden_size],
                "embedding_hidden_sizes": [cfg.ppo.embedding_hidden_size],
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
                "intrinsic_reward_network": {
                    "encoder_hidden_sizes": [],
                    "encoding_dim": 1,
                    "num_heads": 1,
                    "head_hidden_sizes": None,
                    "layers": [{"type": "attention"}],
                },
                "extrinsic_value_hidden_layers": [
                    cfg.ppo.extrinsic_value_hidden_size_1,
                    cfg.ppo.extrinsic_value_hidden_size_2,
                ],
                "vf_share_layers": True,
                "max_seq_len": cfg.env.length + 1,
            },
        )
        self.evaluation(
            evaluation_interval=cfg.evaluation.evaluation_interval,
            evaluation_duration=cfg.evaluation.evaluation_duration,
            evaluation_duration_unit=cfg.evaluation.evaluation_duration_unit,
            evaluation_force_reset_envs_before_iteration=cfg.evaluation.evaluation_force_reset_envs_before_iteration,
            evaluation_num_env_runners=cfg.evaluation.evaluation_num_env_runners,
        )

        # This is a bug in rllib, if you do not set self.grad_clip, but log_gradients is True (default)
        # Then no gradients will be sent through to the model updates in Learner:
        # ray/rllib/core/learner/learner.py:559
        self.log_gradients = False

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
                INTRINSIC_REWARD_MODULE_ID: AlgorithmConfig.overrides(
                    lr=cfg.intrinsic_learner.lr
                )
                # own learning rate for intrinsic reward
            },
        )
        self.debugging(seed=cfg.seed)
        self.num_cpus_for_main_process = cfg.num_cpus_for_main_process

        # Configure intrinsic reward coefficient and other meta-learning parameters
        self.learner_config_dict.update(
            {
                # Coefficient for intrinsic rewards
                "intrinsic_reward_coeff": cfg.intrinsic_learner.intrinsic_reward_coeff
            }
        )

    @override(PPOConfig)
    def get_default_learner_class(self) -> type[Learner] | str:
        return IntrinsicAttentionMetaLearner
