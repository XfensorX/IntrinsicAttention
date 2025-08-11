from omegaconf import DictConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


def get_ppo_config(cfg: DictConfig) -> PPOConfig:
    config = (
        PPOConfig()
        .environment(cfg.env.name)
        .framework("torch")
        .api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                model_config=DefaultModelConfig(
                    vf_share_layers=True,
                    fcnet_hiddens=[cfg.ppo.hidden_size_1, cfg.ppo.hidden_size_2],
                ),
            )
        )
        .training(
            lr=cfg.ppo_learner.lr,
            gamma=cfg.training.gamma,
            num_epochs=cfg.ppo_learner.num_epochs,
            minibatch_size=cfg.ppo_learner.minibatch_size,
            shuffle_batch_per_epoch=cfg.ppo_learner.shuffle_batch_per_epoch,
            train_batch_size_per_learner=cfg.training.train_batch_size_per_learner,
            #### PPO ###
            use_critic=cfg.ppo.use_critic,
            use_gae=cfg.ppo.use_gae,
            lambda_=cfg.ppo.lambda_,
            use_kl_loss=cfg.ppo.use_kl_loss,
            kl_coeff=cfg.ppo.kl_coeff,
            kl_target=cfg.ppo.kl_target,
            vf_loss_coeff=cfg.ppo.vf_loss_coeff,
            entropy_coeff=cfg.ppo.entropy_coeff,
            clip_param=cfg.ppo.clip_param,
            vf_clip_param=cfg.ppo.vf_clip_param,
            grad_clip=cfg.ppo.grad_clip,
        )
        .learners(
            num_learners=cfg.ppo_learner.num_learners, num_cpus_per_learner="auto"
        )
        .env_runners(
            num_env_runners=cfg.env_runners.num_env_runners,
            num_envs_per_env_runner=cfg.env_runners.num_envs_per_env_runner,
            gym_env_vectorize_mode=cfg.env_runners.gym_env_vectorize_mode,
            num_cpus_per_env_runner=cfg.env_runners.num_cpus_per_env_runner,
            rollout_fragment_length=cfg.env_runners.rollout_fragment_length,
            batch_mode=cfg.env_runners.batch_mode,
        )
        .evaluation(
            evaluation_interval=cfg.evaluation.evaluation_interval,
            evaluation_duration=cfg.evaluation.evaluation_duration,
            evaluation_duration_unit=cfg.evaluation.evaluation_duration_unit,
            evaluation_force_reset_envs_before_iteration=cfg.evaluation.evaluation_force_reset_envs_before_iteration,
            evaluation_num_env_runners=cfg.evaluation.evaluation_num_env_runners,
        )
        .debugging(
            seed=cfg.seed,
        )
    )
    config.num_cpus_for_main_process = cfg.num_cpus_for_main_process
    return config
