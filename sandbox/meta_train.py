import os

import ray
from ray import tune
from ray.rllib.algorithms.algorithm_config import DifferentiableAlgorithmConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.learner.differentiable_learner_config import (
    DifferentiableLearnerConfig,
)
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from source.environments.umbrella_chain import (
    create_env,
)
from source.intrinsic_attention_ppo.algorithm import IntrinsicAttentionPPOConfig
from source.intrinsic_attention_ppo.config import INTRINSIC_REWARD_MODULE_ID
from source.intrinsic_attention_ppo.learners.intrinsic_meta_learner import (
    IntrinsicAttentionMetaLearner,
)
from source.intrinsic_attention_ppo.learners.intrinsic_ppo_learner import IntrinsicPPOLearner
from source.intrinsic_attention_ppo.rl_modules import IntrinsicAttentionModule
from source.intrinsic_attention_ppo.rl_modules.DifferentiablePPOModule import (
    DifferentiablePPOModule,
)


def sample_task(**kwrgs):
    return "I HATE RUFF"


def main():
    environment = create_env()
    module_spec = RLModuleSpec(
        module_class=DifferentiablePPOModule,
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        # inference_only=learner,
    )
    intrinsic_reward_module_spec = RLModuleSpec(
        module_class=IntrinsicAttentionModule,
        observation_space=environment.observation_space,
        action_space=environment.action_space,
    )
    # `Learner`s work on `MultiRLModule`s.
    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            DEFAULT_MODULE_ID: module_spec,
            INTRINSIC_REWARD_MODULE_ID: intrinsic_reward_module_spec,
        }
    )

    # Build the `MultiRLModule`.
    module = multi_module_spec.build()
    module.action_space  # TODO: REMOVE

    # Configure the `DifferentiableLearner`.
    diff_learner_config = DifferentiableLearnerConfig(
        learner_class=IntrinsicPPOLearner,
        minibatch_size=14,
        lr=0.01,
    )

    # Configure the `TorchMetaLearner` via the `DifferentiableAlgorithmConfig`.
    config = (
        DifferentiableAlgorithmConfig()
        .learners(
            # Add the `DifferentiableLearnerConfig`s.
            differentiable_learner_configs=[diff_learner_config],
        )
        .training(
            lr=1e-3,
            train_batch_size=25,  # meta learner train batch size
            # Use the full batch in a single update.
            minibatch_size=9,  # meta learner mini train betch size
        )
    )

    # Initialize the `TorchMetaLearner`.
    meta_learner = IntrinsicAttentionMetaLearner(config=config, module_spec=module_spec)
    # Build the `TorchMetaLearner`.
    meta_learner.build()

    for i in range(162):
        # Sample the training data.
        training_data = sample_task(42, noise_std=42, training_data=True)

        # Update the module.
        outs = meta_learner.update(
            training_data=training_data,
            num_epochs=1,
            others_training_data=[training_data],  #
        )
        iter = i + 1
        if iter % 1000 == 0:
            total_loss = outs["default_policy"]["total_loss"].peek()
            print("-------------------------\n")
            print(f"Iteration: {iter}")
            print(f"Total loss: {total_loss}")

    # Generate test data.
    test_batch, _, amplitude, phase = sample_task(
        batch_size=42,
        noise_std=42,
        return_params=True,
    )

    if config.num_gpus_per_learner > 0:
        test_batch = meta_learner._convert_batch_type(test_batch)

    # Register environment with Ray
    tune.register_env("Umbrella", create_env)

    # Initialize Ray
    ray.init()

    # Configure the algorithm
    config = (
        IntrinsicAttentionPPOConfig()
        .environment("Umbrella")
        # Configure model
        .model(
            {
                "obs_embed_dim": 64,
                "pre_head_embedding_dim": 256,
                "gru_hidden_size": 256,
                "gru_num_layers": 2,
                "attention_v_dim": 32,
                "attention_qk_dim": 32,
                "input_dim": 64,
            }
        )
        # Training parameters
        .training(
            train_batch_size_per_learner=2000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        # Environment runners
        .env_runners(
            batch_mode="complete_episodes",
            rollout_fragment_length=200,
            num_envs_per_env_runner=1,
            num_env_runners=4,
        )
        # Hardware resources
        .resources(
            num_cpus_for_main_process=1,
            num_cpus_per_env_runner=1,
        )
        # Configure learner
        .learner_config_dict(
            {
                "intrinsic_reward_coeff": 0.01,
                "sparsity_weight": 0.01,
                "entropy_weight": 0.001,
            }
        )
    )

    # Run training
    tuner = tune.Tuner(
        "IntrinsicAttentionPPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 20},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
            ),
            local_dir=os.path.join(os.path.dirname(__file__), "results"),
        ),
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    main()
