import os

import gymnasium as gym
import numpy as np
import ray
from gymnasium.wrappers import DtypeObservation, FlattenObservation
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
from ray import tune

from source.brainstorming.algorithm import IntrinsicAttentionPPOConfig


# Define environment creation function
def create_env(config):
    import minigrid  # noqa

    # Create MiniGrid environment
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    # Apply wrappers
    env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = DtypeObservation(env, dtype=np.float32)
    env = FlattenObservation(env)

    return env


def main():
    # Define environment creation function
    env_creator = lambda config: create_env(config)

    # Register environment with Ray
    tune.register_env("MiniGrid-DoorKey-5x5-v0", create_env)

    # Initialize Ray
    ray.init()

    # Configure the algorithm
    config = (
        IntrinsicAttentionPPOConfig()
        .environment("MiniGrid-DoorKey-5x5-v0")
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
