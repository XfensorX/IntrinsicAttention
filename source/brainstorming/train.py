import os

import ray
from ray import tune
from ray.rllib.algorithms import PPO

from source.brainstorming.algorithm.IntrinsicAttentionPPOConfig import (
    IntrinsicAttentionPPOConfig,
)
from source.brainstorming.environments.umbrella_chain import create_env

# Define environment creation function


def main():
    # Register environment with Ray
    tune.register_env("Umbrella", create_env)

    # Initialize Ray
    ray.init()

    # Configure the algorithm
    config = (
        IntrinsicAttentionPPOConfig().environment("Umbrella")
        # Hardware resources
        # .resources(
        #     num_cpus_for_main_process=1,
        #     num_cpus_per_env_runner=1,
        # )
    )
    # Run training
    tuner = tune.Tuner(
        trainable=PPO,
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 20},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
            ),
            storage_path=os.path.join(os.path.dirname(__file__), "results"),
        ),
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    main()
