from pprint import pprint

import ray
from ray import tune

from source.brainstorming.algorithm.IntrinsicAttentionPPOConfig import (
    IntrinsicAttentionPPOConfig,
)
from source.brainstorming.environments.umbrella_chain import create_env


def main():
    tune.register_env("Umbrella", create_env)

    ray.init()
    config = IntrinsicAttentionPPOConfig()
    config.environment("Umbrella")
    # .resources(
    #     num_cpus_for_main_process=1,
    #     num_cpus_per_env_runner=1,
    # )
    # Run training
    # tuner = tune.Tuner(
    #     config.algo_class,
    #     param_space=config,
    #     # Specify a stopping criterion. Note that the criterion has to match one of the
    #     # pretty printed result metrics from the results returned previously by
    #     # ``.train()``. Also note that -1100 is not a good episode return for
    #     # Pendulum-v1, we are using it here to shorten the experiment time.
    #     run_config=train.RunConfig(
    #         stop={"env_runners/episode_return_mean": -1100.0},
    #     ),
    # )
    # Run the Tuner and capture the results.
    # results = tuner.fit()
    algo = config.build_algo()

    pprint(algo.train())

    # tuner = tune.Tuner(
    #     trainable=IntrinsicAttentionPPO,
    #     param_space=config.build(),
    #     run_config=tune.RunConfig(
    #         stop={"training_iteration": 20},
    #         checkpoint_config=tune.CheckpointConfig(
    #             checkpoint_frequency=10,
    #         ),
    #         storage_path=os.path.join(os.path.dirname(__file__), "results"),
    #     ),
    # )
    # results = tuner.fit()

    # return results


if __name__ == "__main__":
    main()
