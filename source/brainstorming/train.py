import os

import ray
from ray import tune

from source.brainstorming.algorithm.IntrinsicAttentionPPO import IntrinsicAttentionPPO
from source.brainstorming.algorithm.IntrinsicAttentionPPOConfig import (
    IntrinsicAttentionPPOConfig,
)
from source.brainstorming.environments.t_maze_pas import create_env_tmaze


def main():
    tune.register_env("Umbrella", lambda _: UmbrellaChainEnv(5))
    tune.register_env("TMaze", create_env_tmaze)

    ray.init()
    config = IntrinsicAttentionPPOConfig(environment="CartPole-v1")
    # .resources(
    #     num_cpus_for_main_process=1,
    #     num_cpus_per_env_runner=1,
    # )
    results = tune.run(
        IntrinsicAttentionPPO,
        config=config.to_dict(),
        name="FirstTry",
        stop={"num_env_steps_sampled_lifetime": 200000},
        storage_path=os.path.join(os.path.dirname(__file__), "results"),
        verbose=1,
    )

    import pprint

    pprint.pprint(results)


if __name__ == "__main__":
    main()
