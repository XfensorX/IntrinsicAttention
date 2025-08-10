import ray
from ray import tune

from source.environments.t_maze_pas import create_env_tmaze
from source.environments.umbrella_chain import UmbrellaChainEnv
from source.intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPOConfig import (
    IntrinsicAttentionPPOConfig,
)


def main():
    tune.register_env("Umbrella", lambda _: UmbrellaChainEnv(5))
    tune.register_env("TMaze", create_env_tmaze)

    ray.init()
    config = IntrinsicAttentionPPOConfig(environment="CartPole-v1")
    algo = config.build_algo()
    results = algo.train()
    import pprint

    pprint.pprint(results)


if __name__ == "__main__":
    main()
