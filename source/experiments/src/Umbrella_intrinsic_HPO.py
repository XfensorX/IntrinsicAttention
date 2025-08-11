import os
import time
from pathlib import Path

import hydra
import ray
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from ray import tune

from source.environments.umbrella_chain import UmbrellaChainEnv
from source.experiments.utils.stopper import TrialStopper
from source.intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPO import (
    IntrinsicAttentionPPO,
)
from source.intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPOHydraConfig import (
    IntrinsicAttentionPPOHydraConfig,
)

ray.init()
time_path = time.strftime("%Y-%m-%d_%H-%M-%S")


@hydra.main(
    config_path="../configs/",
    config_name="Umbrella_intrinsic_HPO.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    tune.register_env(cfg.env.name, lambda _: UmbrellaChainEnv(cfg.env.length))
    config = IntrinsicAttentionPPOHydraConfig(cfg=cfg)

    root_path = Path(get_original_cwd())

    data_path = os.path.join(
        root_path, cfg.data_dir, f"./Umbrella_intrinsic_{time_path}"
    )

    stopper = TrialStopper(cfg.env_steps)

    results = tune.run(
        IntrinsicAttentionPPO,
        config=config.to_dict(),
        name=f"IntrinsicAttentionPPO_seed{cfg.seed}_length{cfg.env.length}",
        stop=stopper,
        storage_path=data_path,
        verbose=0,
    )
    print(f"Type: {type(results)}")

    for key, item in results.results.items():
        print(f"Key: {key}")
        print(f"Item keys: {list(item['evaluation']['env_runners'].keys())}")
        episode_return_mean = item["evaluation"]["env_runners"]["episode_return_mean"]
        print(f"{episode_return_mean}")
        return -episode_return_mean


if __name__ == "__main__":
    main()
