import os
import time
from pathlib import Path

import hydra
import ray
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from ray import tune

from source.toy_example.PPOAgent import get_ppo_config

ray.init()
time_path = time.strftime("%Y-%m-%d_%H-%M-%S")


@hydra.main(
    config_path="../configs/", config_name="basic_experiment", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    ppo_config = get_ppo_config(cfg)

    root_path = Path(get_original_cwd())

    data_path = os.path.join(root_path, cfg.data_dir, f"./PPO_{time_path}")

    results = tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        name=f"PPO_seed{cfg.seed}",
        stop={"num_env_steps_sampled_lifetime": cfg.env_steps},
        storage_path=data_path,
        verbose=0,
    )

    import pprint

    pprint.pprint(results)


if __name__ == "__main__":
    main()
