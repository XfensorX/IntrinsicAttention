import os
import time
import uuid
from pathlib import Path

import hydra
import ray
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from ray import tune

from source.environments.umbrella_chain import UmbrellaChainEnv
from source.experiments.PPOBaseModel import get_ppo_config
from source.experiments.utils.stopper import TrialStopper

ray.init()
time_path = time.strftime("%Y-%m-%d_%H-%M-%S")


@hydra.main(
    config_path="../configs/",
    config_name="Umbrella_PPO_HPO.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    tune.register_env(
        cfg.env.name, lambda _: UmbrellaChainEnv(cfg.env.length, seed=cfg.seed)
    )
    config = get_ppo_config(cfg)

    root_path = Path(get_original_cwd())

    data_path = os.path.join(root_path, cfg.data_dir, f"./Umbrella_PPO_{time_path}")

    stopper = TrialStopper(cfg.env_steps)

    run_uuid = uuid.uuid4().hex

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        name=f"PPO_seed{cfg.seed}_length{cfg.env.length}_{run_uuid[:6]}",
        stop=stopper,
        storage_path=data_path,
        verbose=0,
    )

    for key, item in results.results.items():
        episode_return_mean = item["evaluation"]["env_runners"]["episode_return_mean"]
        return -episode_return_mean


if __name__ == "__main__":
    main()
