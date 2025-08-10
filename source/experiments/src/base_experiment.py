from typing import Dict

import os
import time
from pathlib import Path

import hydra
import ray
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from ray import tune
from ray.tune.stopper.function_stopper import FunctionStopper

from source.toy_example.PPOAgent import get_ppo_config


class TrialStopper(FunctionStopper):
    def __init__(self, max_steps_lifetime: int):
        self.steps_lifetime = 0
        self.max_steps_lifetime = max_steps_lifetime
        self._fn = self._stop_fn

    def _stop_fn(self, trial_id: str, trial_result: Dict) -> bool:
        self.steps_lifetime += trial_result["env_runners"]["num_env_steps_sampled"]
        return self.steps_lifetime >= self.max_steps_lifetime

    def __call__(self, trial_id, result):
        return self._fn(trial_id, result)


ray.init()
time_path = time.strftime("%Y-%m-%d_%H-%M-%S")


@hydra.main(
    config_path="../configs/", config_name="basic_experiment", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    ppo_config = get_ppo_config(cfg)

    root_path = Path(get_original_cwd())

    data_path = os.path.join(root_path, cfg.data_dir, f"./PPO_{time_path}")

    stopper = TrialStopper(cfg.env_steps)

    results = tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        name=f"PPO_seed{cfg.seed}",
        stop=stopper,
        storage_path=data_path,
        verbose=0,
    )

    import pprint

    pprint.pprint(results)


if __name__ == "__main__":
    main()
