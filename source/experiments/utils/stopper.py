from typing import Dict

from ray.tune.stopper.function_stopper import FunctionStopper


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
