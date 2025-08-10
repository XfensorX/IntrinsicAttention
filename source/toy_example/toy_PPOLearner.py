from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType


class ToyPPOLearner(PPOTorchLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[ModuleID, Any],
        fwd_out: Dict[ModuleID, Any],
    ) -> TensorType:
        self.metrics.log_value(
            key="ActionDist",
            value=fwd_out[Columns.ACTION_DIST_INPUTS],
            reduce=None,
            clear_on_reduce=True,
        )
        return super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
