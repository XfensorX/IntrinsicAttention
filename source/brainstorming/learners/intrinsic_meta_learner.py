from typing import Any, Dict, List

import contextlib

from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.learner.torch.torch_meta_learner import TorchMetaLearner
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, NamedParamDict, ParamDict, TensorType

from source.brainstorming.config import INTRINSIC_REWARD_MODULE_ID, PPO_AGENT_POLICY_ID
from source.brainstorming.learners.CustomPPOLearner import CustomPPOLearner
from source.brainstorming.learners.IntrinsicRewardLearnerConnector import (
    remove_gae_from_learner_connectors,
)

torch, nn = try_import_torch()


class IntrinsicAttentionMetaLearner(TorchMetaLearner, CustomPPOLearner):
    """Meta-learner for updating the intrinsic reward network"""

    @override(TorchMetaLearner)
    def build(self) -> None:
        """Build the meta-learner with a proper connector pipeline."""
        # Initialize the base learner
        super().build()
        remove_gae_from_learner_connectors(self)
        print(f"Meta Learner: {self._learner_connector=}")
        self._custom_with_one_ts_to_episode = bool(
            "AddOneTsToEpisodesAndTruncate"
            in [str(x) for x in self._learner_connector.connectors]
        )

    @override(TorchMetaLearner)
    def compute_losses(
        self,
        *,
        fwd_out: Dict[str, Any],
        batch: Dict[str, Any],
        others_loss_per_module: List[Dict[ModuleID, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        loss = self.compute_ppo_loss(
            config=self.config.get_config_for_module(PPO_AGENT_POLICY_ID),
            batch=batch,
            fwd_out=fwd_out,
            use_intrinsic_rewards=False,
            with_one_ts_to_episode=self._custom_with_one_ts_to_episode,
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )

        return {
            PPO_AGENT_POLICY_ID: loss,
            INTRINSIC_REWARD_MODULE_ID: torch.tensor([0], device=loss.device),
        }

    @override(TorchLearner)
    def configure_optimizers_for_module(
        self,
        module_id: ModuleID,
        config: "AlgorithmConfig" = None,
    ) -> None:
        if module_id in self.get_inner_loop_policies():
            return

        module = self._module[module_id]

        params = self.get_parameters(module)
        optimizer = torch.optim.Adam(params)

        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )

    @override(TorchMetaLearner)
    def _make_functional_call(
        self, params: Dict[ModuleID, NamedParamDict], batch: MultiAgentBatch
    ) -> Dict[ModuleID, NamedParamDict]:
        """Make a functional forward call to all modules in the `MultiRLModule`.
        Only using the parameters present in params."""

        return {
            module_id: torch.func.functional_call(
                module, params[module_id], batch[module_id]
            )
            for module_id, module in self._module._rl_modules.items()
            if module_id in params
        }

    @override(TorchLearner)
    def compute_gradients(
        self,
        loss_per_module: Dict[ModuleID, TensorType],
        params: Dict[ModuleID, NamedParamDict],
        **kwargs,
    ) -> ParamDict:
        # for optim in self._optimizer_parameters:
        #     optim.zero_grad(set_to_none=True)

        if self._grad_scalers is not None:
            total_loss = sum(
                self._grad_scalers[mid].scale(loss)
                for mid, loss in loss_per_module.items()
            )
        else:
            total_loss = sum(loss_per_module.values())

        grads = torch.autograd.grad(
            total_loss,
            sum((list(param.values()) for mid, param in params.items()), []),
        )

        grads_list = list(grads)
        offset = 0
        named_grads = {}
        for module_id, module_params in params.items():
            n = len(module_params)
            module_slice = grads_list[offset : offset + n]
            named_grads[module_id] = {
                name: g for (name, _), g in zip(module_params.items(), module_slice)
            }
            offset += n

        return named_grads

    @override(TorchMetaLearner)
    def _uncompiled_update(
        self,
        batch: Dict,
        params: Dict[ModuleID, NamedParamDict],
        others_loss_per_module: List[Dict[ModuleID, TensorType]] = None,
        **kwargs,
    ):
        """
        This function gets overriden and chnged a bit from the original:
        ray.rllib.core.learner.torch.torch_meta_learner.TorchMetaLearner._uncompiled_update
        because we want to update the gradients from the PPO-Network to the inner loops gradients.
        """

        self._compute_off_policyness(batch)

        # for policy in self.module.keys():
        #     for p in self.module[policy].parameters():
        #         p.detach_()  # break ties to previous graph
        #         p.requires_grad_(True)  # make them leafs again
        #         p.grad = None  # be tidy
        inner_loop_policies = self.get_inner_loop_policies()
        outer_loop_policies = set(self.module.keys()) - inner_loop_policies

        inner_loop_params = {
            mid: p for mid, p in params.items() if mid in inner_loop_policies
        }
        outer_loop_params = {
            mid: p for mid, p in params.items() if mid in outer_loop_policies
        }

        fwd_out = self._make_functional_call(params, batch)
        loss_per_module = self.compute_losses(
            fwd_out=fwd_out, batch=batch, others_loss_per_module=others_loss_per_module
        )

        for policy in self.module.keys():
            for p in self.module[policy].parameters():
                p.detach_()  # break ties to previous graph
                p.requires_grad_(True)  # make them leafs again
                p.grad = None  # be tidy

        gradients = self.compute_gradients(loss_per_module, outer_loop_params)

        with contextlib.ExitStack() as stack:
            if self.config.num_learners > 1:
                for mod in self.module.values():
                    # Skip non-torch modules, b/c they may not have the `no_sync` API.
                    if isinstance(mod, torch.nn.Module):
                        stack.enter_context(mod.no_sync())
            postprocessed_gradients = self.postprocess_gradients(gradients)

            for policy in outer_loop_policies:
                for name, p in self.module[policy].named_parameters():
                    if name in postprocessed_gradients:
                        p.grad = postprocessed_gradients[name]

            self.apply_gradients({})

        # for optim in self._optimizer_parameters:
        #     # `set_to_none=True` is a faster way to zero out the gradients.
        #     optim.zero_grad(set_to_none=True)

        self.update_gradients_from_inner_loop(params)

        # outer_loop_policies = set(self.module.keys()) - self.get_inner_loop_policies()
        #
        # for policy in outer_loop_policies:
        #     for p in self.module[policy].parameters():
        #         p.detach_()  # break ties to previous graph
        #         p.requires_grad_(True)  # make them leafs again
        #         p.grad = None  # be tidy

        for policy in self.module.keys():
            for p in self.module[policy].parameters():
                p.detach_()  # break ties to previous graph
                p.requires_grad_(True)  # make them leafs again
                p.grad = None  # be tidy

        self._params = {}
        return fwd_out, loss_per_module, {}

    def update_gradients_from_inner_loop(
        self, inner_loop_parameters: Dict[ModuleID, NamedParamDict]
    ):
        inner_loop_policies = self.get_inner_loop_policies()

        # 1) Write fast weights into the module WITHOUT building a graph
        with torch.no_grad():
            for policy in inner_loop_policies:
                mod = self.module[policy]
                for name, p in mod.named_parameters():
                    if name in inner_loop_parameters[policy]:
                        p.copy_(
                            inner_loop_parameters[policy][name]
                        )  # no .data, no .detach()

        # 2) Start a brand-new graph next iteration
        for policy in inner_loop_policies:
            for p in self.module[policy].parameters():
                p.detach_()  # break ties to previous graph
                p.requires_grad_(True)  # make them leafs again
                p.grad = None  # be tidy

    def get_inner_loop_policies(self) -> set[ModuleID]:
        inner_loop_policies = set()
        for config in self.config.differentiable_learner_configs:
            for policy in config.policies_to_update:
                inner_loop_policies.add(policy)

        return inner_loop_policies
