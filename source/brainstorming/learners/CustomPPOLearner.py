from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import (
    PPOConfig,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, TensorType

from source.brainstorming.config import (
    COL_EX_IN_VF_PREDS,
    COL_EX_VF_PREDS,
    INTRINSIC_REWARD_MODULE_ID,
    PPO_AGENT_POLICY_ID,
)
from source.brainstorming.learners.pytorch_differentiable_funcs import compute_gae

torch, nn = try_import_torch()


# def are_equal_batches(batch1: Dict[str, torch.Tensor], batch2: Dict[str, torch.Tensor]):
#     assert batch1.accessed_keys == batch2.accessed_keys
#     for key in batch1.accessed_keys - set(["state_in"]):
#         assert batch1[key].shape == batch2[key].shape
#         assert (batch1[key] == batch2[key]).all()
#
#     if "state_in" in batch1.accessed_keys:
#         for key in batch1["state_in"].keys():
#             assert batch1["state_in"][key].shape == batch2["state_in"][key].shape
#             assert (batch1["state_in"][key] == batch2["state_in"][key]).all()
#     return True


class CustomPPOLearner(PPOTorchLearner):
    def compute_ppo_loss(
        self,
        use_intrinsic_rewards: bool,  # New parameter
        with_one_ts_to_episode: bool,  # whether AddOneTsToEpisodeAndTruncate Learner Connector was used
        gamma: float,
        lambda_: float,
        *,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[ModuleID, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        This function was taken over from
        ray.rllib.algorithms.ppo.torch.ppo_torch_learner.PPOTorchLearner.compute_loss_for_module
        but slightly adjusted to allow correct meta-gradient calculation.
        """

        _batch = batch[
            PPO_AGENT_POLICY_ID
        ]  # placeholder to give hints on where module id is arbitrary

        curr_action_dist_logits_ = fwd_out[PPO_AGENT_POLICY_ID][
            Columns.ACTION_DIST_INPUTS
        ]
        prev_action_dist_logits_ = _batch[Columns.ACTION_DIST_INPUTS]
        taken_actions_ = _batch[Columns.ACTIONS]
        prev_actions_logp_ = _batch[Columns.ACTION_LOGP]
        rewards = _batch[Columns.REWARDS]

        module = self.module[PPO_AGENT_POLICY_ID].unwrapped()

        if use_intrinsic_rewards:
            value_predictions_ = fwd_out[PPO_AGENT_POLICY_ID][COL_EX_IN_VF_PREDS]
            rewards += (
                fwd_out[INTRINSIC_REWARD_MODULE_ID][Columns.INTRINSIC_REWARDS]
                * config.learner_config_dict["intrinsic_reward_coeff"]
            )
        else:
            value_predictions_ = fwd_out[PPO_AGENT_POLICY_ID][COL_EX_VF_PREDS]

        advantages, value_targets = compute_gae(
            gamma=gamma,
            lambda_=lambda_,
            module=module,
            vf_preds=value_predictions_,
            truncateds=_batch[Columns.TRUNCATEDS],
            terminateds=_batch[Columns.TERMINATEDS],
            rewards=rewards,
            real_seq_lens=_batch[Columns.SEQ_LENS]
            - (1 if with_one_ts_to_episode else 0),
        )

        if Columns.LOSS_MASK in _batch:
            mask = _batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        else:
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(curr_action_dist_logits_)
        prev_action_dist = action_dist_class_exploration.from_logits(
            prev_action_dist_logits_
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(taken_actions_) - prev_actions_logp_
        )
        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        if config.use_critic:
            vf_loss = torch.pow(value_predictions_ - value_targets, 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[
                    PPO_AGENT_POLICY_ID
                ].get_current_value()
                * curr_entropy
            )
        )

        if config.use_kl_loss:
            total_loss += (
                self.curr_kl_coeffs_per_module[PPO_AGENT_POLICY_ID] * mean_kl_loss
            )

        # TODO: do correct metrics after the loss is correct
        # self.metrics.log_dict(
        #     {
        #         POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
        #         VF_LOSS_KEY: mean_vf_loss,
        #         LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
        #         LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
        #             batch[Postprocessing.VALUE_TARGETS], value_fn_out
        #         ),
        #         ENTROPY_KEY: mean_entropy,
        #         LEARNER_RESULTS_KL_KEY: mean_kl_loss,
        #     },
        #     key=PPO_AGENT_POLICY_ID,
        #     window=1,  # <- single items (should not be mean/ema-reduced over time).
        # )
        return total_loss

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        raise NotImplementedError("This should not be used. Use compute_losses instead")
