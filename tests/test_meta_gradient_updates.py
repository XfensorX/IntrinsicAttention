# from intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPO import IntrinsicAttentionPPO
# from source.intrinsic_attention_ppo.algorithm.IntrinsicAttentionPPOConfig import (
#     IntrinsicAttentionPPOConfig,
# )
# from source.intrinsic_attention_ppo.environments.umbrella_chain import create_env


def test_correct_gradient_updates():
    """
    Tests the update of a meta gradient step,
    In the first step, the parameters of the PPO model should change, but no intrinsic-reward-parameters
    In the second step vice versa
    """
    # TODO: seeding and actually build this
    # tune.register_env("Umbrella", create_env)
    #
    # config = IntrinsicAttentionPPOConfig()
    # config.environment("Umbrella")
    # algo: IntrinsicAttentionPPO = config.build_algo()
    #
    # ex_batch = algo.custom_sample_batch()
    # ex_in_batch = algo.custom_add_intrinsic_rewards(copy.deepcopy(ex_batch))
    # learner_results = algo.custom_ppo_with_intrinsic_update(ex_in_batch)
    #
    # algo.custom_sync_weights(learner_results)
    #
    # # TODO: philipp, somehow the ppo network gets updated in here, with this step, we need to stop this.
    # # TODO: philipp, the intrinsic network on the other hand, does not change
    # #               (most likely due to the missing value head gradients)
    # learner_results = algo.custom_meta_gradient_update(
    #     met_step_batch=ex_batch, inner_step_batch=ex_in_batch
    # )
    #
    # algo.custom_sync_weights(learner_results)
