### Manual Control

```
 python .venv/lib/python3.12/site-packages/minigrid/manual_control.py
```

### Important Links:

-   [Gym Wrappers](https://gymnasium.farama.org/api/wrappers/)

# Docs RLlib
- [Config](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.html)
- [Evaluation](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html)
- [Training](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html)
- [Customizing of Model](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.core.rl_module.default_model_config.DefaultModelConfig.html)

# Implementation Examples (for new API-Stack)
- [Custom Loss Function](https://github.com/ray-project/ray/blob/master/rllib/examples/learners/classes/custom_ppo_loss_fn_learner.py)
- [Custom IntrinsicReward Learner](https://github.com/ray-project/ray/blob/master/rllib/examples/learners/classes/intrinsic_curiosity_learners.py)
- [Custom IntrinsicReward RLModule](https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/intrinsic_curiosity_model_rlm.py)
- [Recurrent RLModule (LSTM) with InnerState](https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py)
- [Connectors(Custom TrainingBatches with PrevActPrevRew)](https://github.com/ray-project/ray/blob/master/rllib/examples/connectors/prev_actions_prev_rewards.py)

