import gymnasium as gym
from gymnasium.wrappers import FilterObservation
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from .model import IntrinsicAttentionPPOModel

ENV_NAME = "MiniGrid-DoorKey-5x5-v0"


def evaluate() -> None:
    def create_env(config):
        import minigrid

        env = gym.make(ENV_NAME)
        env = FilterObservation(env, ["direction", "image"])
        return env

    tune.register_env(ENV_NAME, create_env)
    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IntrinsicAttentionPPOModel,
                model_config={},  # TODO: later necessary, maybe HPO
            ),
        )
    )

    # config = (
    #     config.training(train_batch_size=4000)
    #     .env_runners(
    #         num_envs_per_env_runner=4, num_env_runners=1, rollout_fragment_length="auto"
    #     )
    #     .resources(num_gpus=0)

    # )
    # config["seed"] = 42
    # config["num_env_runners"] = 8

    algo = config.build()
    for _ in range(2):
        result = algo.train()
        print(result["episode_reward_mean"])
    algo.stop()

    # tuner = tune.Tuner(
    #     "PPO",
    #     param_space=config,
    #     run_config=tune.RunConfig(
    #         stop={"num_env_steps_sampled_lifetime": 8000},
    #     ),
    # )
    # results = tuner.fit()

    # # Build the PPO agent
    # agent = config.build()

    # timesteps = 0
    # while timesteps < cfg.max_timesteps:
    #     # Perform training
    #     result = agent.train()
    #     timesteps = result["num_env_steps_sampled_lifetime"]

    # evaluation_results = agent.evaluate()
    # final_reward = evaluation_results["env_runners"]["episode_return_mean"]

    # print(f"Final evaluation reward: {final_reward}")


if __name__ == "__main__":
    # Run the evaluation function with the provided configuration
    evaluate()
