from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import NormalizeObservation

ENV_NAME = "MiniGrid-DoorKey-5x5-v0"


def evaluate() -> None:
    def create_env(config):
        import minigrid

        env = gym.make(ENV_NAME)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
        env = NormalizeObservation(env)
        return env

    tune.register_env(ENV_NAME, create_env)

    config = PPOConfig().environment(ENV_NAME).framework("torch")
    config["seed"] = 42
    config["num_env_runners"] = 8

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"num_env_steps_sampled_lifetime": 100000},
        ),
    )

    results = tuner.fit()

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
