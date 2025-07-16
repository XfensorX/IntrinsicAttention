import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from source.proposal.models.model import IntrinsicAttentionPPOModel

ENV_NAME = "MiniGrid-DoorKey-5x5-v0"


def evaluate() -> None:
    def create_env(config):
        import minigrid  # noqa

        env = gym.make(ENV_NAME)
        # FIXME: This is a hack to make the env work with RLlib
        # env = FilterObservation(env, ["direction", "image"])
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)

        return env

    tune.register_env(ENV_NAME, create_env)
    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .framework("torch")
        .env_runners(num_envs_per_env_runner=1, num_env_runners=1)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IntrinsicAttentionPPOModel(
                    model_config={
                        "obs_dim": 147,
                        "obs_embed_dim": 16,
                        "pre_head_embedding_dim": 64,
                        "gru_hidden_size": 64,
                        "gru_num_layers": 4,
                        "attention_v_dim": 32,
                        "attention_qk_dim": 32,
                    },
                    action_space=create_env(None).action_space,
                ),
            ),
        )
        .training()
        .resources(num_gpus=0)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"num_env_steps_sampled_lifetime": 8000},
        ),
    )
    results = tuner.fit()
    print(results)


if __name__ == "__main__":
    evaluate()
