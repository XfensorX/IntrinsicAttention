import pprint

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import DtypeObservation, FlattenObservation
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from source.proposal.full_episode_connector import FullEpisodeConnector
from source.proposal.models.model import IntrinsicAttentionPPOModel

ENV_NAME = "MiniGrid-DoorKey-5x5-v0"


def evaluate() -> None:
    def create_env(config):
        import minigrid  # noqa

        env = gym.make(ENV_NAME)
        # FIXME: This is a hack to make the env work with RLlib
        # env = FilterObservation(env, ["direction", "image"])
        env = OneHotPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env = DtypeObservation(env, dtype=np.float32)
        env = FlattenObservation(env)

        return env

    tune.register_env(ENV_NAME, create_env)
    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .framework("torch")
        # TODO: full episodes
        .env_runners(
            batch_mode="complete_episodes",
            rollout_fragment_length=500,
            num_envs_per_env_runner=1,
            num_env_runners=1,
            num_cpus_per_env_runner=4,
            num_gpus_per_env_runner=0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IntrinsicAttentionPPOModel,
                model_config={
                    "obs_embed_dim": 64,
                    "pre_head_embedding_dim": 256,
                    "gru_hidden_size": 256,
                    "gru_num_layers": 2,
                    "attention_v_dim": 32,
                    "attention_qk_dim": 32,
                    "max_seq_len": 251,
                },
                action_space=create_env(None).action_space,
                observation_space=create_env(None).observation_space,
            ),
        )
        .training(
            train_batch_size_per_learner=4096,
            num_epochs=10,
            minibatch_size=500,
            learner_connector=lambda obs, action: FullEpisodeConnector(obs, action),
        )
        .learners(num_gpus_per_learner=1, num_learners=1)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"num_env_steps_sampled_lifetime": 80000},
        ),
    )
    results = tuner.fit()
    pprint.pprint(results.get_dataframe())


if __name__ == "__main__":
    evaluate()
