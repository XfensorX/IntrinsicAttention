import numpy as np
# import gym
import matplotlib.pyplot as plt
import os
from typing import Any, Dict, Mapping, Optional

"""
T-Maze: originated from (Bakker, 2001) and earlier neuroscience work, 
    and here extended to unit-test several key challenges in RL:
- Exploration
- Memory and credit assignment
- Discounting and distraction
- Generalization

Finite horizon problem: episode_length
Has a corridor of corridor_length
Looks like
                        g1
o--s---------------------j
                        g2
o is the oracle point, (x, y) = (0, 0)
s is starting point, (x, y) = (o, 0)
j is T-juncation, (x, y) = (o + corridor_length, 0)
g1 is goal candidate, (x, y) = (o + corridor_length, 1)
g2 is goal candidate, (x, y) = (o + corridor_length, -1)

RL Project Note: We used the exisiting Code for the Env from the (https://github.com/twni2016/Memory-RL/tree/main) repo and added a few changes
"""

# TODO. Sicherstellen das Agent immer nur nach rechts gehen kann!!! Damit er sich in jeder Episode bei J entscheiden muss
# und so immer entweder 1 oder 0 kriegt. Ein random Agent der zufällig entscheidet würde dann immer auf 0.5 kommen -> Baseline


def create_env_tmaze(env_config: Optional[Mapping[str, Any]] = None):

    defaults: Dict[str, Any] = {
        "episode_length": 200,  # int
        "corridor_length": 199,  # int
        "oracle_length": 0,  # int
        "goal_reward": 1.0,  # float
        "penalty": 0.0,  # float
        "distract_reward": 0.0,  # float
        "ambiguous_position": True,  # bool
        "expose_goal": False,  # bool
        "add_timestep": True,  # bool
    }
    if env_config:
        defaults.update(env_config)

    return TMazeBase(
        episode_length=int(defaults["episode_length"]),
        corridor_length=int(defaults["corridor_length"]),
        oracle_length=int(defaults["oracle_length"]),
        goal_reward=float(defaults["goal_reward"]),
        penalty=float(defaults["penalty"]),
        distract_reward=float(defaults["distract_reward"]),
        ambiguous_position=bool(defaults["ambiguous_position"]),
        expose_goal=bool(defaults["expose_goal"]),
        add_timestep=bool(defaults["add_timestep"]),
    )


class TMazeBase(gym.Env):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,  # In config Datei: penalty=-1.0 / corridor_length,  # NOTE: \sum_{t=1}^T -1/T = -1
        distract_reward: float = 0.0,
        ambiguous_position: bool = True,
        expose_goal: bool = False,
        add_timestep: bool = True,
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2016): distract_reward = eps > 0, goal_reward is given at T-junction (no choice).
                This only tests the exploration and discounting of agent, no memory required
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(
            4
        )  # four directions #TODO: Hier changen, dass agent nur nach rechts gehen kann und bei J nach oben oder unten???
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[[self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1] = (
            True  # goal candidates
        )
        print(self.tmaze_map.astype(np.int32))

        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:  # test Markov policies
            assert self.ambiguous_position is False
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            # oracle position
            if not self.oracle_visited:
                # only appear at first
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                # oracle position
                return [0, exposure]
            elif x < self.oracle_length + self.corridor_length:
                # intermediate positions (on the corridor)
                return [0, 0]
            else:
                # T-junction or goal candidates
                return [1, y]
        else:
            if self.expose_goal:
                return [x, y, goal_y if self.oracle_visited else 0]
            if x == 0:
                # oracle position
                return [x, y, exposure]
            else:
                return [x, y, 0]

    def timestep_encoding(self):
        return (
            [
                self.time_step,
            ]
            if self.add_timestep
            else []
        )

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:  # only give bonus at the final time step
            return float(y == goal_y) * self.goal_reward
        else:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward
            else:
                return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = self.x + move_x, self.y + move_y

        done = self.time_step >= self.episode_length
        rew = self.reward_fn(done, self.x, self.y, self.goal_y)
        return self.get_obs(), rew, done, {}

    def reset(self):
        self.x, self.y = self.oracle_length, 0
        self.goal_y = np.random.choice([-1, 1])

        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs()

    """
    def visualize(self, trajectories: np.array, idx: str):
        from utils import logger

        # trajectories: (B, T+1, O)
        batch_size, seq_length, _ = trajectories.shape
        xs = np.arange(seq_length)

        for traj in trajectories:
            # plot the 0-th element
            plt.plot(xs, traj[:, 0])

        plt.xlabel("Time Step")
        plt.ylabel("Position X")
        plt.savefig(
            os.path.join(logger.get_dir(), "plt", f"{idx}.png"),
            dpi=200,  # 200
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
    """
