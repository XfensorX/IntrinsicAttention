"""
RL Project Note: We used the existing code for the environment from
(https://github.com/twni2016/Memory-RL/tree/main) and made modifications.

Rewards
    All per-step shaping is removed. No step penalties and no distractors.
    Reward is terminal only:
    Default: 0/1 scheme → +1 if the final choice matches the cue, else 0.
    Optional: ±1 scheme (toggle via terminal_plus_minus=True) → +1 correct, -1 wrong.
    This makes the metric a pure memory score (either the agent recalls the cue or it doesn’t).

Action model
    Action space reduced to 3 actions: forward, up, down.
    Before the junction:
        Only forward actually moves the agent along the corridor.
        Up/down are no-ops (ignored).
    At the junction (final decision step):
        Up = choose top goal (y=+1), Down = choose bottom goal (y=-1).
        Forward at the junction is a no-op (agent cannot avoid making a choice).
    This guarantees the agent always reaches the decision point and must commit to a goal.

Episode horizon constraint
    Enforced: episode_length == corridor_length + 1.
    This ensures exactly L forward steps to reach the junction, then the final step is the decision.
    No slack time, preventing accidental shaping or late penalties.

Ambiguous corridor (state aliasing)
    ambiguous_position=True remains active. All corridor observations look identical (no position info).
    The cue (correct goal: top/bottom) is only shown at the oracle at t=0.
    This isolates cue memory: the agent must remember that initial signal until the last step.

Time feature to remove counting requirement
    add_timestep=True adds a normalized countdown time_to_live ∈ [0, 1] to the observation.
    This indicates “where we are in the episode” without requiring the agent to learn to count steps,
    so we measure memory for the cue, not step-counting ability.
"""

import numpy as np
import gym
from typing import Any, Dict, Mapping, Optional


def create_env_tmaze(env_config: Optional[Mapping[str, Any]] = None):
    """Factory: sets sensible defaults for memory-only variant."""
    defaults: Dict[str, Any] = {
        "episode_length": 200,
        "corridor_length": 199,
        "oracle_length": 0,
        "goal_reward": 1.0,  # terminal reward
        "terminal_plus_minus": False,  # False: 0/1; True: +/-1 reward scheme
        "ambiguous_position": True,
        "expose_goal": False,  # don't show goal
        "add_timestep": True,  # provide timestep feature
    }
    if env_config:
        defaults.update(env_config)

    L = int(defaults["corridor_length"])
    T = int(defaults["episode_length"])
    assert T == L + 1, "episode_length must be corridor_length + 1"

    return TMazeMemoryOnly(
        episode_length=T,
        corridor_length=L,
        oracle_length=int(defaults["oracle_length"]),
        goal_reward=float(defaults["goal_reward"]),
        terminal_plus_minus=bool(defaults.get("terminal_plus_minus", False)),
        ambiguous_position=bool(defaults["ambiguous_position"]),
        expose_goal=bool(defaults["expose_goal"]),
        add_timestep=bool(defaults["add_timestep"]),
    )


class TMazeMemoryOnly(gym.Env):
    """
    Memory-only T-Maze:
      - Corridor length L, Episode length T = L+1.
      - t=0: Start (Oracle/Cue visible: goal_y in {+1,-1}).
      - t=1..L-1: Move forward (Up/Down are no-ops).
      - t=L: Agent chooses Up/Down (final decision) -> episode ends.
      - Reward: Terminal only (+1/0 or +1/-1).
      - Observation: Ambiguity in the corridor optional (ambiguous_position=True),
        plus optional normalized time feature.
    Actions:
      0: forward (x+1)
      1: up (y+1) -> only possible at junction
      2: down (y-1) -> only possible at junction
    """

    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        terminal_plus_minus: bool = False,  # False: 0/1, True: +/-1
        ambiguous_position: bool = True,
        expose_goal: bool = False,
        add_timestep: bool = True,
    ):
        super().__init__()
        assert (
            corridor_length >= 1 and episode_length == corridor_length + 1
        ), "For memory-only set episode_length == corridor_length + 1"
        self.T = episode_length
        self.L = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = float(goal_reward)
        self.terminal_plus_minus = bool(terminal_plus_minus)

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        # Actions: forward, up, down
        self.action_space = gym.spaces.Discrete(3)
        self._action_mapping = {
            0: (1, 0),  # forward
            1: (0, 1),  # up -> only at junction
            2: (0, -1),  # down -> only at junction
        }

        # Build map: corridor + goals
        self.bias_x, self.bias_y = 1, 2
        width = self.oracle_length + self.L + 1 + 2
        height = 3 + 2
        self.tmaze_map = np.zeros((height, width), dtype=bool)
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor row
        self.tmaze_map[[self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1] = (
            True  # goals
        )

        # Observation space
        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:
            assert (
                not self.ambiguous_position
            ), "expose_goal makes env Markov; not for memory-only."
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ------------- Core -------------
    def _reset_state(self):
        self.time_step = 0
        self.x, self.y = self.oracle_length, 0  # start at oracle_length, y=0
        self.goal_y = int(np.random.choice([+1, -1]))  # cue: up or down
        self.oracle_visited = False
        self._done = False

    def reset(self):
        self._reset_state()
        return self.get_obs()

    def step(self, action: int):
        assert self.action_space.contains(action)
        if self._done:
            raise RuntimeError("step() after done. Call reset().")

        self.time_step += 1
        at_junction = self.x == self.oracle_length + self.L
        dx, dy = self._action_mapping[action]

        if not at_junction:
            # Before junction: only forward has effect
            dx, dy = (1, 0)
            if self._is_valid(self.x + dx, self.y + dy):
                self.x += dx
                self.y += dy
            reward = 0.0
            done = self.time_step >= self.T
            obs = self.get_obs()
            self._done = done
            return obs, float(reward), bool(done), {}

        # At junction: forward is no-op; up/down are final choice
        decided_y = self.y
        if action == 1:
            decided_y = +1
        elif action == 2:
            decided_y = -1

        done = True
        correct = decided_y == self.goal_y
        if self.terminal_plus_minus:
            reward = self.goal_reward if correct else -self.goal_reward
        else:
            reward = self.goal_reward if correct else 0.0

        self.y = decided_y
        obs = self.get_obs()
        self._done = True
        return obs, float(reward), bool(done), {}

    def _is_valid(self, nx, ny) -> bool:
        return bool(self.tmaze_map[self.bias_y + ny, self.bias_x + nx])

    # ------------- Observations -------------
    def position_encoding(self, x: int, y: int, goal_y: int):
        # t=0: Oracle/Cue visible (once)
        if x == 0:
            if not self.oracle_visited:
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                return [0, exposure]  # oracle with cue
            elif x < self.oracle_length + self.L:
                return [0, 0]  # indistinguishable corridor
            else:
                return [1, y]  # junction/goals
        else:
            if self.expose_goal:
                return [x, y, goal_y if self.oracle_visited else 0]
            if x == 0:
                return [x, y, exposure]
            else:
                return [x, y, 0]

    def timestep_encoding(self):
        if not self.add_timestep:
            return []
        t_norm = 1.0 - self.time_step / float(self.T)
        return [t_norm]

    def get_obs(self):
        vec = (
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding()
        )
        return np.array(vec, dtype=np.float32)
