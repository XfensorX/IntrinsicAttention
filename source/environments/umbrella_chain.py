# umbrella_chain_gymnasium.py
# -----------------------------------------------------------------------------
# MIT-style rewrite of DeepMind’s “Umbrella Chain” (bsuite) environment
# for the Gymnasium API.  Heavily commented for clarity.
#
# Original Source: https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/umbrella_chain.py
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UmbrellaChainEnv(gym.Env):
    """
    A simple diagnostic credit-assignment challenge.

    ▸ Observation (float32 Box):
        index 0 : need_umbrella  ∈ {0,1}
        index 1 : have_umbrella  ∈ {0,1}
        index 2 : countdown      ∈ [0,1]   –– 1 at t=0, 0 just before final step
        index 3+: n_distractor   ∈ {0,1}   –– i.i.d. Bernoulli(0.5) noise

    ▸ Action (Discrete(2)):
        0 ≙ *do not* pick up umbrella
        1 ≙ *pick up* umbrella
        Only the very first action (t = 1) has any effect; afterwards actions are ignored.

    ▸ Reward:
        * Intermediate steps (t < chain_length):   +1 or −1, 50 % chance each (distractor).
        * Final step (t == chain_length):
              +1  if have_umbrella == need_umbrella
              −1  otherwise
          (The agent therefore wants to match need_umbrella, which is hidden until the end.)

    ▸ Episode length = chain_length  (no truncation).
    """

    metadata = {"render_modes": []}  # no visualiser

    # --------------------------------------------------------------------- #
    # Constructor & helpers                                                 #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        chain_length: int,
        n_distractor: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args
        ----
        chain_length : int      – number of time-steps per episode
        n_distractor : int      – number of binary noise features appended to the observation
        seed         : Optional – seed for NumPy’s RNG (determinism / reproducibility)
        """
        super().__init__()

        # Basic parameters ------------------------------------------------- #
        self.chain_length = chain_length
        self.n_distractor = n_distractor
        self.rng = np.random.RandomState(seed)

        # Gymnasium spaces -------------------------------------------------- #
        obs_dim = 3 + n_distractor  # total length of flat observation vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # {0, 1}

        # Episode-specific state (initialised in reset) --------------------- #
        self._timestep: int = 0  # current time-step (0-based inside Gymnasium)
        self._need_umbrella: int = 0  # ground-truth requirement (sampled each episode)
        self._has_umbrella: int = 0  # agent’s chosen value (set only at t = 1)

        # Performance statistics (optional) -------------------------------- #
        self._total_regret: float = 0.0  # counts how far from optimal the agent was

    # --------------------------------------------------------------------- #
    # Gymnasium API                                                         #
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment for a new episode.

        Returns
        -------
        observation : np.ndarray  – shape (obs_dim,)
        info        : dict        – contains cumulative statistics (can be empty)
        """
        # Respect *external* seeding requests (Gymnasium convention) -------- #
        if seed is not None:
            self.rng.seed(seed)

        # Episode-specific initialisation ----------------------------------- #
        self._timestep = 0
        self._need_umbrella = self.rng.binomial(1, 0.5)  # hidden coin-flip
        self._has_umbrella = self.rng.binomial(1, 0.5)  # random initial guess

        observation = self._get_observation()
        info = {"total_regret": self._total_regret}  # could also be {}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advances the environment by one time-step.

        Parameters
        ----------
        action : int  – must be 0 or 1.  Only first step matters.

        Returns
        -------
        observation : np.ndarray  – next observation
        reward      : float       – reward obtained at this step
        terminated  : bool        – True iff episode finished *normally*
        truncated   : bool        – True iff episode stopped early (never here)
        info        : dict        – diagnostics (empty)
        """
        # ------------------------- Core transition logic ------------------ #
        self._timestep += (
            1  # Gymnasium expects _timestep to be incremented *before* reward
        )

        # On the *first* step only, allow agent to pick an umbrella -------- #
        if self._timestep == 1:
            # Clip defensive programming: makes sure  action ∈ {0,1}
            action = int(action) & 1
            self._has_umbrella = action

        # ------------------------------------------------------------------ #
        # Determine reward & termination flag                                #
        # ------------------------------------------------------------------ #
        terminated = self._timestep >= self.chain_length
        truncated = False  # no external time-limit

        if terminated:
            # Final step – reward depends on matching umbrella state
            correct = int(self._has_umbrella == self._need_umbrella)
            reward = 1.0 if correct else -1.0

            # Track regret: +2 if agent was wrong (optimal would be +1 vs −1)
            if not correct:
                self._total_regret += 2.0
        else:
            # Intermediate distractor reward (+1 or −1 with equal prob.)
            reward = 2.0 * self.rng.binomial(1, 0.5) - 1.0

        # Generate next observation (always even on terminal step) --------- #
        observation = self._get_observation()
        info: Dict = (
            {}
        )  # Could add `"need_umbrella": self._need_umbrella` for debugging

        return observation, float(reward), terminated, truncated, info

    # Gymnasium’s render() & close() are optional – omitted for brevity.

    # --------------------------------------------------------------------- #
    # Helper utilities                                                      #
    # --------------------------------------------------------------------- #
    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current observation vector.

        • index 0 : Whether an umbrella will be needed at the *end* of episode
        • index 1 : Whether the agent currently holds an umbrella
        • index 2 : Normalised “time to live”  (1 at start → 0 at final step)
        • index 3+: Independent distractor bits
        """
        obs_len = 3 + self.n_distractor
        obs = np.zeros(obs_len, dtype=np.float32)

        # Core informative features --------------------------------------- #
        obs[0] = float(self._need_umbrella)
        obs[1] = float(self._has_umbrella)
        obs[2] = 1.0 - self._timestep / self.chain_length

        # Distractors: fresh Bernoulli(0.5) every step --------------------- #
        if self.n_distractor:
            obs[3:] = self.rng.binomial(1, 0.5, size=self.n_distractor)

        return obs

    # --------------------------------------------------------------------- #
    # Optional convenience properties                                       #
    # --------------------------------------------------------------------- #
    @property
    def optimal_return(self) -> int:
        """Maximum possible episodic return (always +1)."""
        return 1

    @property
    def total_regret(self) -> float:
        """
        Cumulative regret across *all* episodes so far.

        Regret is counted only when the agent miss-matches the umbrella
        requirement at the final step (difference of 2 reward points).
        """
        return self._total_regret
