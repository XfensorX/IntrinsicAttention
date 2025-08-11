# umbrella_chain_gymnasium_optimized.py
# -----------------------------------------------------------------------------
# MIT-style rewrite of DeepMind’s “Umbrella Chain” (bsuite) environment
# for the Gymnasium API.  Heavily commented for clarity.
#
# Original Source: https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/umbrella_chain.py
#
# Notes on this version:
# - Same semantics as the original file you shared, with micro-optimizations to
#   reduce RAM churn and per-step allocations:
#     • Preallocated observation buffer written in-place
#     • Shared empty info dict (instead of new {} each step)
#     • Precomputed countdown lookup (values match original float32 rounding)
#     • NumPy Generator + in-place integer buffer for distractors (no temps)
#     • Optional copy-on-return (copy_obs) for safety if callers mutate obs
#     • __slots__ to shrink instance size (handy for many env instances)
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
    __slots__ = (
        "chain_length",
        "n_distractor",
        "rng",
        "observation_space",
        "action_space",
        "_timestep",
        "_need_umbrella",
        "_has_umbrella",
        "_total_regret",
        # Preallocated / cached structures for performance:
        "_obs",
        "_obs_core_view",
        "_obs_distr_view",
        "_countdown_lut",
        "_distr_int_buf",
        "copy_obs",
        "_EMPTY_INFO",
    )

    def __init__(
        self,
        chain_length: int,
        n_distractor: int = 0,
        seed: Optional[int] = None,
        *,
        copy_obs: bool = False,  # Return a copy of obs on each API call (safer but slower).
    ) -> None:
        """
        Args
        ----
        chain_length : int      – number of time-steps per episode
        n_distractor : int      – number of binary noise features appended to the observation
        seed         : Optional – seed for NumPy’s RNG (determinism / reproducibility)
        copy_obs     : bool     – if True, .reset()/.step() return a copy of the internal
                                  observation buffer (prevents accidental downstream mutation).
        """
        super().__init__()

        # Basic parameters ------------------------------------------------- #
        self.chain_length = int(chain_length)
        self.n_distractor = int(n_distractor)
        self.copy_obs = bool(copy_obs)
        # Use modern Generator for in-place sampling with "out=" buffers
        self.rng = np.random.default_rng(seed)

        # Gymnasium spaces ------------------------------------------------- #
        obs_dim = 3 + self.n_distractor  # total length of flat observation vector
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

        # ---------------- Performance-focused preallocations ----------------
        # Preallocate observation buffer and convenient views
        self._obs = np.empty(obs_dim, dtype=np.float32)
        self._obs_core_view = self._obs[:3]
        self._obs_distr_view = self._obs[3:] if self.n_distractor else None

        # Small, reusable integer buffer for distractor sampling (0/1 ints)
        # We write ints here (out=...), then cast into the float32 view without temp arrays.
        self._distr_int_buf = (
            np.empty(self.n_distractor, dtype=np.uint8) if self.n_distractor else None
        )

        # Precompute countdown values with *the same rounding* as the original:
        # original formula was: float32(1.0 - t / chain_length), for t = 0..chain_length
        self._countdown_lut = np.array(
            [
                np.float32(1.0 - (t / self.chain_length))
                for t in range(self.chain_length + 1)
            ],
            dtype=np.float32,
        )

        # Shared empty info dict to avoid per-step {} allocation
        self._EMPTY_INFO: Dict = {}

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
            # Recreate the Generator with the provided seed (deterministic run)
            self.rng = np.random.default_rng(seed)

        # Episode-specific initialisation ----------------------------------- #
        self._timestep = 0
        self._need_umbrella = int(self.rng.integers(0, 2))  # hidden coin-flip
        self._has_umbrella = int(self.rng.integers(0, 2))  # random initial guess

        # Build initial observation (in-place) ------------------------------ #
        self._write_observation()

        # Keep the same info contract as your original version
        info = {"total_regret": self._total_regret}
        return (self._obs.copy() if self.copy_obs else self._obs), info

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
        # Gymnasium expects _timestep to be incremented *before* reward
        t = self._timestep + 1
        self._timestep = t

        # On the *first* step only, allow agent to pick an umbrella -------- #
        if t == 1:
            # Clip defensive programming: makes sure  action ∈ {0,1}
            self._has_umbrella = int(action) & 1

        # ------------------------------------------------------------------ #
        # Determine reward & termination flag                                #
        # ------------------------------------------------------------------ #
        terminated = t >= self.chain_length
        truncated = False  # no external time-limit

        if terminated:
            # Final step – reward depends on matching umbrella state
            correct = int(self._has_umbrella == self._need_umbrella)
            reward = 10.0 if correct else -10.0

            # Track regret: +2 if agent was wrong (optimal would be +1 vs −1)
            if not correct:
                self._total_regret += 2.0
        else:
            # Intermediate distractor reward (+1 or −1 with equal prob.)
            reward = float((self.rng.integers(0, 2) * 2 - 1) * 0.1)

        # Generate next observation (always even on terminal step) --------- #
        self._write_observation()
        return (
            (self._obs.copy() if self.copy_obs else self._obs),
            reward,
            terminated,
            truncated,
            self._EMPTY_INFO,
        )

    # Gymnasium’s render() & close() are optional – omitted for brevity.

    # --------------------------------------------------------------------- #
    # Helper utilities                                                      #
    # --------------------------------------------------------------------- #
    def _write_observation(self) -> None:
        """
        Constructs the current observation vector (in-place).

        • index 0 : Whether an umbrella will be needed at the *end* of episode
        • index 1 : Whether the agent currently holds an umbrella
        • index 2 : Normalised “time to live”  (1 at start → 0 at final step)
        • index 3+: Independent distractor bits
        """
        core = self._obs_core_view
        # Core informative features --------------------------------------- #
        core[0] = float(self._need_umbrella)
        core[1] = float(self._has_umbrella)
        # Use LUT (matches original float32 rounding exactly)
        t = self._timestep
        core[2] = (
            self._countdown_lut[t] if t < self._countdown_lut.size else np.float32(0.0)
        )

        # Distractors: fresh Bernoulli(0.5) every step --------------------- #
        if self._obs_distr_view is not None:
            # 1) Fill reusable integer buffer with {0,1} using in-place RNG
            self.rng.integers(
                0, 2, size=self._distr_int_buf.shape, out=self._distr_int_buf
            )
            # 2) Cast/copy into the float32 observation view without allocating temporaries
            #    (multiply-by-1.0 with unsafe casting writes float32 in-place)
            np.multiply(
                self._distr_int_buf, 1.0, out=self._obs_distr_view, casting="unsafe"
            )

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
