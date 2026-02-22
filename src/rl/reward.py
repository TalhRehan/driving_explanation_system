"""
Reward function for the timing policy.

r_t = delta_exec_t * Benefit_t
      - alpha  * WorkloadCost_t
      - gamma  * FreqPenalty_t
      - eta    * StalePenalty_t
"""

import math
from collections import deque


# Discrete action labels that count as rare / important driving events
_RARE_ACTIONS = {"brake", "yield", "turn_left", "turn_right"}


class RewardCalculator:
    """
    Stateful reward calculator — maintains the rolling explanation
    counter required for the frequency budget.

    Parameters
    ----------
    cfg : dict
        Full project config (reward sub-dict read internally).
    """

    def __init__(self, cfg: dict):
        r = cfg["reward"]
        self.alpha      = r["alpha"]
        self.gamma_freq = r["gamma_freq"]
        self.eta        = r["eta"]
        self.rho        = r["rho"]
        self.budget     = r["budget_per_60s"]

        self._window_sec = 60.0
        # Each entry is a timestamp of a displayed explanation
        self._explanation_times: deque[float] = deque()

    def reset(self) -> None:
        self._explanation_times.clear()

    # ------------------------------------------------------------------

    def compute(self, delta_exec: int, rec: dict, timestamp: float) -> float:
        """
        Compute the scalar reward for one timestep.

        Parameters
        ----------
        delta_exec : int
            Effective explanation decision after shield (0 or 1).
        rec : dict
            Full state record for this timestep.
        timestamp : float
            Current episode time in seconds.
        """
        # --- Purge old entries outside the rolling window ---------------
        cutoff = timestamp - self._window_sec
        while self._explanation_times and self._explanation_times[0] < cutoff:
            self._explanation_times.popleft()

        # --- Benefit ----------------------------------------------------
        benefit = self._benefit(rec)

        # --- WorkloadCost -----------------------------------------------
        workload_cost = self._workload_cost(rec)

        # --- FreqPenalty ------------------------------------------------
        freq_penalty = self._freq_penalty(delta_exec, timestamp)

        # --- StalePenalty (deferred explanations) -----------------------
        stale_penalty = self._stale_penalty(rec)

        reward = (
            delta_exec * benefit
            - self.alpha      * workload_cost
            - self.gamma_freq * freq_penalty
            - self.eta        * stale_penalty
        )

        # Track displayed explanations for the budget window
        if delta_exec == 1:
            self._explanation_times.append(timestamp)

        return float(reward)

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def _benefit(self, rec: dict) -> float:
        """
        Benefit is high when the agent explains during an informative moment:
        high theta_t (rare/uncertain event) or an important maneuver label.
        """
        theta    = float(rec.get("theta_t", 0.0))
        action   = rec.get("a_drive_t", "cruise")
        is_rare  = 1.0 if action in _RARE_ACTIONS else 0.0
        # Blend: both theta and action rarity contribute
        return 0.5 * theta + 0.5 * is_rare

    def _workload_cost(self, rec: dict) -> float:
        """
        Cost of showing an explanation — higher in more critical contexts.
        Uses the pre-computed workload proxy w_t.
        """
        return float(rec.get("w_t", 0.0))

    def _freq_penalty(self, delta_exec: int, timestamp: float) -> float:
        """
        Apply a penalty when the rolling-window explanation count exceeds budget.
        Returns 0 if under budget, 1 if at/over budget.
        """
        if delta_exec == 0:
            return 0.0
        n_recent = len(self._explanation_times)   # already trimmed above
        if n_recent >= self.budget:
            return 1.0
        return 0.0

    def _stale_penalty(self, rec: dict) -> float:
        """
        Penalise staleness for deferred explanations.
        The record carries an optional 'deferred_delay' field written
        by the inference pipeline when a buffered event is delivered.
        """
        delay = float(rec.get("deferred_delay", 0.0))
        if delay <= 0.0:
            return 0.0
        return 1.0 - math.exp(-self.rho * delay)

    # ------------------------------------------------------------------

    def explanation_count_in_window(self) -> int:
        """Current number of explanations within the rolling 60-second window."""
        return len(self._explanation_times)