"""
Baseline timing policies used to benchmark the PPO policy.

Spec baselines (Section J1):
  1. AlwaysExplain  — delta_t = 1 every step (spam baseline)
  2. NeverExplain   — delta_t = 0 every step (silence baseline)
  3. RuleBasedGate  — delta_t = 1 if (Benefit - alpha * WorkloadCost) >= lambda
"""

from src.rl.reward import RewardCalculator, _RARE_ACTIONS


class AlwaysExplainPolicy:
    """Always requests an explanation. Blocked by the shield as needed."""

    def predict(self, _obs) -> int:
        return 1


class NeverExplainPolicy:
    """Never requests an explanation."""

    def predict(self, _obs) -> int:
        return 0


class RuleBasedGatePolicy:
    """
    Deterministic gate: explain if the net benefit exceeds a threshold.

        score = Benefit_t - alpha * WorkloadCost_t
        delta_t = 1 if score >= lambda_threshold else 0

    Identical benefit/cost definitions to the RL reward for a fair comparison.
    """

    def __init__(self, cfg: dict, lambda_threshold: float = 0.3):
        r = cfg["reward"]
        self.alpha     = r["alpha"]
        self.threshold = lambda_threshold

    def predict_from_record(self, rec: dict) -> int:
        theta   = float(rec.get("theta_t", 0.0))
        action  = rec.get("a_drive_t", "cruise")
        w_t     = float(rec.get("w_t", 0.0))

        benefit      = 0.5 * theta + 0.5 * (1.0 if action in _RARE_ACTIONS else 0.0)
        workload_cost = w_t
        score = benefit - self.alpha * workload_cost

        return 1 if score >= self.threshold else 0

    def predict(self, _obs) -> int:
        # obs-based prediction not applicable for rule-based; use predict_from_record
        return 0