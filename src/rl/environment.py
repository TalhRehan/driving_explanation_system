"""
Offline RL environment for the timing policy.

Replays logged CARLA episodes (JSONL) as Gymnasium episodes.
The agent sees a feature vector at each timestep and outputs
a binary action: 1 = request explanation, 0 = stay silent.
"""

import random
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required.  Run: pip install gymnasium"
    )

from src.config_loader import CONFIG
from src.data_collection.episode_loader import discover_episodes, load_episode
from src.rl.obs_space import OBS_KEYS, OBS_SCALE as _OBS_SCALE
from src.rl.reward import RewardCalculator
from src.shield.safety_shield import SafetyShield


class DrivingExplanationEnv(gym.Env):
    """
    Gymnasium environment for the binary explanation-timing policy.

    Parameters
    ----------
    episodes_dir : Path
        Directory containing episode sub-folders (each with log.jsonl).
    cfg : dict
        Full config dict (from config_loader.CONFIG).
    shuffle : bool
        If True, randomise episode order each time all episodes are consumed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes_dir: Path,
        cfg: dict | None = None,
        shuffle: bool = True,
    ):
        super().__init__()
        self.cfg = cfg or CONFIG
        self._episodes_dir = Path(episodes_dir)
        self._shuffle = shuffle

        self._log_paths = discover_episodes(self._episodes_dir)
        if not self._log_paths:
            raise FileNotFoundError(
                f"No episode logs found under {self._episodes_dir}.  "
                "Run carla_recorder.py or demo_generator.py first."
            )

        self._shield = SafetyShield(self.cfg)
        self._reward_calc = RewardCalculator(self.cfg)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(OBS_KEYS),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self._episode_pool: list[Path] = []
        self._records: list[dict] = []
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if not self._episode_pool:
            self._episode_pool = list(self._log_paths)
            if self._shuffle:
                random.shuffle(self._episode_pool)

        log_path = self._episode_pool.pop()
        self._records = load_episode(log_path)
        self._step_idx = 0
        self._reward_calc.reset()
        self._shield = SafetyShield(self.cfg)

        obs = self._make_obs(self._records[0])
        return obs, {"episode_log": str(log_path)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        rec = self._records[self._step_idx]

        ttc   = float(rec.get("ttc_t", self.cfg["shield"]["ttc_inf"]))
        accel = float(rec.get("a_t", 0.0))
        omega = float(rec.get("omega_t", 0.0))
        crit  = int(rec.get("crit_zone", 0))
        ts    = float(rec.get("timestamp", self._step_idx * 0.1))

        delta_exec = self._shield.gate(
            policy_decision=int(action),
            ttc=ttc,
            accel=accel,
            omega=omega,
            crit_zone=crit,
            timestamp=ts,
            action=rec.get("a_drive_t", "cruise"),
            evidence=rec.get("evidence", []),
            context={"speed": rec.get("speed", 0.0), "crit_zone": crit},
        )

        reward = self._reward_calc.compute(
            delta_exec=delta_exec,
            rec=rec,
            timestamp=ts,
        )

        self._step_idx += 1
        terminated = self._step_idx >= len(self._records)

        obs = self._make_obs(
            self._records[-1] if terminated else self._records[self._step_idx]
        )

        info = {
            "delta_exec": delta_exec,
            "safe_blocked": int(action) == 1 and delta_exec == 0,
            "timestamp": ts,
            "action_label": rec.get("a_drive_t"),
        }
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------

    def _make_obs(self, rec: dict) -> np.ndarray:
        raw = np.array(
            [float(rec.get(k, 0.0)) for k in OBS_KEYS],
            dtype=np.float32,
        )
        raw[3] = min(raw[3], _OBS_SCALE[3])   # clip TTC before normalising
        return np.clip(raw / _OBS_SCALE, -1.0, 1.0)