"""
PPO trainer for the explanation timing policy.

Trains a binary timing policy (show / stay silent) on logged CARLA episodes
using Stable-Baselines3 PPO.  Saves the best checkpoint and a final model.

Usage:
    python -m src.rl.ppo_trainer
    python -m src.rl.ppo_trainer --episodes-dir data/episodes --timesteps 200000
"""

import argparse
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    raise ImportError(
        "stable-baselines3 is required.  Run: pip install stable-baselines3"
    )

from src.config_loader import CONFIG
from src.rl.environment import DrivingExplanationEnv


class _SafetyAuditCallback(BaseCallback):
    """
    Logs a warning if the shield ever fails (delta_exec=1 while blocked).
    In a correctly wired pipeline this count must remain zero.
    """

    def __init__(self):
        super().__init__(verbose=0)
        self._violations = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("safe_blocked"):
                self._violations += 1
        return True

    def _on_training_end(self) -> None:
        if self._violations > 0:
            print(f"[WARNING] Shield violations detected during training: {self._violations}")
        else:
            print("[OK] Shield held throughout training — zero violations.")


# ---------------------------------------------------------------------------

def build_env(episodes_dir: Path, cfg: dict, shuffle: bool = True) -> Monitor:
    env = DrivingExplanationEnv(episodes_dir, cfg=cfg, shuffle=shuffle)
    return Monitor(env)


def train(
    episodes_dir: Path,
    model_dir: Path,
    cfg: dict,
    total_timesteps: int,
) -> PPO:
    rl_cfg = cfg["rl"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_env(episodes_dir, cfg, shuffle=True)
    eval_env  = build_env(episodes_dir, cfg, shuffle=False)

    check_env(train_env, warn=True)

    model = PPO(
        policy=rl_cfg["policy"],
        env=train_env,
        learning_rate=rl_cfg["lr"],
        n_steps=rl_cfg["n_steps"],
        batch_size=rl_cfg["batch_size"],
        n_epochs=rl_cfg["n_epochs"],
        gamma=rl_cfg["gamma"],
        verbose=1,
        tensorboard_log=str(model_dir / "tb_logs"),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(model_dir / "eval_logs"),
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(2000, total_timesteps // 10),
            save_path=str(model_dir / "checkpoints"),
            name_prefix="timing_policy",
            verbose=0,
        ),
        _SafetyAuditCallback(),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = model_dir / "timing_policy_final"
    model.save(str(final_path))
    print(f"Training complete.  Final model saved → {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model_path: Path, episodes_dir: Path, cfg: dict, n_episodes: int = 10) -> dict:
    """
    Run the saved policy on held-out episodes and return evaluation metrics.
    """
    model = PPO.load(str(model_path))
    env   = build_env(episodes_dir, cfg, shuffle=False)

    metrics = {
        "shield_violations":    0,
        "total_explanations":   0,
        "rare_event_explains":  0,
        "total_steps":          0,
        "episodes":             0,
    }

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            metrics["total_steps"] += 1
            if info.get("safe_blocked"):
                metrics["shield_violations"] += 1
            if info.get("delta_exec") == 1:
                metrics["total_explanations"] += 1
                if info.get("action_label") in {"brake", "yield", "turn_left", "turn_right"}:
                    metrics["rare_event_explains"] += 1

        metrics["episodes"] += 1

    env.close()

    steps = max(1, metrics["total_steps"])
    metrics["explains_per_step"]    = round(metrics["total_explanations"] / steps, 4)
    metrics["rare_event_rate"]      = round(
        metrics["rare_event_explains"] / max(1, metrics["total_explanations"]), 4
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PPO timing policy.")
    p.add_argument("--episodes-dir", default="data/episodes",
                   help="Directory containing recorded episode logs.")
    p.add_argument("--model-dir", default="models/timing_policy",
                   help="Where to save model checkpoints and final weights.")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Total training timesteps (overrides config).")
    p.add_argument("--eval-only", default=None,
                   help="Path to a saved model to evaluate without training.")
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    cfg   = CONFIG
    eps_dir   = Path(args.episodes_dir)
    model_dir = Path(args.model_dir)
    timesteps = args.timesteps or cfg["rl"]["total_timesteps"]

    if args.eval_only:
        metrics = evaluate(Path(args.eval_only), eps_dir, cfg)
        print("\nEvaluation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    train(eps_dir, model_dir, cfg, timesteps)


if __name__ == "__main__":
    main()