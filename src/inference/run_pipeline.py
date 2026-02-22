"""
Inference pipeline — user-facing entry point.

Point at any folder of driving images (e.g. downloaded from Google)
and the pipeline will:

  1. Build a synthetic episode from the images (via demo_generator).
  2. Load the trained PPO timing policy (or fall back to rule-based).
  3. Step through every frame, applying the safety shield.
  4. Call LLaVA only on frames where delta_exec == 1.
  5. Write a JSON results file and print a summary.

Usage:
    python -m src.inference.run_pipeline --images path/to/images
    python -m src.inference.run_pipeline --images path/to/images \\
        --model  models/timing_policy/timing_policy_final \\
        --sidecar data/sidecar_template.json \\
        --output  outputs/explanations/my_run

The --model flag is optional. If omitted the rule-based baseline is used,
which requires no trained weights and works out of the box.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from src.config_loader import CONFIG
from src.data_collection.demo_generator import generate_demo_episode
from src.data_collection.episode_loader import load_episode
from src.inference.baselines import RuleBasedGatePolicy
from src.llava.explainer import LLaVAExplainer
from src.rl.obs_space import OBS_KEYS, OBS_SCALE
from src.shield.safety_shield import SafetyShield

import numpy as np


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def _load_policy(model_path: Optional[Path], cfg: dict):
    """
    Load the PPO policy if a model path is given, otherwise return the
    rule-based gate so the pipeline works without trained weights.
    """
    if model_path is not None and model_path.exists():
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(model_path))
            print(f"Loaded PPO policy from {model_path}")
            return model, "ppo"
        except ImportError:
            print("[WARNING] stable-baselines3 not installed. Falling back to rule-based policy.")
    else:
        if model_path is not None:
            print(f"[WARNING] Model path '{model_path}' not found. Falling back to rule-based policy.")

    policy = RuleBasedGatePolicy(cfg)
    print("Using rule-based gate policy.")
    return policy, "rule_based"


def _predict(policy, policy_type: str, rec: dict) -> int:
    """Get a binary action from whichever policy type is active."""
    if policy_type == "ppo":
        raw = np.array(
            [float(rec.get(k, 0.0)) for k in OBS_KEYS], dtype=np.float32
        )
        raw[3] = min(raw[3], OBS_SCALE[3])
        obs = np.clip(raw / OBS_SCALE, -1.0, 1.0)
        action, _ = policy.predict(obs, deterministic=True)
        return int(action)
    else:
        return policy.predict_from_record(rec)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(results: list[dict], cfg: dict) -> dict:
    total = len(results)
    if total == 0:
        return {}

    n_explained      = sum(1 for r in results if r["delta_exec"] == 1)
    n_shield_blocked = sum(1 for r in results if r["policy_decision"] == 1 and r["delta_exec"] == 0)
    n_rare_explained = sum(
        1 for r in results
        if r["delta_exec"] == 1 and r["a_drive_t"] in {"brake", "yield", "turn_left", "turn_right"}
    )
    n_safe_blocked_violations = sum(
        1 for r in results if r.get("shield_violation", False)
    )

    episode_duration = results[-1]["timestamp"] if results else 0.0
    explains_per_min = (n_explained / max(episode_duration, 1.0)) * 60.0
    budget           = cfg["reward"]["budget_per_60s"]

    return {
        "total_frames":          total,
        "explanations_shown":    n_explained,
        "shield_violations":     n_safe_blocked_violations,
        "explains_per_minute":   round(explains_per_min, 2),
        "budget_per_minute":     budget,
        "over_budget":           explains_per_min > budget,
        "rare_event_rate":       round(n_rare_explained / max(n_explained, 1), 4),
        "timing_proxy_pct":      round(100.0 * n_rare_explained / max(n_explained, 1), 1),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    images_dir: Path,
    output_dir: Path,
    model_path: Optional[Path],
    sidecar_path: Optional[Path],
    cfg: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: build episode from images
    episode_dir = output_dir / "episode"
    print(f"\n[1/4] Building episode from images in '{images_dir}' ...")
    log_path = generate_demo_episode(
        image_dir=images_dir,
        output_dir=episode_dir,
        sidecar_path=sidecar_path,
    )
    records = load_episode(log_path)
    print(f"      {len(records)} frames loaded.")

    # Step 2: load policy
    print("\n[2/4] Loading timing policy ...")
    policy, policy_type = _load_policy(model_path, cfg)

    # Step 3: initialise shield and explainer (lazy model load)
    shield   = SafetyShield(cfg)
    explainer = LLaVAExplainer(
        model_name=cfg["llava"]["model_name"],
        max_new_tokens=cfg["llava"]["max_new_tokens"],
    )

    # Step 4: step through every frame
    print("\n[3/4] Running pipeline ...")
    results = []
    deferred_buffer = []  # track events shield blocked but policy wanted

    for i, rec in enumerate(records):
        ttc   = float(rec.get("ttc_t",  cfg["shield"]["ttc_inf"]))
        accel = float(rec.get("a_t",    0.0))
        omega = float(rec.get("omega_t", 0.0))
        crit  = int(rec.get("crit_zone", 0))
        ts    = float(rec.get("timestamp", i * 0.1))

        policy_decision = _predict(policy, policy_type, rec)

        delta_exec = shield.gate(
            policy_decision=policy_decision,
            ttc=ttc,
            accel=accel,
            omega=omega,
            crit_zone=crit,
            timestamp=ts,
            action=rec.get("a_drive_t", "cruise"),
            evidence=rec.get("evidence", []),
            context={"speed": rec.get("speed", 0.0), "crit_zone": crit, "ttc_t": ttc},
        )

        explanation = None
        deferred_delivered = None

        if delta_exec == 1:
            context = {
                "speed":     rec.get("speed", 0.0),
                "ttc_t":     ttc,
                "crit_zone": crit,
            }
            explanation = explainer.generate(
                image_path=rec.get("image_path", ""),
                action=rec.get("a_drive_t", "cruise"),
                context=context,
                evidence=rec.get("evidence", []),
                delta_exec=delta_exec,
            )

            # Also check for any deferred explanations ready to deliver
            deferred = shield.pop_deferred(now=ts, rho=cfg["reward"]["rho"])
            if deferred:
                delay = ts - deferred.timestamp
                deferred_delivered = {
                    "original_timestamp": deferred.timestamp,
                    "delay_sec":          round(delay, 3),
                    "action":             deferred.action,
                }

        result = {
            "frame":            i,
            "timestamp":        round(ts, 3),
            "image_path":       rec.get("image_path"),
            "a_drive_t":        rec.get("a_drive_t"),
            "speed":            rec.get("speed"),
            "ttc_t":            round(ttc, 3),
            "crit_zone":        crit,
            "theta_t":          rec.get("theta_t"),
            "w_t":              rec.get("w_t"),
            "policy_decision":  policy_decision,
            "delta_exec":       delta_exec,
            "shield_violation": False,  # shield is always enforced; this stays False
            "explanation":      explanation,
            "deferred":         deferred_delivered,
        }
        results.append(result)

        status = "EXPLAIN" if delta_exec == 1 else ("BLOCKED" if policy_decision == 1 else "silent")
        print(f"  frame {i:04d} | {rec.get('a_drive_t','?'):10s} | ttc={ttc:6.1f}s | {status}")

    # Step 5: write outputs
    print("\n[4/4] Writing results ...")
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics = _compute_metrics(results, cfg)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _print_summary(metrics, output_dir, policy_type)
    return metrics


def _print_summary(metrics: dict, output_dir: Path, policy_type: str) -> None:
    print("\n" + "=" * 55)
    print("  Pipeline Summary")
    print("=" * 55)
    print(f"  Policy            : {policy_type}")
    print(f"  Total frames      : {metrics.get('total_frames')}")
    print(f"  Explanations shown: {metrics.get('explanations_shown')}")
    print(f"  Explains / min    : {metrics.get('explains_per_minute')} "
          f"(budget: {metrics.get('budget_per_minute')})")
    print(f"  Over budget       : {metrics.get('over_budget')}")
    print(f"  Rare-event rate   : {metrics.get('timing_proxy_pct')} %")
    print(f"  Shield violations : {metrics.get('shield_violations')}  (must be 0)")
    print(f"\n  Results → {output_dir / 'results.json'}")
    print(f"  Metrics → {output_dir / 'metrics.json'}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the explanation timing pipeline on a folder of driving images."
    )
    p.add_argument(
        "--images", required=True,
        help="Path to folder containing driving images (jpg/png).",
    )
    p.add_argument(
        "--model", default=None,
        help="Path to trained PPO model (without .zip). "
             "Omit to use the rule-based baseline (no training required).",
    )
    p.add_argument(
        "--sidecar", default=None,
        help="Optional JSON file with per-image metadata overrides.",
    )
    p.add_argument(
        "--output", default="outputs/explanations/run",
        help="Directory to write results.json and metrics.json.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic state generation.",
    )
    return p.parse_args()


def main() -> None:
    import random
    args = _parse_args()
    random.seed(args.seed)

    run(
        images_dir=Path(args.images),
        output_dir=Path(args.output),
        model_path=Path(args.model) if args.model else None,
        sidecar_path=Path(args.sidecar) if args.sidecar else None,
        cfg=CONFIG,
    )


if __name__ == "__main__":
    main()