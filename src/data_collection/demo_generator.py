"""
Demo episode generator.

Builds a synthetic episode log from a folder of images (e.g. downloaded
from Google) so the full pipeline can be exercised without a running
CARLA instance.

State values are either loaded from an optional sidecar JSON file or
generated with configurable variation patterns that produce realistic
driving signal (speed changes, occasional hard brakes, intersection flags).

Usage:
    python -m src.data_collection.demo_generator \\
        --images path/to/my_images \\
        --output data/episodes/demo_01 \\
        --sidecar path/to/metadata.json   # optional

Sidecar JSON format (list, one entry per image, all fields optional):
    [
      {"speed": 12.5, "a_t": 0.1, "omega_t": 0.0, "crit_zone": 0,
       "d_t": 30.0, "dv_t": 2.0, "action": "cruise"},
      ...
    ]
Any missing field falls back to the config defaults or synthesised values.
"""

import argparse
import json
import math
import random
import uuid
from pathlib import Path

from src.config_loader import CONFIG
from src.features.action_labels import derive_action_label
from src.features.ttc import compute_theta, compute_ttc, compute_workload


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(image_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    return paths


def _load_sidecar(sidecar_path: Path | None, n: int) -> list[dict]:
    if sidecar_path is None or not sidecar_path.exists():
        return [{} for _ in range(n)]
    with open(sidecar_path) as f:
        data = json.load(f)
    # Pad or truncate to match image count
    if len(data) < n:
        data += [{}] * (n - len(data))
    return data[:n]


def _synthesise_state(
    i: int,
    n: int,
    prev: dict | None,
    override: dict,
    cfg: dict,
) -> dict:
    """
    Build a plausible state dict for timestep i.
    Values from `override` take precedence over synthesised ones.
    """
    fallback = cfg["inference"]
    sample_dt = 1.0 / cfg["carla"]["sample_rate_hz"]
    ttc_inf = cfg["shield"]["ttc_inf"]
    eps = cfg["shield"]["eps"]

    # Speed: smooth random walk clamped to [0, 20] m/s
    prev_speed = prev["speed"] if prev else fallback["fallback_speed"]
    speed = override.get("speed", max(0.0, min(20.0, prev_speed + random.uniform(-0.5, 0.5))))

    # Acceleration
    if prev:
        a_t = (speed - prev["speed"]) / sample_dt
    else:
        a_t = override.get("a_t", fallback["fallback_accel"])

    # Occasionally inject a hard brake or sharp turn for realism
    hard_event = (i % max(1, n // 5) == 0)
    if hard_event and "a_t" not in override:
        a_t = random.choice([-4.0, -3.8])
    if hard_event and "omega_t" not in override:
        omega_t = random.choice([-0.35, 0.35])
    else:
        omega_t = override.get("omega_t", fallback["fallback_omega"])

    # Intersection flag: flag every ~20 % of steps
    crit_zone = override.get("crit_zone", 1 if (i % 5 == 0) else 0)

    # Forward actor
    d_t = override.get("d_t", random.uniform(15.0, 60.0) if not hard_event else random.uniform(4.0, 10.0))
    dv_t = override.get("dv_t", random.uniform(-1.0, 3.0))

    # Derive throttle / brake / steer from a_t for action label derivation
    throttle = max(0.0, min(1.0, a_t / 5.0)) if a_t > 0 else 0.0
    brake = max(0.0, min(1.0, -a_t / 8.0)) if a_t < 0 else 0.0
    steer = max(-1.0, min(1.0, omega_t / 0.5))

    action = override.get("action") or derive_action_label(throttle, brake, steer, crit_zone)

    ttc = compute_ttc(d_t, dv_t, ttc_inf=ttc_inf, eps=eps)
    theta = compute_theta(
        a_t, omega_t, crit_zone,
        cfg["features"]["rare_event_accel_thresh"],
        cfg["features"]["rare_event_omega_thresh"],
    )
    w_t = compute_workload(ttc, crit_zone, ttc_inf)

    return {
        "speed": round(speed, 4),
        "a_t": round(a_t, 4),
        "omega_t": round(omega_t, 6),
        "throttle": round(throttle, 4),
        "brake": round(brake, 4),
        "steer": round(steer, 4),
        "d_t": round(d_t, 3),
        "dv_t": round(dv_t, 3),
        "actor_density": random.randint(0, 6),
        "crit_zone": crit_zone,
        "ttc_t": round(ttc, 3),
        "theta_t": theta,
        "w_t": round(w_t, 4),
        "a_drive_t": action,
    }


def generate_demo_episode(
    image_dir: Path,
    output_dir: Path,
    sidecar_path: Path | None = None,
    episode_id: str | None = None,
) -> Path:
    """
    Create a synthetic episode from images in image_dir.
    Writes log.jsonl to output_dir and symlinks (or copies) images.
    Returns the path to the JSONL log.
    """
    cfg = CONFIG
    episode_id = episode_id or str(uuid.uuid4())[:8]
    images = _collect_images(image_dir)
    overrides = _load_sidecar(sidecar_path, len(images))

    output_dir = Path(output_dir)
    img_out = output_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.jsonl"

    sample_dt = 1.0 / cfg["carla"]["sample_rate_hz"]
    prev_rec: dict | None = None

    with open(log_path, "w") as fout:
        for i, (img_path, override) in enumerate(zip(images, overrides)):
            # Copy image into episode folder so the episode is self-contained
            dest = img_out / img_path.name
            if not dest.exists():
                import shutil
                shutil.copy2(img_path, dest)

            state = _synthesise_state(i, len(images), prev_rec, override, cfg)
            record = {
                "episode_id": episode_id,
                "timestamp": round(i * sample_dt, 3),
                "image_path": str(dest),
                **state,
                "evidence": [],   # No CARLA GT; YOLO evidence added during inference
                "town": "demo",
                "weather": "unknown",
            }
            fout.write(json.dumps(record) + "\n")
            prev_rec = state

    print(f"Demo episode '{episode_id}' created → {log_path}  ({len(images)} steps)")
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a demo episode from user images.")
    p.add_argument("--images", required=True, help="Directory containing driving images.")
    p.add_argument("--output", default="data/episodes/demo", help="Output episode directory.")
    p.add_argument("--sidecar", default=None, help="Optional JSON file with per-image metadata overrides.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    generate_demo_episode(
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        sidecar_path=Path(args.sidecar) if args.sidecar else None,
    )


if __name__ == "__main__":
    main()