"""
Utilities for loading recorded episodes from disk.
Used by the RL environment to replay logged state sequences.
"""

import json
from pathlib import Path
from typing import Iterator


def iter_records(log_path: Path) -> Iterator[dict]:
    """Yield one record dict per line from a JSONL episode log."""
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_episode(log_path: Path) -> list[dict]:
    """Load all records from a JSONL episode log into a list."""
    return list(iter_records(log_path))


def discover_episodes(episodes_dir: Path) -> list[Path]:
    """
    Return sorted list of log.jsonl paths found under episodes_dir.
    Each episode lives in its own sub-directory.
    """
    return sorted(episodes_dir.glob("*/log.jsonl"))


def episode_summary(records: list[dict]) -> dict:
    """Return high-level stats for a loaded episode (useful for sanity checks)."""
    if not records:
        return {}
    durations = records[-1]["timestamp"] - records[0]["timestamp"]
    n_blocked = sum(1 for r in records if r.get("crit_zone", 0))
    rare = sum(1 for r in records if r.get("theta_t", 0) > 0)
    return {
        "episode_id": records[0].get("episode_id"),
        "n_steps": len(records),
        "duration_sec": round(durations, 2),
        "crit_zone_steps": n_blocked,
        "rare_event_steps": rare,
        "towns": records[0].get("town"),
        "weather": records[0].get("weather"),
    }