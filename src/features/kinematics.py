"""
Compute ego-vehicle kinematic state features from raw episode records.
All functions operate on plain dicts (one timestep) or lists thereof.
"""

from typing import Optional


def compute_acceleration(
    prev_speed: Optional[float],
    curr_speed: float,
    dt: float,
) -> float:
    """Finite-difference acceleration in m/s^2."""
    if prev_speed is None or dt <= 0:
        return 0.0
    return (curr_speed - prev_speed) / dt


def compute_yaw_rate(
    prev_yaw_deg: Optional[float],
    curr_yaw_deg: float,
    dt: float,
) -> float:
    """Yaw rate in rad/s from consecutive yaw angles (degrees)."""
    if prev_yaw_deg is None or dt <= 0:
        return 0.0
    import math
    delta = curr_yaw_deg - prev_yaw_deg
    # Wrap to [-180, 180]
    delta = (delta + 180.0) % 360.0 - 180.0
    return math.radians(delta) / dt


def enrich_kinematics(records: list[dict], sample_rate_hz: float) -> list[dict]:
    """
    In-place: add acceleration and yaw_rate to each record using
    finite differences from the previous timestep.
    """
    dt = 1.0 / sample_rate_hz
    for i, rec in enumerate(records):
        prev = records[i - 1] if i > 0 else None
        rec["a_t"] = compute_acceleration(
            prev["speed"] if prev else None,
            rec["speed"],
            dt,
        )
        rec["omega_t"] = compute_yaw_rate(
            prev["yaw_deg"] if prev else None,
            rec["yaw_deg"],
            dt,
        )
    return records