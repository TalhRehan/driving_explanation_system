"""
Time-To-Collision surrogate and synthetic feature proxies (theta_t, w_t).
"""

import math
from typing import Optional


def compute_ttc(
    distance: Optional[float],
    approaching_speed: Optional[float],
    ttc_inf: float = 999.0,
    eps: float = 1e-6,
) -> float:
    """
    TTC surrogate.  approaching_speed = max(0, dv_t) where dv_t > 0 means
    the gap is closing.  Returns ttc_inf when no forward actor is present.
    """
    if distance is None or approaching_speed is None:
        return ttc_inf
    closing = max(0.0, approaching_speed)
    return distance / max(eps, closing)


def compute_theta(
    accel: float,
    omega: float,
    crit_zone: int,
    accel_thresh: float = 3.0,
    omega_thresh: float = 0.25,
) -> float:
    """
    Uncertainty proxy: 1 for rare/important events, else 0.
    Rare events = hard brake/turn or proximity to a critical zone.
    """
    is_rare = (
        abs(accel) > accel_thresh
        or abs(omega) > omega_thresh
        or crit_zone == 1
    )
    return 1.0 if is_rare else 0.0


def compute_workload(ttc: float, crit_zone: int, ttc_inf: float = 999.0) -> float:
    """
    Workload proxy w_t: monotone function of criticality.
    Higher at intersections and when TTC is low.
    Normalised to [0, 1].
    """
    ttc_component = 1.0 - min(ttc / ttc_inf, 1.0)  # 1 when TTC→0, 0 when TTC→∞
    zone_component = float(crit_zone)
    return 0.6 * ttc_component + 0.4 * zone_component


def enrich_ttc_features(records: list[dict], cfg: dict) -> list[dict]:
    """
    In-place: add ttc_t, theta_t, w_t to each record.
    cfg is the shield/features sub-dict from settings.yaml.
    """
    ttc_inf = cfg["shield"]["ttc_inf"]
    eps = cfg["shield"]["eps"]
    accel_thresh = cfg["features"]["rare_event_accel_thresh"]
    omega_thresh = cfg["features"]["rare_event_omega_thresh"]

    for rec in records:
        rec["ttc_t"] = compute_ttc(
            rec.get("d_t"),
            rec.get("dv_t"),
            ttc_inf=ttc_inf,
            eps=eps,
        )
        rec["theta_t"] = compute_theta(
            rec.get("a_t", 0.0),
            rec.get("omega_t", 0.0),
            rec.get("crit_zone", 0),
            accel_thresh=accel_thresh,
            omega_thresh=omega_thresh,
        )
        rec["w_t"] = compute_workload(rec["ttc_t"], rec.get("crit_zone", 0), ttc_inf)

    return records