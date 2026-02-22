"""
Derive discrete driving action labels from autopilot control signals.
Labels: cruise, accelerate, brake, turn_left, turn_right, yield
"""

CRUISE = "cruise"
ACCELERATE = "accelerate"
BRAKE = "brake"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
YIELD = "yield"

# Thresholds — chosen to match typical CARLA autopilot output ranges
_THROTTLE_ACTIVE = 0.15
_BRAKE_ACTIVE = 0.15
_STEER_ACTIVE = 0.10
_BRAKE_YIELD_THRESH = 0.60   # heavy braking at/near intersection = yield


def derive_action_label(
    throttle: float,
    brake: float,
    steer: float,
    crit_zone: int = 0,
) -> str:
    """
    Map raw autopilot controls to a discrete action label.
    Priority order: yield > brake > turn > accelerate > cruise.
    """
    if brake >= _BRAKE_ACTIVE and crit_zone:
        return YIELD
    if brake >= _BRAKE_ACTIVE:
        return BRAKE
    if steer > _STEER_ACTIVE:
        return TURN_RIGHT
    if steer < -_STEER_ACTIVE:
        return TURN_LEFT
    if throttle >= _THROTTLE_ACTIVE:
        return ACCELERATE
    return CRUISE


def enrich_action_labels(records: list[dict]) -> list[dict]:
    """In-place: add a_drive_t to each record."""
    for rec in records:
        rec["a_drive_t"] = derive_action_label(
            rec.get("throttle", 0.0),
            rec.get("brake", 0.0),
            rec.get("steer", 0.0),
            rec.get("crit_zone", 0),
        )
    return records