"""
Shared observation space constants for the timing policy.
Kept in a standalone module so feature computation and evaluation
utilities can import them without requiring gymnasium.
"""

import numpy as np

# Feature names in observation vector order
OBS_KEYS = ["speed", "a_t", "omega_t", "ttc_t", "crit_zone", "theta_t", "w_t"]

# Per-feature normalisation bounds — raw values are divided by these
# before being clipped to [-1, 1] and fed to the policy network.
OBS_SCALE = np.array([
    30.0,   # speed    (m/s)
    10.0,   # a_t      (m/s^2)
    1.0,    # omega_t  (rad/s)
    60.0,   # ttc_t    (s, clipped before dividing)
    1.0,    # crit_zone
    1.0,    # theta_t
    1.0,    # w_t
], dtype=np.float32)