"""
Evidence extraction: ground-truth 2D bounding boxes from CARLA (Option 1).

Projects world-space actor bounding boxes onto the ego front camera plane
and returns the top-K visible actors sorted by distance.
"""

import math
from typing import Any

try:
    import carla
except ImportError:
    carla = None  # Allows module import outside CARLA environment


# Camera intrinsics matching the sensor spawned in carla_recorder.py
_IMG_W = 640
_IMG_H = 480
_FOV_DEG = 90.0


def _build_projection_matrix(w: int, h: int, fov_deg: float):
    import numpy as np
    fov_rad = math.radians(fov_deg)
    f = w / (2.0 * math.tan(fov_rad / 2.0))
    return np.array([
        [f,  0, w / 2],
        [0,  f, h / 2],
        [0,  0,     1],
    ])


def _world_to_camera(world_pt, cam_tf):
    """Transform a world-space carla.Location into camera space (numpy array)."""
    import numpy as np
    loc = world_pt
    ct = cam_tf

    # World → camera coordinate transform
    forward = ct.get_forward_vector()
    right   = ct.get_right_vector()
    up      = ct.get_up_vector()

    dx = loc.x - ct.location.x
    dy = loc.y - ct.location.y
    dz = loc.z - ct.location.z

    x =  dx * right.x   + dy * right.y   + dz * right.z
    y = -(dx * up.x      + dy * up.y      + dz * up.z)
    z =  dx * forward.x + dy * forward.y + dz * forward.z

    return np.array([x, y, z])


def _project_point(cam_pt, K):
    """Project a camera-space 3D point to pixel coordinates."""
    if cam_pt[2] <= 0:
        return None
    px = int(K[0, 0] * cam_pt[0] / cam_pt[2] + K[0, 2])
    py = int(K[1, 1] * cam_pt[1] / cam_pt[2] + K[1, 2])
    return px, py


def _actor_distance(ego_loc, actor) -> float:
    loc = actor.get_transform().location
    return math.sqrt(
        (loc.x - ego_loc.x) ** 2 +
        (loc.y - ego_loc.y) ** 2 +
        (loc.z - ego_loc.z) ** 2
    )


def _actor_class(actor) -> str:
    type_id = actor.type_id
    if "walker" in type_id:
        return "pedestrian"
    if "vehicle" in type_id:
        return "vehicle"
    return "unknown"


def extract_gt_evidence(
    world: Any,
    ego: Any,
    top_k: int = 5,
    max_distance: float = 80.0,
) -> list[dict]:
    """
    Return a list of up to top_k dicts, each describing one nearby actor
    visible in the ego front camera, using CARLA ground-truth geometry.

    Each dict:
        class       : "vehicle" | "pedestrian"
        distance    : float (metres)
        bbox        : [x1, y1, x2, y2] in pixel coords, or None if off-screen
    """
    if carla is None:
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    ego_tf = ego.get_transform()
    # Camera is mounted 2 m forward, 1.4 m up
    cam_loc = carla.Location(
        x=ego_tf.location.x + 2.0 * math.cos(math.radians(ego_tf.rotation.yaw)),
        y=ego_tf.location.y + 2.0 * math.sin(math.radians(ego_tf.rotation.yaw)),
        z=ego_tf.location.z + 1.4,
    )
    cam_tf = carla.Transform(cam_loc, ego_tf.rotation)
    K = _build_projection_matrix(_IMG_W, _IMG_H, _FOV_DEG)

    actors = list(world.get_actors().filter("vehicle.*")) + \
             list(world.get_actors().filter("walker.*"))

    candidates = []
    for actor in actors:
        if actor.id == ego.id:
            continue
        dist = _actor_distance(ego_tf.location, actor)
        if dist > max_distance:
            continue

        bb = actor.bounding_box
        corners_local = [
            carla.Location( bb.extent.x,  bb.extent.y, 0),
            carla.Location(-bb.extent.x,  bb.extent.y, 0),
            carla.Location( bb.extent.x, -bb.extent.y, 0),
            carla.Location(-bb.extent.x, -bb.extent.y, 0),
        ]
        actor_tf = actor.get_transform()
        pixels = []
        for corner in corners_local:
            world_loc = actor_tf.transform(corner)
            cam_pt = _world_to_camera(world_loc, cam_tf)
            px = _project_point(cam_pt, K)
            if px is not None:
                pixels.append(px)

        if not pixels:
            continue

        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Discard completely off-screen boxes
        if x2 < 0 or x1 > _IMG_W or y2 < 0 or y1 > _IMG_H:
            continue

        candidates.append({
            "class": _actor_class(actor),
            "distance": round(dist, 2),
            "bbox": [
                max(0, x1), max(0, y1),
                min(_IMG_W, x2), min(_IMG_H, y2),
            ],
        })

    candidates.sort(key=lambda c: c["distance"])
    return candidates[:top_k]