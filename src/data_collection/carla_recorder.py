"""
CARLA episode recorder.

Drives the ego vehicle using CARLA's built-in autopilot and records
one JSONL file per episode.  Each line is a single timestep containing
all fields required by the dataset schema.

Usage:
    python -m src.data_collection.carla_recorder --episodes 5 --duration 120
"""

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

# CARLA Python egg is installed separately; provide a clear error if missing.
try:
    import carla
except ImportError:
    sys.exit(
        "CARLA Python API not found.  Add the .egg from your CARLA release to "
        "PYTHONPATH or install it per the README."
    )

from src.config_loader import CONFIG
from src.data_collection.evidence import extract_gt_evidence
from src.features.action_labels import derive_action_label, enrich_action_labels
from src.features.kinematics import compute_acceleration, compute_yaw_rate
from src.features.ttc import compute_theta, compute_ttc, compute_workload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEATHER_PRESETS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
}


def _yaw_deg(transform: "carla.Transform") -> float:
    return transform.rotation.yaw


def _is_at_junction(waypoint) -> int:
    return 1 if waypoint.is_junction else 0


def _nearest_forward_actor(
    ego_tf: "carla.Transform",
    actors: list,
) -> tuple[float | None, float | None]:
    """
    Return (distance, relative_speed) to the nearest actor in front of ego.
    relative_speed > 0 means the gap is closing.
    Returns (None, None) when no forward actor is present.
    """
    ego_fwd = ego_tf.get_forward_vector()
    ego_loc = ego_tf.location
    best_d: float | None = None
    best_dv: float | None = None

    for actor in actors:
        if not hasattr(actor, "get_velocity"):
            continue
        loc = actor.get_transform().location
        diff = carla.Location(
            loc.x - ego_loc.x,
            loc.y - ego_loc.y,
            loc.z - ego_loc.z,
        )
        # Only consider actors ahead (positive dot product)
        dot = diff.x * ego_fwd.x + diff.y * ego_fwd.y
        if dot <= 0:
            continue
        d = math.sqrt(diff.x ** 2 + diff.y ** 2 + diff.z ** 2)
        v_ego = ego_tf.get_forward_vector()   # unit vector; speed handled below
        vel_actor = actor.get_velocity()
        # Closing speed along the forward axis
        dv = -( vel_actor.x * ego_fwd.x + vel_actor.y * ego_fwd.y )
        if best_d is None or d < best_d:
            best_d = d
            best_dv = dv

    return best_d, best_dv


# ---------------------------------------------------------------------------
# Camera attachment
# ---------------------------------------------------------------------------

class RGBCamera:
    """Lightweight wrapper around a CARLA RGB sensor."""

    def __init__(self, world: "carla.World", ego: "carla.Actor", image_dir: Path):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "640")
        bp.set_attribute("image_size_y", "480")
        bp.set_attribute("fov", "90")
        tf = carla.Transform(carla.Location(x=2.0, z=1.4))
        self._sensor = world.spawn_actor(bp, tf, attach_to=ego)
        self._image_dir = image_dir
        self._latest_path: str | None = None
        self._sensor.listen(self._on_image)

    def _on_image(self, image: "carla.Image") -> None:
        path = self._image_dir / f"rgb_{image.frame:08d}.png"
        image.save_to_disk(str(path))
        self._latest_path = str(path)

    def latest_path(self) -> str | None:
        return self._latest_path

    def destroy(self) -> None:
        self._sensor.destroy()


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ccfg = cfg["carla"]
        self.dcfg = cfg["data"]
        self.sample_dt = 1.0 / self.ccfg["sample_rate_hz"]

        self._client = carla.Client(self.ccfg["host"], self.ccfg["port"])
        self._client.set_timeout(self.ccfg["timeout"])

    # ------------------------------------------------------------------

    def record_episode(
        self,
        town: str,
        weather_name: str,
        duration_sec: int,
        episode_id: str | None = None,
    ) -> Path:
        """
        Record one episode and write a JSONL file + images folder.
        Returns the path to the JSONL file.
        """
        episode_id = episode_id or str(uuid.uuid4())[:8]
        world = self._client.load_world(town)
        world.set_weather(_WEATHER_PRESETS.get(weather_name, carla.WeatherParameters.ClearNoon))

        episode_dir = Path(self.dcfg["episodes_dir"]) / episode_id
        image_dir = episode_dir / self.dcfg["images_subdir"]
        image_dir.mkdir(parents=True, exist_ok=True)
        log_path = episode_dir / "log.jsonl"

        tm = self._client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.sample_dt
        world.apply_settings(settings)

        ego, camera = None, None
        try:
            ego = self._spawn_ego(world, tm)
            camera = RGBCamera(world, ego, image_dir)
            self._record_loop(
                world, ego, camera, log_path, episode_id, weather_name, duration_sec
            )
        finally:
            if camera:
                camera.destroy()
            if ego:
                ego.destroy()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        print(f"Episode {episode_id} saved → {log_path}")
        return log_path

    # ------------------------------------------------------------------

    def _spawn_ego(self, world, tm) -> "carla.Actor":
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        ego = world.try_spawn_actor(bp, spawn_points[0])
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        ego.set_autopilot(True, tm.get_port())
        return ego

    # ------------------------------------------------------------------

    def _record_loop(
        self,
        world: "carla.World",
        ego: "carla.Actor",
        camera: RGBCamera,
        log_path: Path,
        episode_id: str,
        weather_name: str,
        duration_sec: int,
    ) -> None:
        cfg = self.cfg
        ttc_inf = cfg["shield"]["ttc_inf"]
        eps = cfg["shield"]["eps"]
        top_k = cfg["data"]["top_k_boxes"]
        accel_thresh = cfg["features"]["rare_event_accel_thresh"]
        omega_thresh = cfg["features"]["rare_event_omega_thresh"]

        total_ticks = int(duration_sec * self.ccfg["sample_rate_hz"])
        prev_speed: float | None = None
        prev_yaw: float | None = None

        with open(log_path, "w") as fout:
            for tick in range(total_ticks):
                world.tick()
                ts = tick * self.sample_dt

                tf = ego.get_transform()
                vel = ego.get_velocity()
                ctrl = ego.get_control()

                speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                yaw = _yaw_deg(tf)

                accel = compute_acceleration(prev_speed, speed, self.sample_dt)
                omega = compute_yaw_rate(prev_yaw, yaw, self.sample_dt)

                prev_speed = speed
                prev_yaw = yaw

                waypoint = world.get_map().get_waypoint(tf.location)
                crit_zone = _is_at_junction(waypoint)

                actors = world.get_actors().filter("vehicle.*")
                peds = world.get_actors().filter("walker.*")
                all_nearby = list(actors) + list(peds)
                # Exclude ego itself
                all_nearby = [a for a in all_nearby if a.id != ego.id]

                d_t, dv_t = _nearest_forward_actor(tf, all_nearby)
                actor_density = len(all_nearby)

                ttc = compute_ttc(d_t, dv_t, ttc_inf=ttc_inf, eps=eps)
                theta = compute_theta(accel, omega, crit_zone, accel_thresh, omega_thresh)
                workload = compute_workload(ttc, crit_zone, ttc_inf)

                action = derive_action_label(
                    ctrl.throttle, ctrl.brake, ctrl.steer, crit_zone
                )

                evidence = extract_gt_evidence(world, ego, top_k=top_k)

                record = {
                    "episode_id": episode_id,
                    "timestamp": round(ts, 3),
                    "image_path": camera.latest_path(),
                    "speed": round(speed, 4),
                    "yaw_deg": round(yaw, 4),
                    "a_t": round(accel, 4),
                    "omega_t": round(omega, 6),
                    "throttle": round(ctrl.throttle, 4),
                    "brake": round(ctrl.brake, 4),
                    "steer": round(ctrl.steer, 4),
                    "d_t": round(d_t, 3) if d_t is not None else None,
                    "dv_t": round(dv_t, 3) if dv_t is not None else None,
                    "actor_density": actor_density,
                    "crit_zone": crit_zone,
                    "ttc_t": round(ttc, 3),
                    "theta_t": theta,
                    "w_t": round(workload, 4),
                    "a_drive_t": action,
                    "evidence": evidence,
                    "town": town if hasattr(self, '_current_town') else "",
                    "weather": weather_name,
                }

                fout.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record CARLA driving episodes.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--duration", type=int, default=120, help="Duration per episode in seconds.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = CONFIG
    towns = cfg["carla"]["towns"]
    weathers = cfg["carla"]["weather_presets"]
    recorder = EpisodeRecorder(cfg)

    for i in range(args.episodes):
        town = towns[i % len(towns)]
        weather = weathers[i % len(weathers)]
        print(f"Recording episode {i + 1}/{args.episodes} | {town} | {weather}")
        recorder.record_episode(town, weather, args.duration)


if __name__ == "__main__":
    main()