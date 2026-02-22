# Timing-Controlled Driving Explanations

A system that generates natural-language explanations of autonomous driving
decisions, controlled by a learned timing policy (PPO) and a hard safety
shield. Explanations are produced by LLaVA and are shown **only** when the
timing policy requests it **and** the safety shield permits it.

---

## Project Structure

```
driving_explanation_system/
├── config/
│   └── settings.yaml          ← all thresholds, hyperparameters, paths
├── data/
│   ├── episodes/              ← recorded or generated episode logs
│   └── sidecar_template.json  ← metadata template for user-provided images
├── models/
│   └── timing_policy/         ← PPO checkpoints saved here after training
├── outputs/
│   └── explanations/          ← pipeline results written here
├── src/
│   ├── config_loader.py
│   ├── data_collection/
│   │   ├── carla_recorder.py  ← records CARLA episodes to JSONL
│   │   ├── demo_generator.py  ← builds episodes from user-provided images
│   │   ├── episode_loader.py  ← reads JSONL logs for the RL environment
│   │   └── evidence.py        ← CARLA ground-truth bounding box extraction
│   ├── features/
│   │   ├── kinematics.py      ← acceleration, yaw-rate computation
│   │   ├── ttc.py             ← TTC surrogate, theta_t, w_t proxies
│   │   └── action_labels.py   ← throttle/brake/steer → discrete action label
│   ├── shield/
│   │   └── safety_shield.py   ← SafeBlock logic + deferred event buffer
│   ├── rl/
│   │   ├── obs_space.py       ← observation vector constants
│   │   ├── environment.py     ← Gymnasium env replaying logged episodes
│   │   ├── reward.py          ← four-component reward function
│   │   └── ppo_trainer.py     ← PPO training + evaluation entry point
│   ├── llava/
│   │   └── explainer.py       ← LLaVA-1.5 wrapper (lazy load)
│   └── inference/
│       ├── baselines.py       ← AlwaysExplain / NeverExplain / RuleBasedGate
│       └── run_pipeline.py    ← user-facing entry point
└── requirements.txt
```

---

## Quick Start — Test With Your Own Images

No CARLA installation or trained model required for this path.

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

> CARLA's Python API must be installed separately (see CARLA section below).
> For image-only testing you can skip that step.

**2. Drop your images into a folder**

```
my_images/
    frame_001.jpg
    frame_002.jpg
    ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
Images are sorted alphabetically, so name them in driving order.

**3. Run the pipeline**

```bash
python -m src.inference.run_pipeline --images my_images
```

Results are written to `outputs/explanations/run/`:

| File           | Contents                                                         |
| -------------- | ---------------------------------------------------------------- |
| `results.json` | Per-frame: action, delta_exec, explanation text, deferred events |
| `metrics.json` | Summary: shield violations, explains/min, rare-event rate        |

**4. Optional — provide metadata via sidecar**

If you know the driving context for your images (speed, whether the car is
braking, etc.), fill in `data/sidecar_template.json` and pass it:

```bash
python -m src.inference.run_pipeline \
    --images   my_images \
    --sidecar  data/sidecar_template.json \
    --output   outputs/explanations/my_run
```

Any field you omit is synthesised automatically.

---

## Full Pipeline — With CARLA

### 1. Install CARLA

Download CARLA 0.9.15 from https://carla.org and follow the official
installation guide. Then add the Python API egg to your environment:

```bash
# Example — adjust path to match your CARLA installation
export PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg
```

### 2. Record Episodes

Start the CARLA server, then:

```bash
python -m src.data_collection.carla_recorder --episodes 10 --duration 120
```

Each episode is saved under `data/episodes/<episode_id>/` as `log.jsonl`
plus an `images/` folder of RGB frames.

Episode count, duration, towns, and weather presets are all controlled from
`config/settings.yaml`.

### 3. Train the Timing Policy

```bash
python -m src.rl.ppo_trainer
```

Optional flags:

```bash
python -m src.rl.ppo_trainer \
    --episodes-dir data/episodes \
    --model-dir    models/timing_policy \
    --timesteps    500000
```

Checkpoints are saved every 10 % of training. The best model (by eval
reward) is saved to `models/timing_policy/best/`.

### 4. Evaluate the Trained Policy

```bash
python -m src.rl.ppo_trainer \
    --eval-only models/timing_policy/timing_policy_final
```

Prints all spec metrics: shield violations, explains/min, rare-event rate.

### 5. Run With the Trained Model

```bash
python -m src.inference.run_pipeline \
    --images my_images \
    --model  models/timing_policy/timing_policy_final
```

If `--model` is omitted the rule-based baseline is used automatically.

---

## Configuration

All tunable parameters live in `config/settings.yaml`.
The most commonly adjusted values:

| Key                     | Default                    | Description                                |
| ----------------------- | -------------------------- | ------------------------------------------ |
| `shield.tau_ttc`        | `3.0`                      | Block explanations if TTC < this (seconds) |
| `shield.tau_a`          | `3.5`                      | Block on hard braking/acceleration (m/s²)  |
| `shield.tau_omega`      | `0.3`                      | Block on sharp steering (rad/s)            |
| `reward.budget_per_60s` | `4`                        | Max explanations per 60-second window      |
| `reward.alpha`          | `0.3`                      | Workload cost weight                       |
| `reward.gamma_freq`     | `0.2`                      | Frequency penalty weight                   |
| `rl.total_timesteps`    | `500000`                   | PPO training steps                         |
| `llava.model_name`      | `llava-hf/llava-1.5-7b-hf` | HuggingFace model ID                       |
| `llava.max_new_tokens`  | `256`                      | Max tokens per explanation                 |

---

## Baselines

Three baselines are available for comparison against the PPO policy:

| Baseline        | Class                 | Behaviour                               |
| --------------- | --------------------- | --------------------------------------- |
| Always explain  | `AlwaysExplainPolicy` | `delta_t = 1` every step                |
| Never explain   | `NeverExplainPolicy`  | `delta_t = 0` every step                |
| Rule-based gate | `RuleBasedGatePolicy` | Explain if Benefit − α·WorkloadCost ≥ λ |

All three pass through the same safety shield, so shield violations remain
zero regardless of which policy is active.

To run with the rule-based baseline explicitly:

```bash
python -m src.inference.run_pipeline --images my_images
# (omitting --model automatically selects rule-based)
```

---

## Output Format

### `results.json`

List of records, one per frame:

```json
{
  "frame": 7,
  "timestamp": 0.7,
  "image_path": "...",
  "a_drive_t": "brake",
  "speed": 8.2,
  "ttc_t": 4.1,
  "crit_zone": 0,
  "theta_t": 1.0,
  "w_t": 0.43,
  "policy_decision": 1,
  "delta_exec": 1,
  "shield_violation": false,
  "explanation": "The vehicle braked because a pedestrian ...",
  "deferred": null
}
```

`delta_exec = 1` means an explanation was shown.
`shield_violation` will always be `false` — the shield is a hard constraint.

### `metrics.json`

```json
{
  "total_frames": 120,
  "explanations_shown": 6,
  "shield_violations": 0,
  "explains_per_minute": 3.0,
  "budget_per_minute": 4,
  "over_budget": false,
  "rare_event_rate": 0.8333,
  "timing_proxy_pct": 83.3
}
```

---

## Safety Guarantee

The safety shield is a **hard constraint** applied after every policy
decision. An explanation is blocked (`delta_exec = 0`) if **any** of the
following hold at that timestep:

- TTC < `tau_ttc` (imminent collision risk)
- |acceleration| > `tau_a` (hard braking or acceleration)
- |yaw rate| > `tau_omega` (sharp steering)
- Vehicle is at an intersection, merge, or crosswalk (`crit_zone = 1`)

Blocked events are optionally buffered and delivered retrospectively once
the shield clears, penalised by a staleness term in the reward.

`shield_violations` in `metrics.json` must always be `0`. If it is not,
the shield thresholds in `config/settings.yaml` need to be tightened.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for Python packages
- CARLA 0.9.15 (only required for data collection, not for image-based inference)
- ~14 GB disk space for LLaVA-1.5-7b weights (downloaded automatically on first run)
- GPU recommended for LLaVA inference; CPU works but is slow
