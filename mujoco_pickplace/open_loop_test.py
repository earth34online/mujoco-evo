"""Strict, teacher-forced action evaluation of native MuJoCo episode datasets.

Run ``python mujoco_pickplace/open_loop_test.py --help`` from any directory.
Reports use a fresh timestamped project output directory by default. Physics replay and
real-feedback probes are opt-in; no training cache is built. Expert annotations
are never model inputs.
An open-loop error is not a closed-loop task success rate.
"""

from __future__ import annotations

import argparse
import os
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import secrets
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "Evo-1/Evo_1"
DEFAULT_BACKBONE = ROOT / "ckpt/evo1_mujoco_pickplace_stage3/step_final"
ARCHIVED_DIAGNOSTIC_BACKBONE = ROOT / "ckpt/archive/evo1_mujoco_pickplace_stage3_stepfinal80000_native_feedback_v1"
HUMAN_PHASES = {"observe_target", "presentation_close"}
VARIANTS = ("full", "current_only", "current_images", "reverse_history_images")


@dataclass(frozen=True)
class TaskContract:
    number: int
    key: str
    instruction: str
    action_mask: tuple


# Public task specifications, independent of the runtime registry and its masks.
TASK_CONTRACTS = {
    1: TaskContract(1, "mujoco_pickplace", "pick up the blue cube and place it on the green target", (1,1,1,0,0,0,1)),
    3: TaskContract(3, "mujoco_chopstick", "pick up the chopstick and place it inside the storage box", (1,1,1,0,0,0,1)),
    4: TaskContract(4, "mujoco_drawer_memory", "take the object you just saw out of its drawer and place it on the green receiving area", (1,1,1,1,1,1,1)),
}


def validate_declared_quality(quality, task):
    """Check annotations independently; their physical truth remains unverified."""
    require(isinstance(quality, dict), "Missing quality annotations")
    if task.number == 1:
        require(quality.get("randomize_task") is True and quality.get("randomization_scale") == 1.,
                "Task1 requires full task randomization")
        require(quality.get("quality_schema_version") == 3, "Unsupported Task1 quality evidence")
        require(quality.get("first_attachment_has_two_pad_contact") is True
                and quality.get("attachment_lost_before_release") is False
                and quality.get("two_pad_contact_steps", 0) >= 2,
                "Task1 declared contact/retention failed")
        limits = {"successful_close_xy_error": (0., .006), "successful_close_z_above": (-.012, .001),
                  "attachment_xy_error": (0., .006), "attachment_z_above": (-.012, .001),
                  "pregrasp_cube_displacement": (0., .004), "pregrasp_cube_tilt_deg": (0., 3.),
                  "attachment_cube_tilt_deg": (0., 3.), "held_cube_tilt_deg": (0., 8.),
                  "attachment_cube_angular_speed": (0., .15)}
        for name, (low, high) in limits.items():
            value = float(quality.get(name, float("nan")))
            require(np.isfinite(value) and low - 1e-8 <= value <= high + 1e-8,
                    f"Task1 invalid declared grasp measurement: {name}={value}")
    else:
        require(quality.get("schema") == "mujoco-panda-contact" and quality.get("schema_version") == 3,
                "Unsupported contact evidence schema")
        require(quality.get("task_id") == task.number and quality.get("success") is True
                and quality.get("transport_drop") is False and quality.get("motion_failures") == []
                and quality.get("failed_checks") == [] and quality.get("timestamps_preserve_control_time") is True,
                "Declared trajectory contains a failure")
        required = {"complete_motion", "two_finger_grasp", "object_lifted", "transport_retained",
                    "no_robot_table_contact", "no_unexpected_robot_contact", "no_execution_external_force",
                    "no_object_gravity_assistance", "robot_actuators_only", "stable_placement",
                    "confirmed_release_and_support", "continuous_contact_monitoring",
                    "physics_substep_placement_monitoring", "contact_gap_descent_monitoring"}
        if task.number == 4:
            required |= {"horizontal_handle_approach", "handle_contact_through_pull", "no_wrong_drawer_opened",
                         "correct_first_drawer", "presentation_complete", "presentation_robot_clear", "selection_within_history"}
            require(bool(quality.get("target_visible_history_slots_at_selection")), "No declared visible target history")
        if task.number == 4 and (quality.get("correction_source") or {}).get("kind") == "object_empty_grasp":
            from mujoco_pickplace.episode_dataset import validate_correction_context
            validate_correction_context(None, quality, task.number)
            history_checks = {"horizontal_handle_approach", "handle_contact_through_pull"}
            recorded = quality.get("policy_context_checks", {})
            require(set(recorded) == history_checks and all(type(v) is bool for v in recorded.values()),
                    "Missing excluded policy handle-stage diagnostics")
            required -= history_checks
        checks = quality.get("checks", {})
        require(required <= checks.keys() and all(value is True for value in checks.values()),
                "Missing/failed declared trajectory checks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invalid_json_number(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid_json_number)


def read_jsonl(path):
    return [json.loads(line, parse_constant=invalid_json_number) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def safe_child(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            f"Expected dataset-relative path: {relative!r}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Dataset path escapes root: {relative}")
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty file: {path}")
    return path


def file_stamp(path):
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def trajectory_signature(states, actions, times):
    digest = hashlib.sha256()
    for value in (states, actions, np.asarray(times) - times[0]):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def history_indices(times, index, frames, stride_seconds):
    """Causal, float32-tolerant simulation-time sampling, including left padding."""
    times = np.asarray(times, dtype=np.float64)
    require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0), "History timestamps must strictly increase")
    require(0 <= index < len(times) and frames >= 1, "Invalid history index/window")
    require(np.isfinite(stride_seconds) and stride_seconds > 0, "Invalid history stride")
    tolerance = max(1e-8, 2 * np.finfo(np.float32).eps * max(1., abs(times[index])))
    deadlines = times[index] - np.arange(frames - 1, -1, -1) * stride_seconds
    raw = np.searchsorted(times[:index + 1], deadlines + tolerance, side="right") - 1
    return np.maximum(raw, 0), (raw >= 0)


def window_indices(phases, actions, times, stride, selection_time=None):
    """Include phase and grip transitions even when regular sampling is sparse."""
    eligible = np.array([p not in HUMAN_PHASES and p != "policy_context" for p in phases])
    require(eligible.any(), "Episode contains no policy execution frames")
    candidates = set(np.flatnonzero(eligible)[::stride].tolist())
    boundaries = set(np.flatnonzero(np.r_[True, phases[1:] != phases[:-1]]).tolist())
    boundaries.update(np.flatnonzero(np.r_[False, np.abs(np.diff(actions[:, 6])) > 0.02]).tolist())
    boundaries.update([int(np.flatnonzero(eligible)[0]), len(phases) - 1])
    if selection_time is not None:
        boundaries.add(int(np.argmin(np.abs(times - float(selection_time)))))
    for boundary in boundaries:
        candidates.update(i for i in (boundary - 1, boundary, boundary + 1)
                          if 0 <= i < len(phases) and eligible[i])
    return sorted(candidates)


@dataclass
class Episode:
    root: Path
    row: dict
    states: np.ndarray
    actions: np.ndarray
    times: np.ndarray
    phases: np.ndarray
    videos: dict
    hashes: dict
    signature: str


def load_episode(root, row, metadata, task):
    import pandas as pd

    path = safe_child(root, row["data_path"])
    stamp = file_stamp(path)
    frame = pd.read_parquet(path)
    required = {"episode_index", "frame_index", "timestamp", "video_timestamp", "seed",
                "task_index", "observation.state", "action", "expert.phase", "next.done"}
    require(required <= set(frame.columns), f"Missing columns in {path}: {required - set(frame.columns)}")
    n = len(frame)
    require(n > 0 and n == row["length"], f"Episode length mismatch: {path}")
    require(row.get("success") is True and row.get("task") == task.instruction,
            f"Episode is not an accepted demonstration of {task.key}")
    for field in ("episode_index", "seed", "task_index"):
        require(np.all(frame[field].to_numpy() == row[field]), f"Mixed/mismatched {field}: {path}")
    require(np.array_equal(frame.frame_index.to_numpy(), np.arange(n)), f"Broken frame indices: {path}")
    states = np.asarray(np.stack(frame["observation.state"]), dtype=np.float64)
    actions = np.asarray(np.stack(frame["action"]), dtype=np.float64)
    require(states.shape == (n, 8) and actions.shape == (n, 7), f"Invalid feature shape: {path}")
    require(np.isfinite(states).all() and np.isfinite(actions).all(), f"NaN/Inf features: {path}")
    require(np.all((actions[:, 6] >= 0) & (actions[:, 6] <= 1)), f"Invalid expert gripper command: {path}")
    inactive = ~np.asarray(task.action_mask, dtype=bool)
    require(np.all(np.abs(actions[:, inactive]) <= 1e-7), f"Nonzero inactive action labels: {path}")
    times = frame.timestamp.to_numpy(dtype=np.float64)
    period = float(metadata["control_period_seconds"])
    tolerance = max(1e-6, 4 * np.finfo(np.float32).eps * max(1., np.max(np.abs(times))))
    require(np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0),
            f"Invalid simulation timestamps: {path}")
    require(np.allclose(np.diff(times), period, rtol=0, atol=tolerance),
            f"Compacted, missing, or mistimed control frames: {path}")
    require(np.allclose(frame.video_timestamp, np.arange(n) / metadata["fps"], rtol=0, atol=1e-6),
            f"Video/simulation timestamp confusion: {path}")
    dones = frame["next.done"].to_numpy()
    require(all(isinstance(x, (bool, np.bool_)) for x in dones)
            and bool(dones[-1]) and not np.any(dones[:-1]), f"Invalid terminal flags: {path}")
    phases = frame["expert.phase"].to_numpy()
    require(all(isinstance(p, str) and p.strip() for p in phases), f"Missing expert phases: {path}")
    context = np.flatnonzero(phases == "policy_context")
    if len(context) or row["quality"].get("correction_source") is not None:
        from mujoco_pickplace.episode_dataset import validate_correction_context
        # The data contract distinguishes empty-grasp recovery (Task3) from
        # an early handle-approach intervention after presentation (Task4).
        # Physical reproduction is checked separately during replay below.
        validate_correction_context(phases, row["quality"], task.number, times)
        evidence = row["quality"].get("correction_source", {})
        if task.number == 3:
            failure_time = evidence.get("recorded_empty_grasp_sim_time", float('nan'))
            require(np.any(np.isclose(times[context], failure_time, atol=1e-4, rtol=0)),
                    f"Recorded empty grasp is absent from policy prefix: {path}")
    validate_declared_quality(row["quality"], task)
    require(set(row["video_paths"]) == {"front"}, f"Unsupported/missing physical cameras: {path}")
    videos = {name: safe_child(root, value) for name, value in row["video_paths"].items()}
    hashes = {"parquet_sha256": sha256(path)}
    require(file_stamp(path) == stamp, f"Episode changed during read: {path}")
    # Detect re-indexed copies without depending on parquet encoding or episode IDs.
    return Episode(root, row, states, actions, times, phases, videos, hashes,
                   trajectory_signature(states, actions, times))


def decode_video(path, expected_frames, fps, shape, wanted):
    """Decode the entire stream, validate count/PTS, retain only requested frames."""
    import av
    stamp = file_stamp(path)
    result = {}
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        require(stream.average_rate is not None and abs(float(stream.average_rate) - fps) < 1e-4,
                f"Video FPS mismatch: {path}")
        count, origin = 0, None
        for index, frame in enumerate(container.decode(stream)):
            require(frame.pts is not None, f"Video frame without PTS: {path}")
            pts = float(frame.pts * stream.time_base)
            if origin is None:
                origin = pts
            require(abs((pts - origin) - index / fps) < max(1e-5, 0.02 / fps),
                    f"Missing/irregular video frame at {index}: {path}")
            require([frame.height, frame.width, 3] == shape, f"Video resolution mismatch: {path}")
            if index in wanted:
                image = frame.to_image().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                result[index] = base64.b64encode(buffer.getvalue()).decode("ascii")
            count += 1
    require(count == expected_frames and set(result) == set(wanted),
            f"Truncated/extra video frames: {path}; expected {expected_frames}, decoded {count}")
    digest = sha256(path)
    require(file_stamp(path) == stamp, f"Video changed during read: {path}")
    return result, digest


def payload_for(episode, index, task, images, frames, stride, variant, seed):
    indices, valid = history_indices(episode.times, index, frames, stride)
    image_indices = indices.copy()
    states = episode.states[indices].copy()
    if variant == "current_only":
        valid[:-1] = False
        image_indices[:] = index
        states[:] = episode.states[index]
    elif variant == "current_images":
        # Keep proprioception and validity identical: only remove visual history.
        image_indices[:] = index
    elif variant == "reverse_history_images":
        past = np.flatnonzero(valid[:-1])
        image_indices[past] = image_indices[past[::-1]]
    else:
        require(variant == "full", f"Unknown history variant: {variant}")
    return {
        "memory_images": [[images[int(i)]] for i in image_indices],
        "image_encoding": "png_base64", "history_mask": valid.tolist(),
        "state": states.tolist(), "image_mask": [1],
        "action_mask": list(task.action_mask) + [0] * 17,
        "prompt": task.instruction, "task_key": task.key, "flow_seed": seed,
    }, indices, valid


def corrupt_masked_history(payload):
    """Adversarial canary: masked observations must have zero effect on output."""
    import copy
    from PIL import Image
    changed = copy.deepcopy(payload)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 255)).save(buffer, format="PNG")
    pixels = base64.b64encode(buffer.getvalue()).decode("ascii")
    for index, valid in enumerate(changed["history_mask"]):
        if not valid:
            changed["memory_images"][index] = [pixels] * len(changed["image_mask"])
            changed["state"][index] = [1e3, -1e3, 1e3, -1e3, 1e3, -1e3, 1e3, -1e3]
    return changed


def verify_mask_invariance(predictor, payload, reference, atol=1e-5):
    if all(payload["history_mask"]):
        return {"status": "not_applicable_no_masked_slots"}
    changed = np.asarray(predictor(corrupt_masked_history(payload)), float)
    require(changed.shape == reference.shape and np.isfinite(changed).all(), "Invalid masked-history canary output")
    delta = float(np.max(np.abs(changed-reference)))
    require(delta <= atol, f"Masked history changes model output (max delta {delta}); history mask is ineffective")
    return {"status": "passed", "max_abs_action_delta": delta}


def action_metrics(prediction, target, active, low, high, *, continuous_gripper=False):
    """Raw physical-unit errors; never clip predictions or score terminal padding."""
    pred, truth = np.asarray(prediction, float), np.asarray(target, float)
    require(pred.shape == truth.shape and pred.ndim == 2 and pred.shape[1] == 7
            and len(pred) > 0, "Prediction/target shape mismatch")
    require(np.isfinite(pred).all() and np.isfinite(truth).all(), "Non-finite prediction/target")
    active = np.asarray(active, bool)
    error = pred - truth
    xyz = np.linalg.norm(error[:, :3], axis=1) * 1000
    widths = np.asarray(high)[:7] - np.asarray(low)[:7]
    scaled = active & (widths > 1e-8)
    moving = np.linalg.norm(truth[:, :3], axis=1) > 1e-4
    pred_moving = np.linalg.norm(pred[:, :3], axis=1) > 1e-4
    signed = moving & pred_moving
    cosine = (np.sum(pred[signed, :3] * truth[signed, :3], axis=1)
              / (np.linalg.norm(pred[signed, :3], axis=1) * np.linalg.norm(truth[signed, :3], axis=1)))
    # Task4's 0.25 aperture is a position target, not a closed-hand label.
    # Keep binary hysteresis semantics for Tasks 1/3 and count only actual
    # endpoint targets for the continuous gripper. Its MAE still scores every
    # requested aperture, including handle approach.
    close = truth[:, 6] <= (1e-6 if continuous_gripper else .4)
    opened = truth[:, 6] >= (1. - 1e-6 if continuous_gripper else .6)
    holding = ~moving
    return {
        "xyz_error_mm_mean": float(xyz.mean()), "xyz_error_mm_max": float(xyz.max()),
        "first_xyz_error_mm": float(xyz[0]),
        "axis_bias_mm": (error[:, :3].mean(axis=0) * 1000).tolist(),
        "axis_mae_mm": (np.abs(error[:, :3]).mean(axis=0) * 1000).tolist(),
        "delta_sum_error_mm": float(np.linalg.norm(error[:, :3].sum(axis=0)) * 1000),
        "rotation_delta_error_rad": (float(np.linalg.norm(error[:, 3:6], axis=1).mean())
                                     if active[3:6].any() else None),
        "gripper_mae": float(np.abs(error[:, 6]).mean()),
        "gripper_out_of_range_count": int(np.sum((pred[:, 6] < 0) | (pred[:, 6] > 1))),
        "reference_close_count": int(close.sum()), "reference_open_count": int(opened.sum()),
        "false_open_count": int(np.sum(close & (pred[:, 6] >= .6))),
        "false_close_count": int(np.sum(opened & (pred[:, 6] <= .4))),
        "ambiguous_on_endpoint_count": int(np.sum((close | opened) & (pred[:, 6] > .4) & (pred[:, 6] < .6))),
        "moving_reference_count": int(moving.sum()),
        "stalled_count": int(np.sum(moving & ~pred_moving)),
        "opposite_direction_count": int(np.sum(cosine < 0)),
        "unclipped_normalized_rmse": (float(np.sqrt(np.mean((2 * error[:, scaled] / widths[scaled]) ** 2)))
                                      if scaled.any() else None),
        "zero_span_active_dims": np.flatnonzero(active & ~scaled).tolist(),
        "valid_steps_count": len(truth),
        "reference_hold_count": int(holding.sum()),
        "hold_translation_drift_mm_max": (float(np.linalg.norm(pred[holding, :3], axis=1).max() * 1000)
                                           if holding.any() else None),
        "hold_vertical_bias_mm": float(pred[holding, 2].mean() * 1000) if holding.any() else None,
    }


def distribution(values):
    values = np.asarray(values, float)
    if not values.size:
        return None
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


class Scores:
    """Separate tasks, variants, episodes, current phases and future target phases."""
    def __init__(self):
        self.groups = defaultdict(lambda: defaultdict(list))
        self.counts = defaultdict(Counter)

    def add(self, key, metrics):
        for name, value in metrics.items():
            if name.endswith("_count"):
                self.counts[key][name] += value
            elif isinstance(value, (float, int)):
                self.groups[key][name].append(value)
            elif name in {"axis_bias_mm", "axis_mae_mm"}:
                for axis, number in zip("xyz", value):
                    self.groups[key][f"{name}_{axis}"].append(number)

    def report(self):
        return {key: {"metrics": {name: distribution(values) for name, values in self.groups[key].items()},
                      "counts": dict(self.counts[key])} for key in sorted(self.groups)}

    def macro_report(self):
        """Equal-weight episode/phase means, independent of their trajectory lengths."""
        groups = defaultdict(lambda: defaultdict(list))
        for key, metrics in self.groups.items():
            for marker in ("/episode/", "/current_phase/"):
                if marker in key:
                    prefix = key.split(marker)[0] + ("/episode_macro" if marker == "/episode/" else "/phase_macro")
                    for name, values in metrics.items():
                        groups[prefix][name].append(float(np.mean(values)))
        return {key: {name: distribution(values) for name, values in metrics.items()}
                for key, metrics in sorted(groups.items())}


def validate_checkpoint(path, tasks):
    config = read_json(path / "config.json")
    stats = read_json(path / "norm_stats.json")
    require((path / "mp_rank_00_model_states.pt").is_file()
            and (path / "mp_rank_00_model_states.pt").stat().st_size > 0, "Missing/empty model weights")
    require(config.get("use_state", True), "Checkpoint must use robot proprioception")
    require(config.get("per_action_dim", 24) == 24 and config.get("state_dim", 24) == 24,
            "Unsupported checkpoint dimensions")
    for task in tasks:
        require(task.key in stats, f"Checkpoint has no normalization key {task.key}")
        for feature in ("action", "observation.state"):
            entry = stats[task.key][feature]
            low, high = np.asarray(entry["min"], float), np.asarray(entry["max"], float)
            require(low.shape == high.shape == (24,) and np.isfinite(low).all()
                    and np.isfinite(high).all() and np.all(low <= high),
                    f"Invalid checkpoint bounds: {task.key}/{feature}")
    frames = int(config.get("memory_frames", 1))
    horizon = int(config.get("action_horizon", config.get("horizon", 14)))
    seconds = config.get("memory_stride_seconds")
    # Match the public evaluation client's five-Hz reference-step fallback.
    stride = float(seconds) if seconds is not None else float(config.get("memory_stride_steps", 5)) * .2
    require(frames > 0 and horizon > 0 and np.isfinite(stride) and stride > 0, "Invalid checkpoint window")
    return config, stats, frames, horizon, stride


def recorded_drawer_target(row, task_number):
    if task_number != 4:
        return None
    target = row.get("quality", {}).get("target_drawer_diagnostic_only")
    require(type(target) is int and target in range(4), "Task4 episode lacks a valid presentation target")
    return target


def dataset_manifest(root, task, expected=None):
    metadata = read_json(root / "meta/dataset.json")
    # Counts can be internally consistent BETWEEN writes while collection is
    # still active. Independently reject unfinished snapshots before auditing.
    collection_report = metadata.get('last_collection_report')
    if collection_report is not None:
        require(isinstance(collection_report, dict), f"Invalid collection report: {root}")
        attempts = collection_report.get('attempt_results', [])
        require(isinstance(attempts, list) and all(isinstance(row, dict) for row in attempts),
                f"Invalid collection attempts: {root}")
        require(collection_report.get('status') != 'running'
                and not any(row.get('status') == 'running' for row in attempts),
                f"Unfinished collection snapshot: {root}")
        require(collection_report.get('status') in {'completed', 'attempt_limit', 'interrupted', 'error'},
                f"Unknown collection snapshot status: {root}")
    rows = read_jsonl(root / "meta/episodes.jsonl")
    require(metadata.get("format") == "mujoco-evo-episodes" and metadata.get("format_version") == "1.0",
            f"Unsupported dataset: {root}")
    require(metadata.get("task") == task.instruction and rows, f"Wrong/empty task dataset: {root}")
    require(metadata.get("total_episodes") == len(rows)
            and metadata.get("total_frames") == sum(r["length"] for r in rows),
            f"Manifest incomplete or collection still running: {root}")
    for name in ("episode_index", "data_path"):
        require(len({r[name] for r in rows}) == len(rows), f"Duplicate {name}: {root}")
    presentations = {(row["seed"], recorded_drawer_target(row, task.number)) for row in rows}
    require(len(presentations) == len(rows), f"Duplicate scene/presentation: {root}")
    require(metadata.get("cameras", {}).keys() == {"front"}, f"Unsupported cameras: {root}")
    fps, period = float(metadata["fps"]), float(metadata["control_period_seconds"])
    require(np.isfinite(fps) and fps > 0 and np.isfinite(period) and period > 0
            and abs(fps * period - 1) < 1e-6, f"FPS/control period mismatch: {root}")
    camera = metadata["cameras"]["front"]
    require(camera.get("fps") == fps and len(camera.get("shape", [])) == 3
            and camera["shape"][2] == 3 and all(type(x) is int and x > 0 for x in camera["shape"]),
            f"Invalid camera dimensions/FPS: {root}")
    collection = metadata.get("collection_config", {})
    if "control_period_seconds" in collection:
        require(abs(collection["control_period_seconds"] - period) < 1e-6,
                f"Collection control period mismatch: {root}")
    require(not collection.get("compact_static_frames", False), f"Compacted temporal data: {root}")
    if expected:
        version = expected.get("required_source_policy_version")
        if version:
            require(metadata.get("source_policy_version") == version, f"Source policy mismatch: {root}")
        for key, value in expected.get("required_collection_config", {}).items():
            require(collection.get(key) == value, f"Collection contract mismatch {key}: {root}")
    if task.number in (3, 4):
        require(metadata.get("source_policy_version") == f"task{task.number}-contact-v3",
                f"Unsupported contact source policy: {root}")
    mapping = read_jsonl(root / "meta/tasks.jsonl")
    require(len(mapping) == 1 and mapping[0] == {"task_index": 0, "task": task.instruction},
            f"Instruction mapping mismatch: {root}")
    return metadata, rows


def memory_selection_evidence(episode, frames, stride):
    """Check certified visibility against actual sampled history; diagnostics only."""
    quality = episode.row["quality"]
    if quality.get("drawer_selection_observation_sim_time") is None:
        return None
    selection = float(quality["drawer_selection_observation_sim_time"])
    index = int(np.argmin(np.abs(episode.times - selection)))
    require(abs(episode.times[index] - selection) < 1e-4, "Drawer selection timestamp absent from trajectory")
    indices, valid = history_indices(episode.times, index, frames, stride)
    certified_times = quality["sampled_history_sim_times_at_selection"]
    slots = quality["target_visible_history_slots_at_selection"]
    require(all(isinstance(s, int) and 0 <= s < len(certified_times) for s in slots),
            "Invalid certified visibility slots")
    visible = []
    for slot, (i, usable) in enumerate(zip(indices, valid)):
        if usable and (episode.phases[i] == "observe_target" or
                       any(abs(episode.times[i] - certified_times[s]) < 1e-4 for s in slots)):
            visible.append(slot)
    return {"frame": index, "history_indices": indices.tolist(), "certified_visible_slots": visible,
            "memory_target_available": bool(visible),
            "meaning": "Visibility certified by collection metadata; not inferred from pixels"}


def compare_closed_loop(report, paths):
    """Descriptive paired evidence, never an invented open-loop success formula."""
    mapping = {1: "mujoco_pickplace", 3: "mujoco_chopstick", 4: "mujoco_drawer_memory"}
    identity = report["checkpoint"]
    rows, seen = [], set()
    for path in paths:
        content = read_json(path)
        source_rows = content if isinstance(content, list) else content["episodes"]
        for row in source_rows:
            number = int(row["task"])
            target = row.get("target_drawer_annotation_only") if number == 4 else None
            key = (number, int(row["seed"]), target)
            require(key not in seen, f"Duplicate closed-loop scene/presentation: {key}")
            seen.add(key)
            require(type(row.get("success")) is bool, "Closed-loop success must be boolean")
            metadata = row.get("checkpoint_metadata", {})
            other = metadata.get("checkpoint_identity") or {}
            for field in ("weights_sha256", "config_sha256", "norm_stats_sha256"):
                require(identity.get(field) and other.get(field) == identity[field],
                        f"Closed-loop {field} missing or differs; cannot pair {path}")
            require(metadata.get("memory_frames") == report["memory_frames"], "Closed-loop K differs")
            seconds = metadata.get("memory_stride_seconds")
            if seconds is None:
                seconds = float(metadata.get("memory_stride_steps", 5)) * .2
            require(abs(float(seconds) - report["memory_stride_seconds"]) < 1e-8,
                    "Closed-loop memory stride differs")
            require(row.get("environment_compatibility", {}).get("status") != "mismatch",
                    "Closed-loop environment mismatch; cannot pair this run")
            rows.append(row)
    result = {"status": "descriptive_only_not_success_rate_calibration", "tasks": {},
              "limitations": [
                  "Both dataset and closed-loop environments must be verified against the same checkpoint; this still does not establish identical step budgets or stochastic actions.",
                  "Only episodes with exhaustive open-loop execution-frame coverage are paired.",
                  "Expert-state error cannot observe policy-induced drift, contact loss or recovery.",
                  "No threshold is fitted on these outcomes and no open-loop predicted success rate is reported.",
              ]}
    for number, key in mapping.items():
        selected = {(e["seed"], e.get("target_drawer_annotation_only")): e
                    for d in report["datasets"] if d["task"] == key for e in d["episodes"]}
        pairs = []
        unmatched = []
        unverified_environment = []
        for row in rows:
            if row["task"] != number:
                continue
            if (report.get("environment_compatibility", {}).get(key, {}).get("status") != "verified"
                    or row.get("environment_compatibility", {}).get("status") != "verified"):
                unverified_environment.append(row["seed"])
                continue
            target = row.get("target_drawer_annotation_only") if number == 4 else None
            episode = selected.get((row["seed"], target))
            group = (f"{key}/full/episode/{episode['episode_index']}" if episode else "")
            if (not episode or episode["selected_windows"] != episode["candidate_windows"]
                    or report["arguments"]["window_stride"] != 1 or group not in report["scores"]):
                unmatched.append(row["seed"])
                continue
            error = report["scores"][group]["metrics"]["first_xyz_error_mm"]["mean"]
            pairs.append({"seed": row["seed"], "success": row["success"],
                          "mean_first_xyz_error_mm": error,
                          "transport_drop": row.get("transport_drop"),
                          "correct_first_drawer": row.get("correct_first_drawer"),
                          "termination_status": row.get("termination_status")})
        if not pairs and not unmatched and not unverified_environment:
            continue
        n = len(pairs)
        rate = sum(p["success"] for p in pairs) / n if n else None
        interval = None
        if n:
            z = 1.95996398454
            center = (rate + z*z/(2*n)) / (1+z*z/n)
            width = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
            interval = [max(0., float(center-width)), min(1., float(center+width))]
        success = [p["mean_first_xyz_error_mm"] for p in pairs if p["success"]]
        failed = [p["mean_first_xyz_error_mm"] for p in pairs if not p["success"]]
        auc = (float(np.mean([float(f > s) + .5*float(f == s) for f in failed for s in success]))
               if success and failed else None)
        result["tasks"][key] = {"pairs": pairs, "unmatched_or_incomplete_seeds": unmatched,
                                 "unverified_environment_seeds": unverified_environment,
                                 "matched_count": n, "observed_closed_loop_success_rate": rate,
                                 "success_rate_wilson_95_interval": interval,
                                 "success_episode_error_mm": distribution(success),
                                 "failed_episode_error_mm": distribution(failed),
                                 "failure_ranking_auc_from_xyz_error": auc}
    return result


class LocalPredictor:
    def __init__(self, path, vlm_name):
        import torch
        sys.path.insert(0, str(EVO))
        from scripts.Evo1 import EVO1
        from model.action_head.observation_decoder import attach_observation_decoder, backbone_environment_contracts
        require(torch.cuda.is_available(), "CUDA is required for this model's inference")
        config_bytes = (path / "config.json").read_bytes()
        stats_bytes = (path / "norm_stats.json").read_bytes()
        config = json.loads(config_bytes, parse_constant=invalid_json_number)
        self.stats = json.loads(stats_bytes, parse_constant=invalid_json_number)
        legacy = bool(config.get("legacy_inference_contract", "memory_frames" not in config))
        config["legacy_inference_contract"] = legacy
        config.setdefault("num_inference_timesteps", 32 if legacy else 50)
        config["device"] = "cuda"
        if vlm_name:
            config["vlm_name"] = vlm_name
        name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        if Path(name).is_absolute() and not Path(name).exists():
            require(name.replace("\\", "/").rstrip("/").endswith("/OpenGVLab/InternVL3-1B"),
                    "Missing VLM directory: pass --vlm-name")
            config["vlm_name"] = "OpenGVLab/InternVL3-1B"
        weight_path = path / "mp_rank_00_model_states.pt"
        before = file_stamp(weight_path)
        print("Loading checkpoint with independent open-loop loader...", flush=True)
        self.model = EVO1(config).eval()
        checkpoint = torch.load(weight_path, map_location="cpu", mmap=True, weights_only=False)
        progress_state = checkpoint.get("client_state") or checkpoint
        recorded_config = progress_state.get("config") or {}
        require(recorded_config.get("task_environment_contracts") == backbone_environment_contracts(config),
                "Environment provenance differs between config.json and checkpoint client_state")
        require(recorded_config.get("training_dataset_snapshots") == config.get("training_dataset_snapshots"),
                "Dataset snapshots differ between config.json and checkpoint client_state")
        require(recorded_config.get("checkpoint_fp32_temporal_parameter_names") == config.get("checkpoint_fp32_temporal_parameter_names"),
                "Parameter storage precision differs between config.json and checkpoint client_state")
        self.model.load_state_dict(checkpoint["module"], strict=True)
        digest = sha256(weight_path)
        require(file_stamp(weight_path) == before, "Checkpoint changed while loading")
        # Keep adapters unmerged: test the checkpoint's actual parameterized function.
        require(config_bytes == (path / "config.json").read_bytes()
                and stats_bytes == (path / "norm_stats.json").read_bytes(), "Checkpoint config/statistics changed while loading")
        self.identity = {"path": str(path.resolve()),
                         "step": progress_state.get("step", checkpoint.get("global_steps")),
                         "weights_sha256": digest, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                         "norm_stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
                         "lora_merged": False, "num_inference_timesteps": config["num_inference_timesteps"]}
        # Record the implementation that actually ran: optional FlashAttention
        # can fall back at import time, changing BF16 features across processes.
        vision_layers = self.model.embedder.model.vision_model.encoder.layers
        self.identity["attention_backend"] = {
            "vision": "flash_attention_2" if all(
                bool(getattr(layer.attn, "use_flash_attn", False)) for layer in vision_layers
            ) else "standard",
            "language": str(getattr(
                self.model.embedder.model.language_model.config, "_attn_implementation", "unknown"
            )),
        }
        attach_observation_decoder(self.model, path, config, self.stats, self.identity)
        self.model.checkpoint_identity = self.identity

    def check_rgb_preprocessing(self):
        from PIL import Image
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        size = int(self.model.config.get("image_size", 448))
        actual = self.model.embedder._preprocess_images([[Image.new("RGB", (size, size), c) for c in colors]])[0]
        actual = actual.float().cpu().numpy()
        expected = (np.eye(3) - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        require(actual.shape == (3, 3, size, size), "Vision input shape changed")
        delta = float(np.max(np.abs(actual - expected[:, :, None, None])))
        require(delta < .02, f"RGB ordering/scaling/ImageNet normalization incorrect: {delta}")
        return {"status": "passed", "max_abs_normalized_pixel_error": delta, "tolerance": .02}

    def check_serving_request(self, payload, reference):
        """The production request handler is the subject, not the reference oracle."""
        import torch
        from scripts.Evo1_server import Normalizer, infer_from_json_dict
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            actual = np.asarray(infer_from_json_dict(payload, self.model, Normalizer(self.stats), True), float)
        require(actual.shape == reference.shape and np.isfinite(actual).all(), "Serving handler returned invalid actions")
        delta = float(np.max(np.abs(actual-reference)))
        return {"status": "passed" if delta <= 1e-5 else "failed", "max_abs_action_delta": delta,
                "scope": "request preprocessing, normalization and inference on identical loaded weights",
                "separate_checkpoint_loader_and_lora_merge_verified": False}

    def __call__(self, payload):
        import torch
        from PIL import Image
        config = self.model.config
        observation_decoder = getattr(self.model, "observation_action_decoder", None)
        if observation_decoder is not None:
            observation_decoder.validate_request(payload.get("task_key"), payload.get("prompt"), payload.get("action_mask"))
        grid = []
        size = int(config.get("image_size", 448))
        for frame in payload["memory_images"]:
            images = []
            for encoded in frame:
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    images.append(image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
            grid.append(images)
        view_mask = list(payload["image_mask"])
        if self.model.legacy_inference_contract:
            while len(view_mask) < config.get("max_views", 3):
                for frame in grid:
                    frame.append(Image.new("RGB", (size, size)))
                view_mask.append(0)
        normalized = normalize_proprioception(payload["state"], self.stats[payload["task_key"]]["observation.state"])
        # Preserve caller RNGs and use common random noise for paired comparisons.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.inference_mode():
            torch.manual_seed(payload["flow_seed"])
            torch.cuda.manual_seed_all(payload["flow_seed"])
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                action = self.model.run_inference(
                    images=grid, image_mask=torch.tensor(view_mask, dtype=torch.bool, device="cuda"),
                    prompt=payload["prompt"], state_input=torch.tensor(normalized, dtype=torch.float32, device="cuda"),
                    action_mask=torch.tensor([payload["action_mask"]], dtype=torch.int32, device="cuda"),
                    history_mask=torch.tensor(payload["history_mask"], dtype=torch.bool, device="cuda"))
            values = action.detach().float().cpu().numpy().reshape(-1, 24)
        if observation_decoder is not None:
            self.last_diagnostics = dict(observation_decoder.last_diagnostics or {})
        return denormalize_actions(values, self.stats[payload["task_key"]]["action"])


def normalize_proprioception(states, stats):
    states = np.asarray(states, dtype=np.float32)
    require(states.ndim == 2 and states.shape[1] == 8 and np.isfinite(states).all(), "Invalid proprioception")
    low, high = (np.asarray(stats[n], dtype=np.float32)[:8] for n in ("min", "max"))
    result = np.zeros((len(states), 24), dtype=np.float32)
    result[:, :8] = np.clip(2 * (states-low) / (high-low+np.float32(1e-8)) - 1, -1, 1)
    return result


def denormalize_actions(actions, stats):
    actions = np.asarray(actions, dtype=np.float32)
    low, high = (np.asarray(stats[n], dtype=np.float32) for n in ("min", "max"))
    require(actions.ndim == 2 and actions.shape[1] == 24 and np.isfinite(actions).all(), "Invalid normalized action")
    return ((actions+1) * np.float32(.5) * (high-low+np.float32(1e-8)) + low).astype(np.float64)


def balanced_budgets(capacities, budget):
    """Round-robin allocation prevents early tasks/episodes consuming every query."""
    if budget == 0:
        return list(capacities)
    result = [0] * len(capacities)
    left = min(budget, sum(capacities))
    while left:
        for i, capacity in enumerate(capacities):
            if result[i] < capacity and left:
                result[i] += 1
                left -= 1
    return result


def stratified_windows(candidates, phases, actions, budget, selection_index=None):
    """Preserve event coverage, then spread remaining samples over each phase."""
    if budget >= len(candidates):
        return list(candidates)
    if budget <= 0:
        return []
    available = set(candidates)
    selected = []
    def add(i):
        if i in available and i not in selected and len(selected) < budget:
            selected.append(i)
    if selection_index is not None:
        add(selection_index)
    critical = ("close", "object_close", "handle_close", "descend", "object_descend",
                "pull", "topdown_orientation", "object_clearance_lift", "object_lift",
                "lift", "transport", "release", "retreat", "settle")
    groups = defaultdict(list)
    for i in candidates:
        groups[str(phases[i])].append(i)
    for name in critical:
        if name in groups:
            add(groups[name][0])
    transitions = np.flatnonzero(np.r_[False, np.abs(np.diff(actions[:, 6])) > .02])
    for i in transitions:
        add(int(i))
    for indices in groups.values():
        add(indices[len(indices)//2])
    add(candidates[0]); add(candidates[-1])
    # Space-filling selection within each phase, not chronological prefixes.
    for fraction in (.0, 1., .25, .75):
        for indices in groups.values():
            add(indices[round(fraction*(len(indices)-1))])
    if len(selected) < budget:
        remaining = [i for i in candidates if i not in selected]
        for k in np.linspace(0, len(remaining)-1, budget-len(selected)).round().astype(int):
            add(remaining[k])
    return sorted(selected)


def transition_alignment(states, actions, phases, control_period):
    """Detect a better lag than zero; tracking dynamics preclude exact equality.

    This is a diagnostic of translations on robot-owned steps, not proof of
    image/action synchronization. Rotation and gripper contact need replay.
    """
    delta = np.diff(np.asarray(states, float)[:, :3], axis=0)
    actions = np.asarray(actions, float)[:, :3]
    keep = np.array([p not in {"observe_target", "presentation_close"} for p in phases[:-1]])
    indices = np.arange(2, len(delta)-2)
    if len(indices) < 10:
        return {"status": "insufficient_motion_evidence"}
    keep_indices = indices[keep[indices] & (np.linalg.norm(delta[indices], axis=1) > .0002)]
    if len(keep_indices) < 10:
        return {"status": "insufficient_motion_evidence"}
    errors = {}
    for lag in (-2, -1, 0, 1, 2):
        errors[lag] = float(np.median(np.linalg.norm(delta[keep_indices] - actions[keep_indices+lag], axis=1))*1000)
    best = min(errors, key=errors.get)
    improvement = errors[0] - errors[best]
    suspicious = best != 0 and improvement > .5 and errors[best] < .65*errors[0]
    return {"status": "suspected_temporal_misalignment" if suspicious else "no_translation_lag_detected",
            "samples": len(keep_indices), "median_tracking_error_mm_by_label_offset": {str(k):v for k,v in errors.items()},
            "best_label_offset_steps": best, "best_label_offset_seconds": best*control_period,
            "improvement_mm": improvement, "image_alignment_verified": False,
            "note": "Lag can also reflect controller delay; flagged data require replay/instrumented evidence."}


def evaluate_limits(scores, limits):
    """Support counts, rates, signed bounds and absolute bounds; missing is not pass."""
    if not isinstance(limits, list) or not limits:
        raise ValueError("Limits must be a nonempty list")
    results = []
    for rule in limits:
        if not any(k in rule for k in ("minimum", "maximum", "absolute_maximum")):
            raise ValueError("Each limit needs a minimum, maximum or absolute_maximum")
        entry = scores.get(rule["group"])
        metric = rule["metric"]
        kind = rule.get("kind", "metric")
        if kind not in {"metric", "count", "rate"}:
            raise ValueError(f"Unknown limit kind: {kind}")
        value = None
        if entry is not None:
            if kind == "metric":
                value = entry["metrics"].get(metric, {}).get(rule.get("statistic", "mean"))
            elif kind == "count":
                value = entry["counts"].get(metric)
            elif kind == "rate":
                numerator = entry["counts"].get(metric)
                denominator = entry["counts"].get(rule["denominator"])
                if numerator is not None and denominator is not None and denominator > 0:
                    value = numerator/denominator
            else:
                raise ValueError(f"Unknown limit kind: {kind}")
        passed = value is not None and np.isfinite(value)
        for name in ("minimum", "maximum", "absolute_maximum"):
            if name not in rule:
                continue
            bound = float(rule[name])
            if not np.isfinite(bound) or (name == "absolute_maximum" and bound < 0):
                raise ValueError("Invalid acceptance bound")
            if passed:
                passed = (value >= bound if name == "minimum" else
                          abs(value) <= bound if name == "absolute_maximum" else value <= bound)
        results.append({**rule, "actual": value, "passed": bool(passed),
                        "status": "missing_evidence" if value is None else "passed" if passed else "failed"})
    return results


def acceptance_evidence_gaps(report):
    """A numerical limit passing is not sufficient when its context is unknown."""
    gaps = []
    if not report.get("coverage_complete"):
        gaps.append("incomplete_window_coverage")
    if report.get("missing_certified_memory"):
        gaps.append("missing_target_history")
    if report.get("generalization") != "selected_episodes_disjoint_from_declared_training_sources":
        gaps.append("heldout_sources_not_verified")
    compatibility = report.get("environment_compatibility", {})
    tasks = [dataset["task"] for dataset in report.get("datasets", [])]
    if not tasks or any(compatibility.get(task, {}).get("status") != "verified" for task in tasks):
        gaps.append("checkpoint_dataset_environment_not_verified")
    return gaps


def wilson_interval(successes, total):
    if not total:
        return None
    z = 1.95996398454
    rate = successes/total
    center = (rate+z*z/(2*total))/(1+z*z/total)
    width = z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0.,float(center-width)), min(1.,float(center+width))]


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def geom_vertices(model, data, geom):
    import mujoco
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]) * model.geom_size[geom]
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[start:start+count]
    else:
        raise ValueError("Independent placement audit supports box/mesh objects only")
    return local @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]


class PhysicsObserver:
    def __init__(self, env, task):
        self.env, self.task = env, task
        model, data = env.model, env.data
        self.object = model.geom("cube_geom" if task == 1 else "object_geom").id
        self.surface = model.geom("goal_geom" if task == 1 else "bin_bottom" if task == 3 else "receiving_area").id
        self.fingers = {model.body(n).id for n in ("left_finger", "right_finger")}
        joint = model.joint("cube_free" if task == 1 else "object_free")
        self.velocity_start = int(joint.dofadr[0])
        self.start_z = float(data.geom_xpos[self.object, 2])
        self.lifted = False
        self.release_committed = False
        self.retention_armed = False
        self.free_hold_since = None
        self.two_finger_seen = False
        self.drop_seen = False
        self.gap_since = None
        self.gap_z = None
        self.gap_descent = 0.0
        self.gap_release_allowed = False
        self.gap_closed_hold = False
        self.gap_hold_relative = None
        self.last_loaded_time = None
        self.previous_gripper_open = float(env.gripper) > .5
        # Resolve raw geometry independently, without environment hold flags.
        import mujoco
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.stable_since = None
        self.first_drawer_contact = None
        self.wrong_drawer_opened = False
        self.expected_drawer = None
        if task == 4:
            bottom = geom_vertices(model, data, self.object)[:,2].min()
            support_tops = [data.geom_xpos[model.geom(f"drawer_insert{k}").id,2]
                            + model.geom(f"drawer_insert{k}").size[2] for k in range(4)]
            self.expected_drawer = int(np.argmin(np.abs(np.asarray(support_tops)-bottom)))

    @contextmanager
    def monitor_substeps(self):
        """Observe raw physics after each existing Task3/4 retention callback.

        The callback is only a clock hook: none of its classifications are read.
        Restore the exact instance state even when stepping or sampling raises.
        """
        env = self.env
        original = env._update_object_retention
        had_override = "_update_object_retention" in env.__dict__
        def observe():
            result = original()
            self.sample()
            return result
        env._update_object_retention = observe
        try:
            yield self
        finally:
            if had_override:
                env._update_object_retention = original
            else:
                del env._update_object_retention

    def sample(self):
        """Classify observed raw states; standalone calls cannot cover skipped ticks.

        Time-based evidence makes repeated boundary samples idempotent. The
        suite also installs monitor_substeps for Task3/4 so an opening followed
        by detachment inside one control interval remains observable.
        """
        import mujoco
        model, data = self.env.model, self.env.data
        now = float(data.time)
        vertices = geom_vertices(model, data, self.object)
        rotation = data.geom_xmat[self.surface].reshape(3,3)
        local = (vertices-data.geom_xpos[self.surface]) @ rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        size = model.geom_size[self.surface]
        limits_low, limits_high = -size[:2].copy(), size[:2].copy()
        if self.task == 3:
            # Derive the receptacle's inner walls from collision geometry.
            for geom in range(model.ngeom):
                if geom == self.surface or model.geom_bodyid[geom] != model.geom_bodyid[self.surface]:
                    continue
                if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                    continue
                wall = (geom_vertices(model,data,geom)-data.geom_xpos[self.surface]) @ rotation
                lo, hi = wall.min(axis=0), wall.max(axis=0)
                center, half = (lo+hi)/2, (hi-lo)/2
                for axis in (0,1):
                    if abs(center[axis]) > size[axis]*.5 and half[axis] < .02:
                        if center[axis] > 0:
                            limits_high[axis] = min(limits_high[axis],lo[axis])
                        else:
                            limits_low[axis] = max(limits_low[axis],hi[axis])
        if self.task == 1:
            # Preserve the legacy center-in-circle semantics as a separate task contract.
            center = (data.geom_xpos[self.object]-data.geom_xpos[self.surface]) @ rotation
            contained = np.linalg.norm(center[:2]) <= size[0]
        else:
            contained = bool(np.all(lower[:2] >= limits_low-1e-6) and np.all(upper[:2] <= limits_high+1e-6))
        fingers = set(); support = False; handle_contacts = set(); handle_fingers = {}
        loaded_contact = False
        for index, contact in enumerate(data.contact):
            if contact.dist > .0002:
                continue
            pair = (int(contact.geom1), int(contact.geom2))
            if self.object in pair:
                other = pair[1] if pair[0] == self.object else pair[0]
                if model.geom_bodyid[other] in self.fingers:
                    fingers.add(int(model.geom_bodyid[other]))
                    if contact.dist <= .0001:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        loaded_contact |= bool(np.linalg.norm(force[:3]) > 1e-9)
                support |= other == self.surface
            if self.task == 4:
                for a,b in (pair,pair[::-1]):
                    name = model.geom(a).name or ""
                    if name.startswith("handle") and name[6:].isdigit() and model.geom_bodyid[b] in self.fingers:
                        handle_contacts.add(int(name[6:]))
                        handle_fingers.setdefault(int(name[6:]), set()).add(int(model.geom_bodyid[b]))
        if len(handle_contacts) == 1 and self.first_drawer_contact is None:
            self.first_drawer_contact = next(iter(handle_contacts))
        if self.task == 4:
            positions = [float(data.qpos[model.joint(f"drawer_slide{k}").qposadr[0]]) for k in range(4)]
            self.wrong_drawer_opened |= any(value > .02 for k,value in enumerate(positions) if k != self.expected_drawer)
        velocity = data.qvel[self.velocity_start:self.velocity_start+6]
        linear, angular = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
        z = float(data.geom_xpos[self.object,2])
        self.two_finger_seen |= len(fingers) == 2
        self.lifted |= z > self.start_z+.05
        released = not fingers
        gripper_open = float(self.env.gripper) > .5
        opening = gripper_open and not self.previous_gripper_open
        self.previous_gripper_open = gripper_open
        relative = None
        if self.hand >= 0:
            hand_rotation = data.xmat[self.hand].reshape(3,3)
            relative = hand_rotation.T @ (data.geom_xpos[self.object]-data.xpos[self.hand])
        stable = linear < .025 and angular < .15
        surface_top = float(size[1] if self.task == 1 else size[2])
        bottom_on_surface = abs(float(lower[2])-surface_top) <= .003
        supported_placement = bool(contained and support and released and stable and bottom_on_surface)
        if supported_placement:
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.stable_since = None
        release_clearance = float(lower[2])-surface_top
        release_window = contained and gripper_open and -.003 <= release_clearance <= .10
        legal_release = released and release_window
        # `contained` above describes the XY footprint, including while carrying
        # an object over its destination. The environment's placement field also
        # requires the original height interval. Keep both observable facts so
        # their comparison cannot mistake a suspended object for a disagreement.
        contained_at_placement_height = bool(contained)
        if self.task in (3, 4):
            table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
            require(table >= 0, "Contact task is missing its table geometry")
            tabletop = float(geom_vertices(model, data, table)[:, 2].max())
            low_offset, high_offset = ((.003, .012) if self.task == 3 else (0., .010))
            bottom = float(vertices[:, 2].min())
            contained_at_placement_height &= bool(tabletop+low_offset < bottom < tabletop+high_offset)
        # Independently infer a free loaded grasp from raw contact forces. Do
        # not read environment retention flags or change the lift milestone.
        object_body = int(model.geom_bodyid[self.object])
        weight = float(model.body_mass[object_body] * np.linalg.norm(model.opt.gravity))
        minimum_load = max(1e-6, weight * .01)
        finger_loads = {int(body): 0.0 for body in self.fingers}
        external_upward_load = 0.0
        forces_valid = True
        for index, contact in enumerate(data.contact):
            if self.object not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force[:3]).all():
                forces_valid = False
                continue
            other = contact.geom2 if contact.geom1 == self.object else contact.geom1
            body = int(model.geom_bodyid[other])
            if body in finger_loads:
                finger_loads[body] += float(np.linalg.norm(force[:3]))
            else:
                world_force = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
                object_force = world_force if contact.geom2 == self.object else -world_force
                external_upward_load += max(0.0, float(object_force[2]))
        free_hold = bool(
            self.task in (3, 4) and not gripper_open and forces_valid
            and all(load >= minimum_load for load in finger_loads.values())
            and z > self.start_z + .001
            and external_upward_load < weight * .05
        )
        if free_hold:
            if self.free_hold_since is None:
                self.free_hold_since = now
        else:
            self.free_hold_since = None
        sustained_free_hold = bool(self.free_hold_since is not None
                                   and now-self.free_hold_since >= .04-1e-9)
        legacy_lift = bool(len(fingers) == 2 and z > self.start_z + .05
                           and (self.task == 1 or not self.release_committed))
        if sustained_free_hold or legacy_lift:
            self.retention_armed = True
            self.release_committed = False
        monitor_retention = self.lifted if self.task == 1 else self.retention_armed
        if monitor_retention and len(fingers) < 2 and not self.release_committed:
            if self.gap_since is None:
                self.gap_since, self.gap_z = now,z
                self.gap_descent = 0.0
                self.gap_release_allowed = bool(release_window)
                self.gap_closed_hold = bool(not gripper_open and loaded_contact and relative is not None)
                self.gap_hold_relative = None if relative is None else relative.copy()
                self.last_loaded_time = now if loaded_contact else None
            # A closed, loaded one-finger hold may follow a lowering hand.
            # Only that original hold can authorize a subsequent opening edge;
            # a falling object entering the release band or touching later cannot.
            if self.gap_closed_hold:
                if loaded_contact:
                    self.last_loaded_time = now
                self.gap_closed_hold &= bool(
                    not self.drop_seen
                    and np.linalg.norm(relative-self.gap_hold_relative) <= .001
                    and now-self.last_loaded_time <= model.opt.timestep+1e-9)
                if opening and self.gap_closed_hold and loaded_contact and release_window:
                    self.gap_release_allowed = True
                if gripper_open:
                    self.gap_closed_hold = False
            self.gap_release_allowed &= bool(release_window)
            if legal_release and self.gap_release_allowed:
                self.release_committed = True
                self.retention_armed = False
                self.free_hold_since = None
            # A later bounce must not erase descent already observed during
            # the same contact interruption. This is sampled evidence only;
            # events between observation ticks remain unobserved.
            self.gap_descent = max(self.gap_descent, self.gap_z-z)
            self.drop_seen |= (not self.gap_release_allowed
                              and now-self.gap_since >= .1-1e-8
                              and self.gap_descent > .015 and not self.gap_closed_hold)
        else:
            self.gap_since = None
            self.gap_descent = 0.0
            self.gap_release_allowed = False
            self.gap_closed_hold = False
            self.gap_hold_relative = None
            self.last_loaded_time = None
        return {"time":now,"contained":bool(contained),
                "contained_xy":bool(contained),"contained_at_placement_height":contained_at_placement_height,
                "supported":bool(support),"released":bool(released),
                "stable":bool(stable),"physical_placement":supported_placement,
                "gripper_open":gripper_open,"legal_release_committed":self.release_committed,
                "bottom_on_target_surface":bool(bottom_on_surface),
                "stable_seconds":0. if self.stable_since is None else now-self.stable_since,
                "linear_speed_m_s":linear,"angular_speed_rad_s":angular,
                "two_finger_contact":len(fingers)==2,"two_finger_grasp_observed":bool(self.two_finger_seen),
                "two_finger_handle_contact":any(len(bodies) == 2 for bodies in handle_fingers.values()),
                "lift_observed":bool(self.lifted),"drop_observed":bool(self.drop_seen),
                "expected_drawer_from_initial_geometry":self.expected_drawer,
                "first_contacted_drawer":self.first_drawer_contact,"wrong_drawer_opened":bool(self.wrong_drawer_opened),
                "object_bounds_in_target_frame":[lower.tolist(),upper.tolist()],
                "xy_boundary_excess_m":(None if self.task == 1 else
                    float(max(0., np.max(limits_low-lower[:2]), np.max(upper[:2]-limits_high)))),
                "xy_numerical_tolerance_m":0. if self.task == 1 else 1e-6,
                "target_inner_xy_bounds":[limits_low.tolist(),limits_high.tolist()]}


class TaskProgressObserver:
    """Diagnostic milestones, not a success oracle or an early-stop policy.

    Arm motion alone is not task progress. Report repeated empty closures and
    elapsed time without new grasp/lift/drawer/placement evidence. Missing a
    control-tick contact is possible, so these are observations, not certified
    failed grasp counts.
    """
    def __init__(self, now, gripper, *, continuous_gripper=False):
        self.last_progress = float(now)
        self.max_gap = 0.
        # A Task4 handle approach requests aperture .25. Counting every
        # crossing of .5 as a grasp mislabels normal aperture changes as
        # repeated empty closures. Endpoint hysteresis is diagnostic only;
        # it must never modify the executed aperture or task outcome.
        self.continuous_gripper = bool(continuous_gripper)
        self.close_threshold = .1 if self.continuous_gripper else .5
        self.reopen_threshold = .2 if self.continuous_gripper else .5
        self.was_closed = float(gripper) <= self.close_threshold
        self.was_open = float(gripper) > .5
        self.pending_close = False
        self.empty_closures = 0
        self.close_attempts = 0
        self.milestones = set()
        self.drawer_highwater = np.zeros(4)

    def sample(self, physics, gripper, drawer_positions=None):
        now = float(physics['time'])
        opened = float(gripper) > .5
        closed = float(gripper) <= self.close_threshold
        closing_edge = (closed and not self.was_closed if self.continuous_gripper
                        else self.was_open and not opened)
        if closing_edge:
            self.close_attempts += 1
            self.pending_close = True
        contact = physics['two_finger_contact'] or physics.get('two_finger_handle_contact', False)
        if contact:
            self.pending_close = False
        elif (float(gripper) >= self.reopen_threshold if self.continuous_gripper else opened) and self.pending_close:
            self.empty_closures += 1
            self.pending_close = False
        self.was_open = opened
        if not self.continuous_gripper or closed or float(gripper) >= self.reopen_threshold:
            self.was_closed = closed
        new = {name for name in ('two_finger_grasp_observed', 'lift_observed', 'physical_placement')
               if physics.get(name)} - self.milestones
        if physics.get('two_finger_handle_contact'):
            new |= {'two_finger_handle_contact_observed'} - self.milestones
        if physics.get('first_contacted_drawer') is not None:
            new |= {'drawer_contact_observed'} - self.milestones
        self.milestones.update(new)
        drawer_progress = False
        if drawer_positions is not None:
            positions = np.asarray(drawer_positions, dtype=float)
            drawer_progress = bool(np.any(positions > self.drawer_highwater + .005))
            if drawer_progress:
                self.drawer_highwater = np.maximum(self.drawer_highwater, positions)
        gap = max(0., now-self.last_progress)
        self.max_gap = max(self.max_gap, gap)
        if new or drawer_progress:
            self.last_progress = now
        return dict(close_attempts=self.close_attempts,
                    reopened_without_observed_bilateral_contact=self.empty_closures,
                    pending_close_without_observed_bilateral_contact=self.pending_close,
                    seconds_without_new_milestone=now-self.last_progress,
                    longest_seconds_without_new_milestone=self.max_gap,
                    observed_milestones=sorted(self.milestones),
                    gripper_diagnostic_mode=('continuous_aperture' if self.continuous_gripper else 'binary'),
                    diagnostic_only=True)


def physics_disagreements(outcome, physics):
    """Compare like-for-like physical fields, not strict success vs placement."""
    placement = outcome.get('placement', {})
    result = {}
    for name in ('contained', 'released', 'supported'):
        if name in placement:
            observed = (physics.get('contained_at_placement_height', physics[name])
                        if name == 'contained' else physics[name])
            result[name] = bool(placement[name]) != bool(observed)
    reasons = {event['reason'] for event in outcome.get('failure_events', [])}
    result['drop'] = ('transport_drop' in reasons) != bool(physics['drop_observed'])
    return result


def summarize_outcomes(rows):
    completed = [r for r in rows if r["outcome"] != "censored"]
    successes = sum(r["environment_success"] for r in completed)
    missing = len(rows)-len(completed)
    return {"episodes":len(rows), "resolved_episodes":len(completed), "censored_episodes":missing,
            "environment_success_rate":successes/len(completed) if completed else None,
            "environment_success_wilson_95":wilson_interval(successes,len(completed)),
            "all_episode_success_rate_bounds":[successes/len(rows),(successes+missing)/len(rows)] if rows else None,
            "placement_disagreements":sum(r["placement_disagreement"] for r in rows),
            "drop_disagreements":sum(r.get("drop_disagreement", False) for r in rows),
            "episodes_with_tick_disagreements":sum(any(r.get('physics_disagreement_steps', {}).values()) for r in rows),
            "reopened_without_observed_bilateral_contact":sum(r.get('task_progress', {}).get('reopened_without_observed_bilateral_contact', 0) for r in rows),
            "placed_but_task_unsuccessful":sum(not r["environment_success"] and r["physics"]["physical_placement"] for r in rows),
            "observed_drops":sum(r["physics"]["drop_observed"] for r in rows),
            "status":"incomplete_censored_runs" if missing else "measured_closed_loop",
            "independent_full_task_success_certified":False,
            "note":"Actual environment success, cross-checked by independent control-tick physical observations. Not an open-loop prediction; substep-only failures and recovery completeness are not certified."}


def run_closed_loop_suite(predictor, task_seeds, memory_frames, stride, output, max_steps=None,
                          *, target_override=None, stop_after_selection=False):
    from mujoco_pickplace.tasks import get_task, evaluation_failure_stop, task_environment_contract
    from mujoco_pickplace.episode_dataset import compare_environment_contract
    from mujoco_pickplace.eval_policy_client import GripperCommandFilter, PrecisionExecutionController
    import mujoco
    output.mkdir(parents=True,exist_ok=False)
    reports = []
    legacy = predictor.model.legacy_inference_contract
    for number,seeds in task_seeds.items():
        task = get_task(number)
        compatibility = compare_environment_contract(
            (predictor.model.config.get("task_environment_contracts") or {}).get(task.key), task_environment_contract(task))
        if compatibility["status"] == "mismatch":
            raise ValueError(f"Task{number} checkpoint/environment mismatch: {compatibility['differences']}")
        if task.action_mask != TASK_CONTRACTS[number].action_mask or task.instruction != TASK_CONTRACTS[number].instruction:
            raise ValueError("Runtime task registry differs from independent action/instruction contract")
        for seed in seeds:
            env = task.make_env(seed=seed, **(dict(image_size=448,randomize_task=True,randomization_scale=1.)
                                            if number==1 else dict(image_size=448,capture_frames=False,
                                                **({"target":target_override} if number==4 and target_override is not None else {}))))
            history = []
            def remember(obs):
                buf=io.BytesIO(); Image.fromarray(obs["image_front"]).save(buf,format="PNG")
                history.append((float(env.data.time),obs["robot_state"].copy(),base64.b64encode(buf.getvalue()).decode("ascii")))
                window=(memory_frames-1)*stride
                while len(history)>1 and history[1][0] < env.data.time-window-1e-6:
                    history.pop(0)
            observation_hooks = ExitStack()
            try:
                if number==1:
                    env.GRASP_X_BIAS = .006 if legacy else .004
                initial_state_hash=digest_arrays(env.data.qpos,env.data.qvel,env.model.body_pos)
                remember(env.obs())
                if number==4:
                    for obs in env.presentation():
                        remember(obs)
                if number != 1:
                    env.phase = "policy_execution"
                observer=PhysicsObserver(env,number)
                evidence=observer.sample()
                if number != 1:
                    observation_hooks.enter_context(observer.monitor_substeps())
                progress=TaskProgressObserver(env.data.time, env.gripper,
                                              continuous_gripper=task.gripper_mode == 'continuous')
                progress_evidence=progress.sample(evidence, env.gripper)
                disagreement_steps={}
                grip=GripperCommandFilter(required_steps=task.gripper_open_required_steps or 2)
                precision=PrecisionExecutionController()
                default_budget=task.evaluation_max_steps or task.max_steps
                budget=min(default_budget,max_steps) if max_steps is not None else default_budget
                step=0; done=False; stop=None; first_failure=None; selection_history=None
                contract={"mujoco_version":mujoco.__version__,"scene_sha256":sha256(task.scene),
                          "independent_physics_sampling":("control_boundaries_only" if number == 1 else
                              "each_retention_physics_substep_and_control_boundary"),
                          "panda_xml_sha256":sha256(Path(__file__).resolve().parents[1]/"mujoco_menagerie/franka_emika_panda/panda.xml"),
                          "initial_state_sha256":initial_state_hash,"control_period_seconds":float(env.model.opt.timestep*env.CONTROL_NSTEP),
                          "model_dynamics_sha256":digest_arrays(env.model.geom_size,env.model.geom_friction,
                              env.model.body_mass,env.model.body_inertia,env.model.body_gravcomp,
                              env.model.actuator_gainprm,env.model.actuator_biasprm,env.model.mesh_vert),
                          "max_steps":budget,"standard_max_steps":default_budget,
                          "execution_horizon":(14 if legacy else 4) if number==1 else 1,
                          "gripper_debounce_steps":task.gripper_open_required_steps or 2,
                          "precision_replan":not legacy if number==1 else False}
                trace=output/f"task{number}_seed{seed}.jsonl"
                with trace.open("x",encoding="utf-8") as stream:
                    while step<budget and not done and not stop:
                        times=np.array([h[0] for h in history]); indices,valid=history_indices(times,len(times)-1,memory_frames,stride)
                        payload={"memory_images":[[history[i][2]] for i in indices],"image_encoding":"png_base64",
                                 "state":[history[i][1].tolist() for i in indices],"history_mask":valid.tolist(),
                                 "image_mask":[1],"action_mask":list(task.action_mask)+[0]*17,
                                 "prompt":task.instruction,"task_key":task.key,"flow_seed":(int(seed)+step)%(2**63)}
                        # Snapshot the actual query before remember() can trim
                        # history or a multi-action chunk advances simulation.
                        query_times=[history[i][0] for i in indices]
                        query_start_step=step
                        query_time=history[-1][0]
                        chunk=predictor(payload)
                        if chunk.ndim!=2 or chunk.shape[1]!=24 or not len(chunk) or not np.isfinite(chunk).all():
                            raise ValueError("Invalid closed-loop model output")
                        horizon=min(contract["execution_horizon"],len(chunk),budget-step)
                        if number==1:
                            horizon=precision.execution_horizon(chunk,horizon,history[-1][1],grip,enabled=not legacy)
                        for action_index,raw in enumerate(chunk[:horizon]):
                            action=task.prepare_action(raw[:7],grip)
                            before=history[-1][1].copy()
                            if number == 1:
                                # Match the production evaluator's substep physics,
                                # including force refresh and contact monitoring.
                                _,obs,done=env.step_video(action,frames_per_step=1,cameras=())
                            else:
                                obs,done=env.step(action)
                            if number==4 and env.first_selected_drawer is not None and selection_history is None:
                                selected_times=[history[i][0] for i in indices]
                                slots=[slot for slot,(t,ok) in enumerate(zip(selected_times,payload["history_mask"]))
                                       if ok and any(abs(t-v)<1e-6 for v in env._target_visible_times)]
                                selection_history=dict(times=selected_times,target_visible_slots=slots,
                                                       target_available=bool(slots))
                            step+=1; remember(obs); evidence=observer.sample()
                            drawer_positions = ([float(env.data.qpos[env.model.joint(f'drawer_slide{k}').qposadr[0]])
                                                 for k in range(4)] if number == 4 else None)
                            progress_evidence=progress.sample(evidence, env.gripper, drawer_positions)
                            disagreements=physics_disagreements(env.outcome(), evidence) if number != 1 else {}
                            for name, differs in disagreements.items():
                                disagreement_steps[name]=disagreement_steps.get(name, 0)+int(differs)
                            row={"step":step,"raw_action":raw[:7].tolist(),"prepared_action":action.tolist(),
                                 "state_before":before.tolist(),"state_after":obs["robot_state"].tolist(),
                                 "environment_success":bool(done),"physics":evidence,
                                 "task_progress":progress_evidence,"physics_disagreements":disagreements}
                            execution_rows=getattr(env,"rows",[])
                            execution_row=execution_rows[-1] if execution_rows else {}
                            row.update(
                                simulation_time=float(env.data.time),
                                query_simulation_time=query_time,
                                inference_start_step=query_start_step,
                                inference_action_index=action_index,
                                flow_seed=payload["flow_seed"],
                                sampled_history_times=query_times,
                                history_mask=list(payload["history_mask"]),
                                ik_status=(dict(env.ik_status) if getattr(env,"ik_status",None) is not None else {}),
                                ik_status_available=getattr(env,"ik_status",None) is not None,
                                drawer_positions=drawer_positions,
                                object_geometry_center_world_m=(env.data.geom_xpos[env.object_geom].tolist()
                                                                if number != 1 else None),
                                pose_tracking=execution_row.get("pose_tracking",{}),
                                unexpected_robot_contacts_peak=execution_row.get("unexpected_robot_contacts_peak",[]),
                                quality_failure_events=list(getattr(env,"failure_events",[])),
                            )
                            policy_diagnostics=getattr(predictor,"last_diagnostics",None)
                            if policy_diagnostics is not None:
                                row["policy_diagnostics"]=policy_diagnostics
                            stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
                            if number!=1:
                                if env.failure_events and first_failure is None:
                                    first_failure=step
                                stop=evaluation_failure_stop(env,step-first_failure if first_failure is not None else 0,0)
                            if stop_after_selection and number==4 and env.first_selected_drawer is not None and not stop:
                                stop="confirmed_drawer_selection_probe_end"
                            if done or stop:
                                break
                censored=not done and ((not stop and budget<default_budget)
                                       or stop=="confirmed_drawer_selection_probe_end")
                reports.append({"task":number,"seed":int(seed),"steps":step,"environment_success":bool(done),
                    "drop_disagreement":bool(number != 1 and env.transport_dropped != evidence["drop_observed"]),
                    "physics_disagreement_steps":disagreement_steps,"task_progress":progress_evidence,
                    "environment_compatibility":compatibility,
                    "first_selected_drawer":getattr(env,"first_selected_drawer",None),
                    "target_drawer_annotation_only":int(env.target) if number==4 else None,
                    "confirmed_drawer_selection_correct":(
                        bool(env.first_selected_drawer==env.target)
                        if number==4 and env.first_selected_drawer is not None else None),
                    "history_at_confirmed_selection":selection_history,
                    "task_outcome":env.outcome() if number!=1 else None,
                    "outcome":"censored" if censored else "success" if done else "failure",
                    "termination_reason":stop or ("shortened_probe_budget" if censored else "environment_success" if done else "standard_step_limit"),
                    "physics":evidence,"placement_disagreement":bool(done and not evidence["physical_placement"]),
                    "checkpoint_identity":predictor.identity,"run_contract":contract,
                    "feedback_source":"new_rendered_environment_observation_after_each_executed_action"})
                (output/"episodes.json").write_text(json.dumps(reports,indent=2)+"\n")
            finally:
                try:
                    observation_hooks.close()
                finally:
                    env.renderer.close()
    result={"tasks":{str(task):summarize_outcomes([r for r in reports if r["task"]==task]) for task in task_seeds},
            "episodes":reports,"success_rate_source":"real_closed_loop_rollouts_not_action_error_conversion"}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def run_drawer_selection_probe(predictor, memory_frames, stride, output, seed, max_steps=40):
    """Four controlled histories, full/current-only, through real contact control.

    A confirmed selection is bilateral contact or drawer opening from the
    production environment. This bounded subtask is not whole-task success.
    """
    if memory_frames < 2:
        return dict(status="not_applicable_k1_no_past_frames", selection_success_certified=False)
    output.mkdir(parents=True, exist_ok=False)
    first_inputs = {}
    rows = []
    policy_requests = 0
    for target in range(4):
        for variant in ("full", "current_only"):
            class ProbePredictor:
                model = predictor.model
                identity = predictor.identity

                def __call__(self, payload):
                    nonlocal policy_requests
                    policy_requests += 1
                    # The target override only initialized the scene. It is
                    # never added to this public observation request.
                    if (target, variant) not in first_inputs:
                        first_inputs[(target, variant)] = dict(
                            current=payload["memory_images"][-1], state=payload["state"],
                            mask=payload["history_mask"],
                            history_sha256=hashlib.sha256(json.dumps(payload["memory_images"]).encode()).hexdigest())
                        anchor = first_inputs[(0,"full")]
                        require(payload["memory_images"][-1]==anchor["current"]
                                and payload["history_mask"]==anchor["mask"]
                                and np.allclose(payload["state"],anchor["state"],rtol=0,atol=1e-7),
                                "Uncontrolled drawer selection probe: initial current image/state histories differ")
                    if variant == "current_only":
                        # Keep the execution audit's payload in sync with the
                        # actual masked request, including selection-time evidence.
                        payload["history_mask"] = [False]*(memory_frames-1)+[True]
                    return predictor(payload)

            result = run_closed_loop_suite(ProbePredictor(), {4:[seed]}, memory_frames, stride,
                output/f"target{target}_{variant}", max_steps,
                target_override=target, stop_after_selection=True)
            row = dict(target=target, variant=variant, **result["episodes"][0])
            # Ending a selection probe says nothing about full manipulation.
            row["whole_task_success_measured"] = False
            rows.append(row)
            (output/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
    anchor = first_inputs[(0,"full")]
    for initial in first_inputs.values():
        require(initial["current"]==anchor["current"] and initial["mask"]==anchor["mask"]
                and np.allclose(initial["state"],anchor["state"],rtol=0,atol=1e-7),
                "Uncontrolled drawer selection probe: initial current image/state histories differ")
    require(len({first_inputs[(target,"full")]["history_sha256"] for target in range(4)})==4,
            "Drawer selection probe lacks four distinct target presentation histories")
    groups = {}
    for variant in ("full","current_only"):
        selected = [row for row in rows if row["variant"]==variant]
        groups[variant] = dict(targets=4,
            correct_confirmed=sum(row["confirmed_drawer_selection_correct"] is True for row in selected),
            wrong_confirmed=sum(row["confirmed_drawer_selection_correct"] is False for row in selected),
            unresolved=sum(row["confirmed_drawer_selection_correct"] is None for row in selected))
    result = dict(status="measured_selection_probe_not_full_task_acceptance", seed=seed,
                  max_steps_per_case=max_steps, paired_current_and_state_history_verified=True,
                  distinct_four_target_histories_verified=True, policy_requests=policy_requests,
                  groups=groups, episodes=rows, whole_task_success_rate=None,
                  selection_success_certified=False)
    (output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def independent_robot_state(env):
    """Read the specified 8-D proprioception directly from MuJoCo state."""
    model,data=env.model,env.data
    body=model.body("hand").id
    quat=data.xquat[body].copy();quat/=np.linalg.norm(quat)
    w=float(np.clip(quat[0],-1,1)); den=np.sqrt(max(0.,1-w*w))
    rotation=np.zeros(3) if den<1e-8 else quat[1:]*2*np.arccos(w)/den
    return np.r_[data.xpos[body],rotation,
                 [data.qpos[model.joint(n).qposadr[0]] for n in ("finger_joint1","finger_joint2")]]


def recorded_replay_initialization(episode, task_number):
    """Restore the recorded presentation for expert replay only."""
    if task_number != 4:
        return {}
    target = recorded_drawer_target(episode.row, task_number)
    metadata = read_json(episode.root / 'meta/dataset.json')
    protocol = metadata.get('paired_target_initialization')
    if protocol is None:
        return {'target': target}
    require(task_number == 4 and isinstance(protocol, dict)
            and type(protocol.get('version')) is int and protocol['version'] in (1, 2)
            and protocol.get('method') == 'fixed_nuisance_seed_target_override',
            'Unsupported recorded paired-target initialization')
    seeds = protocol.get('nuisance_seeds')
    require(isinstance(seeds, list) and seeds
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and type(episode.row.get('seed')) is int and episode.row['seed'] in seeds,
            'Paired-target episode has an undeclared nuisance seed')
    targets = protocol.get('target_overrides')
    # Version 1 is the original four-target paired protocol. Version 2 records
    # a nonempty, explicitly selected subset (for example one real correction),
    # without claiming that uncollected target pairs exist. Both are replay-only.
    valid_targets = (isinstance(targets, list) and bool(targets)
                     and all(type(value) is int and value in range(4) for value in targets)
                     and len(set(targets)) == len(targets)
                     and (protocol['version'] == 2 or targets == [0, 1, 2, 3]))
    require(valid_targets and type(target) is int and target in range(4) and target in targets,
            'Paired-target episode lacks a valid recorded presentation target')
    return {'target': target}


def replay_alignment(episode, task_number, execution_steps, *, image_size=64, observe_terminal=False):
    """Check stored pre-action states against independently read physical states.

    Bounded prefixes are evidence only for the tested prefix. A failure can mean
    misaligned data, non-replayable expert labels, or a changed environment.
    """
    from mujoco_pickplace.tasks import get_task
    require(not observe_terminal or task_number in (3,4),
            "Independent terminal replay observation is limited to the native contact tasks")
    task=get_task(task_number)
    initialization = recorded_replay_initialization(episode, task_number)
    env=task.make_env(seed=episode.row["seed"], **(dict(image_size=64,randomize_task=True,randomization_scale=1.)
                                                if task_number==1 else dict(image_size=image_size,capture_frames=False,
                                                                          **initialization)))
    errors=[];executed=0;index=0
    observation_hooks=ExitStack()
    observer=None
    context_required=bool(np.any(episode.phases == "policy_context"))
    empty_context_verified=False
    handle_context_verified=False
    handle_errors=[]
    context_indices=np.flatnonzero(episode.phases == "policy_context")
    context_end=int(context_indices[-1])+1 if context_required else 0
    handle_context_evidence={}
    handle_bilateral_seen=False
    handle_prefix_maximum_opening=0.
    context_source=episode.row.get('quality',{}).get('correction_source',{}) if context_required else {}
    policy_state_context=task_number==4 and context_source.get('kind')=='handle_policy_state'
    object_failure_context=task_number==4 and context_source.get('kind')=='object_empty_grasp'
    policy_prefix_active=False
    policy_prefix_substeps=0
    policy_prefix_bilateral=False
    policy_prefix_selected=False
    policy_prefix_unsafe=False
    policy_prefix_opening=0.
    policy_prefix_max_state_error=0.
    try:
        import mujoco
        execution_contract=dict(mujoco_version=mujoco.__version__,
            control_period_seconds=float(env.model.opt.timestep*env.CONTROL_NSTEP),
            recorded_initialization_overrides=initialization)
        if task_number==1:
            # Raw expert replay uses the frame recorded at collection, while
            # policy evaluation negotiates its frame with the checkpoint.
            # The legacy environment default is deliberately kept for old models.
            metadata=read_json(episode.root/'meta/dataset.json')
            bias=metadata.get('collection_config',{}).get('expert_grasp_x_bias_m')
            require(isinstance(bias,(int,float)) and np.isfinite(bias),
                    'Task1 expert replay requires its recorded grasp frame')
            env.GRASP_X_BIAS=float(bias)
            execution_contract['grasp_x_bias_m']=env.GRASP_X_BIAS
        if observe_terminal:
            observer=PhysicsObserver(env,task_number)
            observer.sample()
            observation_hooks.enter_context(observer.monitor_substeps())
        if policy_state_context:
            # Independent raw-physics proof, including instantaneous bilateral
            # contact on any handle. Do not trust the collector's certificate.
            handles={env.model.geom(f'handle{k}').id for k in range(4)}
            fingers={env.model.body(name).id for name in ('left_finger','right_finger')}
            slides=[env.model.joint(f'drawer_slide{k}').qposadr[0] for k in range(4)]
            had_override='_update_object_retention' in env.__dict__
            previous_override=env.__dict__.get('_update_object_retention')
            original_retention=env._update_object_retention
            def observe_policy_prefix():
                nonlocal policy_prefix_substeps,policy_prefix_bilateral,policy_prefix_selected
                nonlocal policy_prefix_unsafe,policy_prefix_opening
                result=original_retention()
                if policy_prefix_active:
                    policy_prefix_substeps+=1
                    contacts={handle:set() for handle in handles}
                    for contact in env.data.contact:
                        if contact.dist>.0001:continue
                        for handle,other in ((int(contact.geom1),int(contact.geom2)),
                                             (int(contact.geom2),int(contact.geom1))):
                            if handle in contacts and int(env.model.geom_bodyid[other]) in fingers:
                                contacts[handle].add(int(env.model.geom_bodyid[other]))
                    policy_prefix_bilateral|=any(len(bodies)==2 for bodies in contacts.values())
                    policy_prefix_opening=max(policy_prefix_opening,max(abs(float(env.data.qpos[j])) for j in slides))
                    policy_prefix_selected|=env.first_selected_drawer is not None
                    policy_prefix_unsafe|=bool(env.failure_events or env.unsafe_execution_contact
                                              or env.transport_dropped or env.motion_failures)
                return result
            def restore_policy_prefix_monitor():
                if had_override:env._update_object_retention=previous_override
                else:del env._update_object_retention
            env._update_object_retention=observe_policy_prefix
            observation_hooks.callback(restore_policy_prefix_monitor)
        presentation=iter(env.presentation()) if task_number==4 else None
        while index<len(episode.states):
            actual=independent_robot_state(env)
            target=episode.states[index]
            error={"frame":index,"xyz_error_mm":float(np.linalg.norm(actual[:3]-target[:3])*1000),
                   "rotation_vector_error_rad":float(np.linalg.norm(actual[3:6]-target[3:6])),
                   "finger_error_m":float(np.max(np.abs(actual[6:]-target[6:]))),
                   "time_error_seconds":abs(float(env.data.time)-float(episode.times[index]))}
            errors.append(error)
            if policy_state_context and episode.phases[index]=='policy_context':
                coordinate_error=float(np.max(np.abs(actual-target)))
                policy_prefix_max_state_error=max(policy_prefix_max_state_error,coordinate_error)
                require(np.isfinite(coordinate_error) and coordinate_error<=1e-4,
                        'Policy-state prefix does not reproduce its recorded observation')
            if error["xyz_error_mm"]>.5 or error["rotation_vector_error_rad"]>.01 or error["finger_error_m"]>.001 or error["time_error_seconds"]>1e-4:
                return {"status":"failed","first_mismatch":error,"checked_states":len(errors),
                        "execution_contract":execution_contract,
                        "possible_causes":["state/action timing error","non-replayable expert labels","environment version mismatch"],
                        "image_alignment_verified":False}
            if episode.phases[index] in {"observe_target","presentation_close"}:
                if presentation is None:
                    raise ValueError("Human presentation phase outside Task4")
                next(presentation)
            else:
                if executed >= execution_steps:
                    break
                # This is an expert-command replay, including history-only
                # policy prefixes. Preserve labels for phase contact audits;
                # no expert controller or privileged correction is called.
                env.phase=str(episode.phases[index])
                policy_prefix_active=policy_state_context and episode.phases[index]=='policy_context'
                env.step(episode.actions[index])
                policy_prefix_active=False
                executed+=1
                if episode.phases[index] == "policy_context" and task_number == 3:
                    # Independently inspect the actual replayed physical state.
                    # A metadata flag alone is not proof of an empty grasp.
                    finger_ids={env.model.body(n).id for n in ('left_finger','right_finger')}
                    object_id=env.model.geom('object_geom').id
                    touching=any(c.dist <= .0001 and object_id in (c.geom1,c.geom2)
                        and env.model.geom_bodyid[c.geom2 if c.geom1==object_id else c.geom1] in finger_ids
                        for c in env.data.contact)
                    aperture=sum(env.data.qpos[env.model.joint(n).qposadr[0]]
                                 for n in ('finger_joint1','finger_joint2'))
                    failed_time=episode.row['quality']['correction_source']['recorded_empty_grasp_sim_time']
                    empty_context_verified |= bool(episode.actions[index,6] < .5
                        and aperture < .004 and not touching
                        and abs(env.data.time-failed_time) < 1e-4)
                elif episode.phases[index] == "policy_context" and object_failure_context:
                    require(not env.failure_events and not env.was_lifted,
                            "Object correction prefix contains a safety failure or prior lift")
                    if index+1 == context_end:
                        finger_bodies={env.model.body(name).id for name in ('left_finger','right_finger')}
                        touching=set()
                        object_id=env.model.geom('object_geom').id
                        for contact in env.data.contact:
                            if contact.dist>.0001 or object_id not in (contact.geom1,contact.geom2):
                                continue
                            other=contact.geom2 if contact.geom1==object_id else contact.geom1
                            body=int(env.model.geom_bodyid[other])
                            if body in finger_bodies:
                                touching.add(body)
                        aperture=sum(float(env.data.qpos[env.model.joint(name).qposadr[0]])
                                     for name in ('finger_joint1','finger_joint2'))
                        opening=float(env.data.qpos[env.model.joint(f'drawer_slide{env.target}').qposadr[0]])
                        handover_error=float(np.max(np.abs(
                            independent_robot_state(env)-episode.states[context_end])))
                        handle_context_verified=bool(
                            len(touching)<2 and aperture<.006 and env.gripper<=.2
                            and opening>=.165 and env.first_selected_drawer==env.target
                            and handover_error<=1e-4
                            and abs(env.data.time-context_source['recorded_intervention_sim_time'])<1e-4)
                        handle_context_evidence=dict(kind='object_empty_grasp',
                            raw_two_finger_contact=len(touching)==2,raw_aperture_m=aperture,
                            raw_drawer_open_m=opening,raw_selected_drawer=env.first_selected_drawer,
                            handover_max_state_coordinate_error=handover_error)
                        require(handle_context_verified,
                                f"Object intervention did not physically reproduce: {handle_context_evidence}")
                elif episode.phases[index] == "policy_context" and task_number == 4:
                    from mujoco_pickplace.task_env import HANDLE_ROTATION
                    # Read geometry directly, rather than treating the stored
                    # safe_handle_deviation_reproduced flag as physical proof.
                    hand_rotation=env.data.xmat[env.hand].reshape(3,3)
                    handle=env.model.geom(f"handle{env.target}").id
                    offset=(env.data.geom_xpos[handle] - env.data.xpos[env.hand]
                            - .1029*hand_rotation[:,2])
                    handle_errors.append(np.abs([offset@HANDLE_ROTATION[:,2],
                                                 offset[2],offset@HANDLE_ROTATION[:,0]]))
                    source=episode.row['quality']['correction_source']
                    contact_loss=source.get('kind')=='handle_contact_loss'
                    if contact_loss:
                        handle_prefix_maximum_opening=max(handle_prefix_maximum_opening,max(
                            abs(float(env.data.qpos[env.model.joint(f'drawer_slide{k}').qposadr[0]]))
                            for k in range(4)))
                    # Independently inspect which raw finger bodies touch the
                    # target handle; the source flag is not physical evidence.
                    finger_bodies={env.model.body(name).id for name in ('left_finger','right_finger')}
                    touching=set()
                    for contact in env.data.contact:
                        if contact.dist>.0001 or handle not in (contact.geom1,contact.geom2):continue
                        other=contact.geom2 if contact.geom1==handle else contact.geom1
                        body=int(env.model.geom_bodyid[other])
                        if body in finger_bodies:touching.add(body)
                    handle_bilateral_seen |= len(touching)==2
                    selection_safe=(env.first_selected_drawer is None
                                    or contact_loss and env.first_selected_drawer==env.target)
                    require(not env.failure_events and selection_safe,
                            "Handle correction prefix is unsafe or has an invalid prior selection")
                    if index+1 == context_end:
                        relative=HANDLE_ROTATION@hand_rotation.T
                        angle=float(np.arccos(np.clip((np.trace(relative)-1.)/2.,-1.,1.)))
                        deviation=float(np.max(handle_errors[-1]-np.min(handle_errors,axis=0)))
                        history_live=(env.last_visible_time is not None
                                      and env.data.time-env.last_visible_time <= 5.+1e-6)
                        # Include the state AFTER the final policy action: a
                        # pre-action-only check would miss a bad handover.
                        handover_error=float(np.max(np.abs(
                            independent_robot_state(env)-episode.states[context_end])))
                        event_reproduced=deviation>=.005 or angle>=np.deg2rad(2.)
                        if contact_loss:
                            maximum_opening=handle_prefix_maximum_opening
                            event_reproduced=bool(handle_bilateral_seen and len(touching)<2
                                and env.first_selected_drawer==env.target and env.gripper<=.1
                                and maximum_opening<=.005
                                and source['recorded_first_selected_drawer']==env.first_selected_drawer
                                and np.isclose(maximum_opening,source['recorded_max_drawer_open_m'],atol=1e-4,rtol=0))
                        if policy_state_context:
                            handle_context_verified=bool(
                                not policy_prefix_bilateral and not policy_prefix_selected
                                and not policy_prefix_unsafe and env.first_selected_drawer is None
                                and policy_prefix_opening<=.005 and history_live and handover_error<=1e-4
                                and policy_prefix_max_state_error<=1e-4
                                and policy_prefix_substeps==source['intervention_step']*env.CONTROL_NSTEP
                                and source['recorded_prefix_physics_substeps']==policy_prefix_substeps
                                and source['recorded_physics_substeps_per_control']==env.CONTROL_NSTEP
                                and abs(env.data.time-source['recorded_intervention_sim_time'])<1e-4
                                and np.isclose(policy_prefix_opening,source['recorded_max_drawer_open_m'],atol=1e-4,rtol=0))
                        else:
                            handle_context_verified=bool(
                                event_reproduced and history_live
                                and handover_error<=1e-4
                                and abs(env.data.time-source['recorded_deviation_sim_time'])<1e-4
                                and np.isclose(deviation,source['recorded_handle_deviation_m'],atol=1e-4,rtol=0)
                                and np.isclose(angle,source.get('recorded_handle_orientation_error_rad',0.),atol=1e-4,rtol=0))
                        handle_context_evidence=dict(
                            deviation_m=deviation,orientation_error_rad=angle,
                            kind=source.get('kind'),raw_bilateral_seen=bool(handle_bilateral_seen),
                            target_history_available=bool(history_live),
                            handover_max_state_coordinate_error=handover_error)
                        if policy_state_context:
                            handle_context_evidence.update(
                                confirmed_failure_claim=False,raw_bilateral_ever=bool(policy_prefix_bilateral),
                                raw_prefix_maximum_drawer_open_m=policy_prefix_opening,
                                raw_prefix_physics_substeps=policy_prefix_substeps,
                                raw_prefix_max_state_coordinate_error=policy_prefix_max_state_error,
                                no_prior_selection=not policy_prefix_selected and env.first_selected_drawer is None)
                        require(handle_context_verified,
                                f"Handle intervention condition did not physically reproduce: {handle_context_evidence}")
            index+=1
            if presentation is not None and (index==len(episode.states) or episode.phases[index] not in {"observe_target","presentation_close"}):
                try:
                    next(presentation)
                    raise ValueError("Presentation length differs from recorded episode")
                except StopIteration:
                    presentation=None
        context_verified=empty_context_verified if task_number == 3 else handle_context_verified
        require(not context_required or index < context_end or context_verified,
                'Policy context did not reproduce its recorded physical intervention condition')
        fully_executed=index==len(episode.states)
        result={"status":"passed_tested_prefix","checked_states":len(errors),"executed_actions":executed,
                "recorded_failure_context_verified":context_verified if context_required and not policy_state_context else None,
                "recorded_policy_context_verified":context_verified if policy_state_context else None,
                "handle_context_evidence":None if object_failure_context else handle_context_evidence,
                "object_context_evidence":handle_context_evidence if object_failure_context else None,
                "execution_contract":execution_contract,
                "whole_episode_verified":False,
                "whole_episode_executed":fully_executed,
                "all_recorded_pre_action_states_checked":len(errors)==len(episode.states),
                "terminal_post_action_state_verified":False,
                "max_xyz_error_mm":max(e["xyz_error_mm"] for e in errors),
                "max_rotation_vector_error_rad":max(e["rotation_vector_error_rad"] for e in errors),
                "max_finger_error_m":max(e["finger_error_m"] for e in errors),
                "max_time_error_seconds":max(e["time_error_seconds"] for e in errors),
                "image_alignment_verified":False}
        if observer is not None:
            # Original datasets do not contain a terminal post-action state.
            # Observe the actual replay endpoint without claiming comparison to
            # an absent original terminal state or authenticating its RGB image.
            result.update(independent_terminal_physics=observer.sample(),
                          current_terminal_outcome=env.outcome(),
                          replay_endpoint_robot_state=independent_robot_state(env).tolist(),
                          terminal_evidence_scope=("complete_replay_endpoint" if fully_executed
                                                   else "prefix_endpoint"),
                          independent_physics_sampling="physics_substeps_and_control_boundaries",
                          replay_image_size=image_size)
            # A matching sequence of pre-action robot states does not certify
            # that the recorded final action actually completed the task.
            # Prefixes remain unassessed; only a fully executed episode has a
            # terminal acceptance verdict. Preserve the alignment result above.
            physics = result['independent_terminal_physics']
            from mujoco_pickplace.task_env import trajectory_quality
            replay_quality=trajectory_quality(env,bool(result['current_terminal_outcome'].get('success',False)))
            result['replayed_expert_quality']={key:replay_quality.get(key) for key in (
                'checks','failed_checks','target_visible_history_slots_at_selection')}
            terminal_checks = {
                'formal_strict_success': bool(result['current_terminal_outcome'].get('success', False)),
                'formal_trajectory_quality': not replay_quality['failed_checks'],
                'independent_placement': bool(physics.get('physical_placement', False)),
                'independent_grasp': bool(physics.get('two_finger_grasp_observed', False)),
                'independent_lift': bool(physics.get('lift_observed', False)),
                'independent_no_drop': not physics.get('drop_observed', True),
                'independent_stable_one_second': physics.get('stable_seconds', 0.) >= 1. - 1e-8,
            }
            result['terminal_acceptance'] = {
                'status': ('passed' if all(terminal_checks.values()) else 'failed')
                          if fully_executed else 'not_assessed_prefix',
                'checks': terminal_checks if fully_executed else {},
                'failed_checks': [key for key, passed in terminal_checks.items() if not passed]
                                 if fully_executed else [],
            }
        return result
    finally:
        try:
            observation_hooks.close()
        finally:
            env.renderer.close()


def run(args, report):
    sys.path.insert(0, str(ROOT))
    import yaml
    tasks = [TASK_CONTRACTS[number] for number in args.tasks]
    config, stats, frames, horizon, stride = validate_checkpoint(args.ckpt_dir, tasks)
    report.update(checkpoint={"path": str(args.ckpt_dir), "config_sha256": sha256(args.ckpt_dir / "config.json"),
                              "norm_stats_sha256": sha256(args.ckpt_dir / "norm_stats.json"),
                              "weights_verified": False},
                  memory_frames=frames, memory_stride_seconds=stride, action_horizon=horizon,
                  datasets=[], query_count=0, prediction_count=0, coverage_complete=False,
                  generalization="unverified", warnings=[
                      "Open-loop action agreement cannot certify grasp retention, recovery or closed-loop success.",
                      "Current data contracts do not prove which scene/control period trained an old checkpoint.",
                      "History ablations measure sensitivity/utility on expert observations, not causal task success.",
                      "Dataset quality flags are declarations, not independently verified physical evidence.",
                  ])
    report["code_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in (
        Path(__file__).resolve(), EVO / "scripts/Evo1_server.py", EVO / "scripts/Evo1.py",
        EVO / "model/action_head/observation_decoder.py",
        EVO / "model/internvl3/temporal_vision_encoder.py", ROOT / "mujoco_pickplace/task_env.py",
        ROOT / "mujoco_pickplace/tasks.py", ROOT / "mujoco_pickplace/pick_place_env.py",
        ROOT / "mujoco_pickplace/eval_policy_client.py")}
    # The shipped archive has independently checked current controller provenance.
    # Use it for teacher-forced diagnostics only, preserving the training contract.
    diagnostic_contracts = {}
    if args.ckpt_dir == ARCHIVED_DIAGNOSTIC_BACKBONE.resolve() and not (args.closed_loop_episodes or args.drawer_selection_probe):
        bundle = read_json(args.ckpt_dir / "feedback_bundle.json")
        source = bundle["source"]
        require(source["config_sha256"] == report["checkpoint"]["config_sha256"] and
                source["norm_stats_sha256"] == report["checkpoint"]["norm_stats_sha256"],
                "Default diagnostic bundle source metadata mismatch")
        diagnostic_contracts = bundle["task_environment_contracts"]
    dataset_config = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    entries = {}
    for group in dataset_config["data_groups"].values():
        for key, entry in group.items():
            require(key not in entries, f"Duplicate dataset configuration key {key}")
            entries[key] = {**entry, "path": str(args.dataset_config.parent / Path(entry["path"]).expanduser())}
    report["dataset_config"] = {"path": str(args.dataset_config), "sha256": sha256(args.dataset_config)}
    # Split checking needs the caller's complete training sources; default is diagnostic.
    require(not args.require_heldout or args.training_dataset, "--require-heldout needs --training-dataset roots")
    training = defaultdict(lambda: {"seeds": set(), "hashes": set(), "trajectories": set()})
    declared_training_tasks = set()
    for root in args.training_dataset:
        import pandas as pd
        meta = read_json(root / "meta/dataset.json")
        training_rows = read_jsonl(root / "meta/episodes.jsonl")
        require(training_rows, f"Empty declared training dataset: {root}")
        declared_training_tasks.add(meta["task"])
        for row in training_rows:
            path = safe_child(root, row["data_path"])
            frame = pd.read_parquet(path, columns=["observation.state", "action", "timestamp"])
            training[meta["task"]]["seeds"].add(row["seed"])
            training[meta["task"]]["hashes"].add(sha256(path))
            training[meta["task"]]["trajectories"].add(trajectory_signature(
                np.stack(frame["observation.state"]), np.stack(frame["action"]), frame.timestamp.to_numpy()))
    training_sources_complete_for_tasks = all(t.instruction in declared_training_tasks for t in tasks)
    require(not args.require_heldout or training_sources_complete_for_tasks,
            "--require-heldout needs training-source declarations for every evaluated task")
    predictor = None
    scores = Scores()
    paired = defaultdict(list)
    rng = np.random.default_rng(args.seed)
    planned, processed, all_overlap = 0, 0, 0
    missing_memory = []
    stamps = {}
    variants = VARIANTS if frames > 1 and not args.no_memory_ablation else ("full",)
    report["variants"] = list(variants)
    # Preflight and allocate across the entire requested suite before inference.
    prepared = {}
    report["environment_compatibility"] = {}
    task_capacities = []
    for task in tasks:
        entry = entries[task.key]
        root = Path(entry["path"]).expanduser().resolve()
        meta, rows = dataset_manifest(root, task, entry)
        from mujoco_pickplace.episode_dataset import compare_environment_contract, validate_recorded_environment_contract
        recorded = validate_recorded_environment_contract(meta)
        checked = compare_environment_contract((config.get("task_environment_contracts") or {}).get(task.key), recorded)
        report["environment_compatibility"][task.key] = checked
        if (checked["status"] == "mismatch" and task.key in diagnostic_contracts
                and compare_environment_contract(diagnostic_contracts[task.key], recorded)["status"] == "verified"
                and set(checked["differences"]) <= {"controller_sha256", "controller_sources_sha256.task_env.py"}
                and not args.require_heldout):
            report["warnings"].append(
                f"{task.key}: original checkpoint controller provenance differs. "
                "Using the archive's separately qualified current controller for teacher-forced diagnosis only.")
        elif checked["status"] == "mismatch" and args.audit_only:
            report["warnings"].append(
                f"{task.key}: checkpoint/environment mismatch recorded. Data-only audit/replay "
                "continues without policy inference; this does not validate the checkpoint. "
                "Normalization bounds below are an incompatible-checkpoint reference only.")
        else:
            require(checked["status"] != "mismatch", f"Checkpoint/dataset environment mismatch: {checked}")
        selected = rows if args.episodes_per_task == 0 else [rows[int(i)] for i in
            sorted(rng.choice(len(rows), min(len(rows), args.episodes_per_task), replace=False))]
        require(args.closed_loop_episodes <= len(selected),
                f"Requested closed-loop episodes exceed selected episodes for {task.key}")
        episodes, candidates = [], []
        for row in selected:
            ep = load_episode(root, row, meta, task)
            episodes.append(ep)
            candidates.append(window_indices(ep.phases, ep.actions, ep.times, args.window_stride,
                row["quality"].get("drawer_selection_observation_sim_time")))
        prepared[task.key] = (episodes, candidates)
        task_capacities.append(sum(map(len, candidates)))
    task_budgets = balanced_budgets(task_capacities, args.max_queries)
    report["sampling"] = {"method": "task_episode_balanced_event_stratified",
                          "task_budgets": dict(zip([t.key for t in tasks], task_budgets))}
    alignment_flags = []
    replay_terminal_failures = []
    serving_checks = {}
    with (args.output / "windows.jsonl").open("x", encoding="utf-8") as records:
        for task_position, task in enumerate(tasks):
            require(task.key in entries, f"Dataset config missing {task.key}")
            entry = entries[task.key]
            root = Path(entry["path"]).expanduser().resolve()
            require(tuple(entry.get("active_action_mask", ())) == task.action_mask,
                    f"Action mask mismatch for {task.key}")
            meta, rows = dataset_manifest(root, task, entry)
            if config.get("memory_stride_seconds") is None and frames > 1:
                require(abs(float(meta["control_period_seconds"]) - .2) < 1e-6,
                        "Checkpoint uses step-based training history but evaluation uses 5-Hz reference steps; specify the actual checkpoint time contract before testing")
            for name in ("dataset.json", "episodes.jsonl", "tasks.jsonl"):
                path = root / "meta" / name
                stamps[path] = file_stamp(path)
            episodes, episode_candidates = prepared[task.key]
            selected = [ep.row for ep in episodes]
            capacity = [len(c) for c in episode_candidates]
            episode_budgets = (balanced_budgets(capacity, task_budgets[task_position])
                               if task_budgets[task_position] else [0]*len(capacity))
            data_report = {"task": task.key, "path": str(root), "total_episodes": len(rows),
                           "checkpoint_environment_compatible":report["environment_compatibility"][task.key]["status"] == "verified",
                           "selected_episodes": [r["episode_index"] for r in selected],
                           "manifest_sha256": sha256(root / "meta/episodes.jsonl"),
                           "metadata": meta, "episodes": []}
            report["datasets"].append(data_report)
            signatures = set()
            for episode_position, row in enumerate(selected):
                report["active_context"] = {"task": task.key, "episode": row["episode_index"]}
                episode = episodes[episode_position]
                require(episode.signature not in signatures, f"Duplicate trajectory in {task.key}")
                signatures.add(episode.signature)
                overlap = (row["seed"] in training[task.instruction]["seeds"] or
                           episode.hashes["parquet_sha256"] in training[task.instruction]["hashes"] or
                           episode.signature in training[task.instruction]["trajectories"])
                all_overlap += int(overlap)
                require(not args.require_heldout or not overlap,
                        f"Training/evaluation overlap: {task.key} episode {row['episode_index']}")
                candidates = episode_candidates[episode_position]
                planned += len(candidates)
                memory_evidence = memory_selection_evidence(episode, frames, stride)
                chosen = stratified_windows(candidates, episode.phases, episode.actions,
                    episode_budgets[episode_position], memory_evidence["frame"] if memory_evidence else None)
                processed += len(chosen)
                wanted = set()
                for index in chosen:
                    report["active_context"] = {"task": task.key, "episode": row["episode_index"], "frame": index}
                    indices, _ = history_indices(episode.times, index, frames, stride)
                    wanted.update(indices.tolist())
                images, video_hash = decode_video(episode.videos["front"], len(episode.states),
                    meta["fps"], meta["cameras"]["front"]["shape"], wanted if not args.audit_only else set())
                episode.hashes["front_video_sha256"] = video_hash
                bounds = stats[task.key]
                sl, sh = (np.asarray(bounds["observation.state"][n])[:8] for n in ("min", "max"))
                al, ah = (np.asarray(bounds["action"][n])[:7] for n in ("min", "max"))
                er = {"episode_index": row["episode_index"], "seed": row["seed"], **episode.hashes,
                      "trajectory_sha256": episode.signature, "training_overlap": overlap,
                      "candidate_windows": len(candidates), "selected_windows": len(chosen),
                      "selected_phase_windows": dict(Counter(episode.phases[chosen])),
                      "missing_execution_phases": sorted(set(episode.phases[candidates]) - set(episode.phases[chosen])),
                      "phase_frames": dict(Counter(episode.phases)),
                      "state_out_of_bounds_fraction_per_dim": np.mean((episode.states < sl-1e-7) | (episode.states > sh+1e-7), axis=0).tolist(),
                      "action_out_of_bounds_fraction_per_dim": np.mean((episode.actions < al-1e-7) | (episode.actions > ah+1e-7), axis=0).tolist()}
                data_report["episodes"].append(er)
                if task.number == 4:
                    er["target_drawer_annotation_only"] = recorded_drawer_target(row, task.number)
                er["drawer_memory_selection"] = memory_evidence
                er["temporal_alignment"] = transition_alignment(episode.states, episode.actions, episode.phases, meta["control_period_seconds"])
                if er["temporal_alignment"]["status"] == "suspected_temporal_misalignment":
                    alignment_flags.append({"task":task.key, "episode":row["episode_index"], **er["temporal_alignment"]})
                er["replay_alignment"] = {"status":"not_run","image_alignment_verified":False}
                if args.replay_steps:

                    er["replay_alignment"] = replay_alignment(
                        episode,task.number,args.replay_steps,
                        # Visibility qualification is pixel-dependent. Native
                        # contact replay must use the recorded camera size,
                        # rather than the low-resolution physics-only default.
                        image_size=int(meta['cameras']['front']['shape'][0]),
                        observe_terminal=task.number in (3,4))
                    if er["replay_alignment"]["status"] == "failed":
                        alignment_flags.append({"task":task.key,"episode":row["episode_index"],"replay":er["replay_alignment"]})
                    if er['replay_alignment'].get('terminal_acceptance', {}).get('status') == 'failed':
                        replay_terminal_failures.append({
                            'task': task.key, 'episode': row['episode_index'],
                            **er['replay_alignment']['terminal_acceptance']})
                if er["drawer_memory_selection"] and not er["drawer_memory_selection"]["memory_target_available"]:
                    missing_memory.append({"task": task.key, "episode": row["episode_index"]})
                if args.audit_only:
                    continue
                if chosen and predictor is None:
                    predictor = LocalPredictor(args.ckpt_dir, args.vlm_name)
                    for key in ("config_sha256", "norm_stats_sha256"):
                        require(predictor.identity[key] == report["checkpoint"][key], "Checkpoint changed since preflight")
                    report["checkpoint"].update(predictor.identity, weights_verified=True)
                    report["rgb_preprocessing_probe"] = predictor.check_rgb_preprocessing()
                for index in chosen:
                    report["active_context"] = {"task":task.key, "episode":row["episode_index"], "frame":index}
                    target = episode.actions[index:min(index+horizon, len(episode.actions))]
                    # Baselines expose low errors caused by static or redundant labels.
                    previous = episode.actions[index-1] if index else np.r_[np.zeros(6), 1.]
                    stationary = np.zeros_like(target)
                    stationary[:, 6] = previous[6]
                    for name, baseline in (("stationary", stationary),
                                           ("previous_action", np.tile(previous, (len(target), 1)))):
                        key = f"{task.key}/baseline_{name}"
                        scores.add(key + "/chunks", action_metrics(baseline, target, task.action_mask, al, ah))
                        scores.add(key + "/horizon/1", action_metrics(baseline[:1], target[:1], task.action_mask, al, ah))
                    seed_metrics = defaultdict(list)
                    first_actions = []
                    for seed in args.flow_seeds:
                        outputs = {}
                        for variant in variants:
                            payload, indices, valid = payload_for(episode, index, task, images, frames, stride, variant, seed)
                            start = time.perf_counter()
                            prediction = predictor(payload)
                            elapsed = time.perf_counter() - start
                            require(prediction.shape == (horizon, 24) and np.isfinite(prediction).all(),
                                    f"Invalid model output: {prediction.shape}, expected {(horizon, 24)}")
                            if variant == "full" and task.key not in serving_checks:
                                serving_checks[task.key] = predictor.check_serving_request(payload, prediction)
                            if report["prediction_count"] == 0:
                                repeat = predictor(payload)
                                require(repeat.shape == prediction.shape and np.isfinite(repeat).all(),
                                        "Invalid repeatability probe output")
                                delta = float(np.max(np.abs(repeat - prediction)))
                                report["same_seed_repeatability_max_abs_delta"] = delta
                                require(delta <= 1e-5, "Identical seeded inference is not reproducible; paired comparisons invalid")
                            if variant == "current_only" and "masked_history_probe" not in report:
                                report["masked_history_probe"] = verify_mask_invariance(predictor, payload, prediction)
                            prediction = prediction[:len(target), :7]
                            metrics = action_metrics(prediction, target, task.action_mask, al, ah,
                                                     continuous_gripper=task.number == 4)
                            key = f"{task.key}/{variant}"
                            scores.add(key + "/chunks", metrics)
                            scores.add(key + f"/episode/{row['episode_index']}", metrics)
                            scores.add(key + f"/current_phase/{episode.phases[index]}",
                                       action_metrics(prediction[:1], target[:1], task.action_mask, al, ah,
                                                      continuous_gripper=task.number == 4))
                            if er["drawer_memory_selection"] and er["drawer_memory_selection"]["frame"] == index:
                                scores.add(key + "/memory_selection", action_metrics(
                                    prediction[:1], target[:1], task.action_mask, al, ah,
                                    continuous_gripper=task.number == 4))
                            for offset in range(len(target)):
                                step = action_metrics(prediction[offset:offset+1], target[offset:offset+1], task.action_mask, al, ah,
                                                      continuous_gripper=task.number == 4)
                                scores.add(key + f"/horizon/{offset+1}", step)
                                scores.add(key + f"/target_phase/{episode.phases[index+offset]}", step)
                            row_out = {"task": task.key, "episode": row["episode_index"], "frame": index,
                                       "phase": str(episode.phases[index]), "variant": variant, "flow_seed": seed,
                                       "history_indices": indices.tolist(), "history_mask": valid.tolist(),
                                       "valid_future_steps": len(target), "inference_seconds": elapsed,
                                       "prediction": prediction.tolist(), "target": target.tolist(), "metrics": metrics}
                            records.write(json.dumps(row_out, allow_nan=False) + "\n")
                            report["prediction_count"] += 1
                            outputs[variant] = (metrics, prediction)
                            seed_metrics[variant].append(metrics["first_xyz_error_mm"])
                        first_actions.append(outputs["full"][1][0])
                        for variant in variants[1:]:
                            paired[f"{task.key}/{variant}/first_xyz_error_increase_mm"].append(
                                outputs[variant][0]["first_xyz_error_mm"] - outputs["full"][0]["first_xyz_error_mm"])
                            paired[f"{task.key}/{variant}/first_prediction_change_mm"].append(float(
                                np.linalg.norm(outputs[variant][1][0, :3] - outputs["full"][1][0, :3]) * 1000))
                    for variant, values in seed_metrics.items():
                        paired[f"{task.key}/{variant}/per_window_seed_mean_first_xyz_error_mm"].append(float(np.mean(values)))
                        paired[f"{task.key}/{variant}/per_window_seed_worst_first_xyz_error_mm"].append(float(np.max(values)))
                    if len(first_actions) > 1:
                        paired[f"{task.key}/flow_noise_first_xyz_std_mm"].append(float(
                            np.linalg.norm(np.std(np.asarray(first_actions)[:, :3], axis=0)) * 1000))
                    report["query_count"] += 1
                    records.flush()
                    if report["query_count"] == 1 or report["query_count"] % 20 == 0:
                        print(f"Scored {report['query_count']} windows ({report['prediction_count']} predictions)", flush=True)
                print(f"Validated {task.key} episode {row['episode_index']}: {len(chosen)}/{len(candidates)} windows", flush=True)
    for path, stamp in stamps.items():
        require(file_stamp(path) == stamp, f"Dataset metadata changed during evaluation: {path}")
    report.update(planned_windows=planned, selected_windows=processed,
                  coverage_complete=(processed == planned and args.window_stride == 1
                                     and all(len(d["selected_episodes"]) == d["total_episodes"] for d in report["datasets"])),
                  training_overlap_episodes=all_overlap,
                  generalization=("overlapping_training_sources" if all_overlap else
                                  "selected_episodes_disjoint_from_declared_training_sources"
                                  if training_sources_complete_for_tasks else "unverified"),
                  scores=scores.report(), macro_scores=scores.macro_report(),
                  missing_certified_memory=missing_memory,
                  suspected_temporal_misalignment=alignment_flags, serving_request_checks=serving_checks,
                  replay_terminal_failures=replay_terminal_failures,
                  paired_memory_and_seed_metrics={k: distribution(v) for k, v in paired.items()})
    if missing_memory:
        report["warnings"].append("Some Task4 selection windows lack certified target history under this checkpoint's memory contract.")
    if args.audit_only:
        report["status"] = "data_audit_complete_model_not_tested"
    else:
        require(report["query_count"] > 0, "No model queries were evaluated")
        report["status"] = "measured_not_certified"
    # Thresholds are user-specified in physical units, never silently invented.
    if args.limits:
        require(not args.audit_only, "Quality limits require actual model predictions")
        results = evaluate_limits(report["scores"], read_json(args.limits))
        report["limits"] = results
        report["acceptance_evidence_gaps"] = acceptance_evidence_gaps(report)
        if not all(r["passed"] for r in results):
            report["status"] = "configured_limits_failed"
        elif report["acceptance_evidence_gaps"]:
            report["status"] = "insufficient_acceptance_evidence"
        else:
            report["status"] = "configured_limits_passed"
    if alignment_flags or replay_terminal_failures or any(check["status"] == "failed" for check in serving_checks.values()):
        report["status"] = "diagnostic_checks_failed"
    if args.closed_loop_episodes:
        require(predictor is not None, "Closed-loop probing requires a loaded model")

        seeds = {task.number: list(dict.fromkeys(ep.row["seed"] for ep in prepared[task.key][0]))[:args.closed_loop_episodes]
                 for task in tasks}
        report["closed_loop_probe"] = run_closed_loop_suite(
            predictor, seeds, frames, stride, args.output / "closed_loop", args.closed_loop_max_steps)
    if args.drawer_selection_probe:
        require(predictor is not None, "Selection probing requires a loaded model")
        report["drawer_selection_probe"] = run_drawer_selection_probe(
            predictor, frames, stride, args.output/"drawer_selection", args.seed,
            args.closed_loop_max_steps or 40)
        report["additional_selection_policy_requests"] = report["drawer_selection_probe"].get("policy_requests",0)
    report["closed_loop_comparison"] = (compare_closed_loop(report, args.closed_loop_summary)
                                         if args.closed_loop_summary else {"status": "not_verified"})
    report["acceptance_scope"] = {
        "configured_limits": "Only specified teacher-forced action metrics; not physical task certification",
        "predicted_closed_loop_success_rate": None,
        "image_action_alignment_verified": False,
        "recovery_and_off_expert_state_coverage_certified": False,
        "memory_ablation_proves_correct_target_selection": False,
        "closed_loop_evidence": ("observed_feedback_rollouts" if args.closed_loop_episodes
                                  else "bounded_drawer_selection_probe" if args.drawer_selection_probe else "not_run"),
    }
    report["active_context"] = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_BACKBONE, help="Checkpoint backbone to test")
    parser.add_argument("--output", type=Path, default=ROOT / "mujoco_pickplace/outputs" / datetime.now(timezone.utc).strftime("open_loop_step_final_%Y%m%d_%H%M%S_%f"), help="NEW report directory; never overwritten")
    parser.add_argument("--tasks", type=int, nargs="+", choices=(1, 3, 4), default=[1, 3, 4])
    parser.add_argument("--dataset-config", type=Path, default=EVO / "dataset/config.yaml",
                        help="Evaluation dataset paths/contracts, not proof of checkpoint training provenance")
    parser.add_argument("--training-dataset", type=Path, action="append", default=[],
                        help="Declare every training dataset root for seed/file-overlap checks; repeatable")
    parser.add_argument("--require-heldout", action="store_true", help="Fail on detected overlap; requires declared training roots")
    parser.add_argument("--audit-only", action="store_true", help="Validate selected episodes and full video streams without loading model")
    parser.add_argument("--episodes-per-task", type=int, default=0, help="0=all; otherwise seeded random episode subset")
    parser.add_argument("--window-stride", type=int, default=1, help="1=exhaustive; larger also includes phase/gripper boundaries")
    parser.add_argument("--max-queries", type=int, default=0, help="0=all; positive caps observation windows for smoke tests (reported as partial)")
    parser.add_argument("--seed", type=int, default=None, help="Episode subset RNG; random by default and recorded")
    parser.add_argument("--flow-seeds", type=int, nargs="+", default=None, help="Three random seeds by default, all scored and recorded")
    parser.add_argument("--no-memory-ablation", action="store_true", help="Skip current-only, repeated-current-images and reversed-past-images checks")
    parser.add_argument("--vlm-name", default=None, help="Optional local VLM path, as in Evo1_server.py")
    parser.add_argument("--limits", type=Path, help='JSON list: {"group":"mujoco_drawer_memory/full/horizon/1", "metric":"first_xyz_error_mm", "statistic":"p95", "maximum":3}')
    parser.add_argument("--closed-loop-summary", type=Path, action="append", default=[],
                        help="Existing eval summary.json (list or episodes object), repeatable; exact weights/config/stats required")
    parser.add_argument("--closed-loop-episodes", type=int, default=0,
                        help="Run this many real feedback episodes per task on selected seeds; 0 disables simulation")
    parser.add_argument("--closed-loop-max-steps", type=int, default=None,
                        help="Optional smoke-test cap. Shortened unresolved runs are censored, never counted as failures")
    parser.add_argument("--drawer-selection-probe", action="store_true",
                        help="Opt-in four-target full/current-only contact-selection probe; 8 short cases, default 40 steps each, not full-task evaluation")
    parser.add_argument("--replay-steps", type=int, default=0,
                        help="Replay this many expert execution actions per selected episode to check state/action alignment; 0 disables physics replay")
    args = parser.parse_args(argv)
    require(len(set(args.tasks)) == len(args.tasks), "Duplicate task selections")
    require(args.episodes_per_task >= 0 and args.max_queries >= 0 and args.window_stride >= 1, "Invalid sample limits")

    require(args.replay_steps >= 0 and args.closed_loop_episodes >= 0 and (args.closed_loop_max_steps is None or args.closed_loop_max_steps > 0),
            "Invalid closed-loop budget")
    require(not args.drawer_selection_probe or 4 in args.tasks, "Drawer selection probe requires Task4")
    require(args.closed_loop_max_steps is None or args.closed_loop_episodes > 0 or args.drawer_selection_probe,
            "--closed-loop-max-steps requires --closed-loop-episodes or --drawer-selection-probe")
    require(not args.audit_only or not (args.closed_loop_summary or args.limits or args.closed_loop_episodes or args.drawer_selection_probe),
            "--audit-only cannot assess model limits or closed-loop agreement")
    args.seed = secrets.randbits(32) if args.seed is None else args.seed
    require(0 <= args.seed < 2**63, "Invalid subset seed")
    if args.flow_seeds is None:
        args.flow_seeds = np.random.default_rng(args.seed).choice(2**32, 3, replace=False).tolist()
    require(args.flow_seeds and len(set(args.flow_seeds)) == len(args.flow_seeds)
            and all(0 <= seed < 2**63 for seed in args.flow_seeds), "Invalid/duplicate flow seeds")
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.dataset_config = args.dataset_config.expanduser().resolve()
    args.training_dataset = [p.expanduser().resolve() for p in args.training_dataset]
    return args


def print_result_summary(report):
    """Expose measured results at the terminal, not just a successful exit code."""
    print("Coverage: " + ("complete dataset windows" if report.get("coverage_complete")
                          else "partial or unverified; not a full acceptance test"))
    for dataset in report.get("datasets", []):
        task = dataset["task"]
        score = report.get("scores", {}).get(f"{task}/full/horizon/1")
        if score:
            xyz = score["metrics"]["first_xyz_error_mm"]
            counts = score["counts"]
            print(f"{task}: first-action XYZ mean={xyz['mean']:.3f} mm, p95={xyz['p95']:.3f} mm; "
                  f"false-open={counts.get('false_open_count', 0)}, "
                  f"opposite-direction={counts.get('opposite_direction_count', 0)}, "
                  f"stall={counts.get('stalled_count', 0)} (prediction counts, not episode rates)")
        else:
            print(f"{task}: model not tested on this task")
        missing = sorted({phase for e in dataset["episodes"] for phase in e.get("missing_execution_phases", [])})
        if missing:
            print(f"  Phases missing in one or more selected episodes: {', '.join(missing)}")
    for task, outcome in report.get("closed_loop_probe", {}).get("tasks", {}).items():
        print(f"Task{task} real feedback: resolved={outcome['resolved_episodes']}, "
              f"censored={outcome['censored_episodes']}, "
              f"observed environment success rate={outcome['environment_success_rate']}")
    print("No closed-loop success rate is inferred from open-loop action error.")


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), f"Output already exists (will not overwrite): {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mujoco-evo-open-loop-v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {k: str(v) if isinstance(v, Path) else
                            [str(x) if isinstance(x, Path) else x for x in v] if isinstance(v, list) else v
                            for k, v in vars(args).items()},
              "metric_notes": [
                  "Teacher-forced: every query uses expert observations, never predicted observations.",
                  "No augmentation, no action clipping, no terminal repeat padding in metrics.",
                  "XYZ units are mm; rotation-vector increments use radians; gripper is continuous [0,1].",
                  "delta_sum_error_mm sums commanded deltas; it is not an achieved end-effector trajectory.",
                  "Chunk/phase statistics include overlapping windows; horizon/1 is the first-action view.",
                  "Future target_phase pools overlapping predictions; current_phase uses first actions only.",
                  "Macro scores weight episodes/phases equally. Count metrics pool predictions, not independent trials.",
                  "Positive paired error increase means full history improved that metric; no significance claim.",
                  "current_images retains state history; current_only masks all past modalities.",
                  "Unknown training membership remains unverified. Supplied training roots must be complete.",
              ]}
    code = 0
    try:
        run(args, report)
        code = 1 if report["status"] in {"configured_limits_failed", "insufficient_acceptance_evidence", "diagnostic_checks_failed"} else 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        print(f"Open-loop test failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        code = 2
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output / 'report.json'}", flush=True)
    print_result_summary(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
