from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import imageio.v2 as imageio
import numpy as np
import pandas as pd

FORMAT_NAME = "mujoco-evo-episodes"
FORMAT_VERSION = "1.0"
SOURCE_POLICY_VERSION = "precision-grasp-stable-v3"
DEFAULT_TASK_DESCRIPTION = "pick up the blue cube and place it on the green target"
DEFAULT_SOURCE_POLICY = "stateful contact-aware precision-grasp scripted expert"
STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_axis_angle_x",
    "eef_axis_angle_y",
    "eef_axis_angle_z",
    "left_finger_qpos",
    "right_finger_qpos",
]
ACTION_NAMES = [
    "eef_dx",
    "eef_dy",
    "eef_dz",
    "eef_droll",
    "eef_dpitch",
    "eef_dyaw",
    "gripper",
]
CAMERAS = ("front",)
CONTACT_QUALITY_SCHEMA = "mujoco-panda-contact"
CONTACT_QUALITY_VERSION = 3

# Memory observations must not acquire codec artifacts from later expert actions.
CAUSAL_VIDEO_ENCODING = dict(
    codec="libx264rgb", pixel_format="rgb24", lossless=True, intra_only=True,
)


def write_rgb_video(path, frames, fps):
    """Store exact RGB observations; no temporal prediction or chroma loss."""
    imageio.mimsave(
        path, np.asarray(frames, dtype=np.uint8), fps=fps, macro_block_size=1,
        codec=CAUSAL_VIDEO_ENCODING["codec"],
        pixelformat=CAUSAL_VIDEO_ENCODING["pixel_format"], quality=None,
        output_params=["-crf", "0", "-preset", "veryfast", "-g", "1", "-bf", "0"],
    )


def validate_rgb_video(path, length, image_size, fps):
    """Check the actual video stream, not only its file name or metadata claim."""
    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
        if (tuple(metadata.get("size", ())) != (image_size, image_size)
                or not np.isclose(float(metadata.get("fps", 0)), fps)):
            raise ValueError(f"Video resolution/frame rate disagrees with its episode: {path}")
        if not str(metadata.get("pix_fmt", "")).startswith("gbrp"):
            raise ValueError(f"Video is not the declared lossless RGB stream: {path}")
        count = reader.count_frames()
        if count != length:
            raise ValueError(f"Video has {count} frames, expected {length}: {path}")
        for index in {0, length - 1}:
            pixels = reader.get_data(index)
            if pixels.shape != (image_size, image_size, 3) or pixels.dtype != np.uint8:
                raise ValueError(f"Video frame {index} is not the declared uint8 RGB image: {path}")


def validate_video_encoding(metadata):
    if (metadata.get("format") == FORMAT_NAME
            and metadata.get("video_encoding") != CAUSAL_VIDEO_ENCODING):
        raise ValueError(
            "Dataset uses legacy temporal/lossy video encoding: decoded observations "
            "can depend on future expert frames. Recollect this task from raw simulator RGB "
            "with the current collector in a new dataset or using --overwrite. "
            "Transcoding existing lossy MP4 files cannot recover causal raw observations."
        )



def _environment_contract_error(contract):
    """Validate recorded structure without inventing provenance for legacy data."""
    required = {"schema", "version", "task_id", "control_period_seconds",
                "action_convention", "state_convention", "collection_config",
                "assets_sha256", "controller_sha256", "quality_schema"}
    if not isinstance(contract, dict) or not required.issubset(contract):
        return "Environment contract is incomplete"
    if contract["schema"] != "mujoco-task-environment" or contract["version"] not in (1, 2):
        return "Unsupported environment contract schema/version"
    if (not isinstance(contract["assets_sha256"], dict) or not contract["assets_sha256"]
            or not isinstance(contract["controller_sha256"], str) or len(contract["controller_sha256"]) != 64):
        return "Environment contract lacks asset/controller fingerprints"
    engine = contract.get("physics_engine")
    if contract["version"] >= 2 or engine is not None:
        if (not isinstance(engine, dict) or engine.get("name") != "mujoco"
                or any(not isinstance(engine.get(key), str) or not engine[key].strip()
                       for key in ("python_package_version", "runtime_version"))):
            return "Environment contract lacks a complete physics engine identity"
        if engine["python_package_version"] != engine["runtime_version"]:
            return "MuJoCo Python package and loaded physics engine versions disagree"
    return None


def compare_environment_contract(trained, current):
    """Compare known fields first; legacy engine provenance remains unverified."""
    for label, contract in (("Checkpoint training data", trained), ("Evaluation environment/data", current)):
        error = _environment_contract_error(contract)
        if error is not None:
            return dict(status="unverified", reason=f"{label}: {error}")
    differences = {}

    def compare(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                compare(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key)
        elif left != right:
            differences[prefix] = dict(trained=left, current=right)

    # v2 adds engine provenance to the v1 fields. Missing legacy provenance is
    # unknown, but must not hide a known scene/control incompatibility.
    compare({k: v for k, v in trained.items() if k not in {"version", "physics_engine"}},
            {k: v for k, v in current.items() if k not in {"version", "physics_engine"}})
    if trained.get("physics_engine") is not None and current.get("physics_engine") is not None:
        compare(trained["physics_engine"], current["physics_engine"], "physics_engine")
    if differences:
        return dict(status="mismatch", differences=differences)
    if trained.get("physics_engine") is None or current.get("physics_engine") is None:
        return dict(status="unverified", differences={},
                    reason="Legacy environment contract has no recorded MuJoCo version; compatibility is unverified")
    return dict(status="verified", differences={})


def validate_recorded_environment_contract(metadata):
    contract = metadata.get("environment_contract")
    if contract is None:
        return None
    error = _environment_contract_error(contract)
    if error is not None:
        raise ValueError(f"Dataset environment contract is incomplete or unsupported: {error}")
    if (contract["collection_config"] != metadata.get("collection_config", {})
            or not np.isclose(contract["control_period_seconds"], metadata["control_period_seconds"], rtol=0, atol=1e-9)):
        raise ValueError("Dataset environment contract disagrees with collection configuration/control period")
    if "fps" in metadata and not np.isclose(1.0 / metadata["fps"], contract["control_period_seconds"], rtol=0, atol=1e-9):
        raise ValueError("Dataset environment control period disagrees with video FPS")
    return contract


def contact_source_policy_version(task_number):
    if task_number not in (3, 4):
        raise ValueError("Native contact expert requires Task3 or Task4")
    return f"task{task_number}-contact-v{CONTACT_QUALITY_VERSION}"


MAX_SUCCESSFUL_CLOSE_XY_ERROR = 0.006
MIN_SUCCESSFUL_CLOSE_Z_ABOVE = -0.012
MAX_SUCCESSFUL_CLOSE_Z_ABOVE = 0.001
MAX_PREGRASP_CUBE_DISPLACEMENT = 0.004
MAX_PREGRASP_CUBE_TILT_DEG = 3.0
MAX_ATTACH_CUBE_TILT_DEG = 3.0
MAX_HELD_CUBE_TILT_DEG = 8.0
MAX_ATTACH_CUBE_ANGULAR_SPEED = 0.15


def validate_precision_grasp_quality(quality, episode_index=0, strict=True):
    """Validate Task1 with the existing precision and contact thresholds."""
    required_quality = {
        "successful_close_xy_error",
        "successful_close_z_above",
        "initial_cube_xy",
        "initial_goal_xy",
        "randomize_task",
        "randomization_scale",
    }
    missing_quality = sorted(required_quality - set(quality))
    if missing_quality:
        raise AssertionError(
            f"Episode {episode_index} is missing precision quality fields: {missing_quality}"
        )
    if quality["randomize_task"] is not True:
        raise AssertionError(
            f"Episode {episode_index} was not collected with randomization"
        )
    if not np.isclose(float(quality["randomization_scale"]), 1.0):
        raise AssertionError(
            f"Episode {episode_index} did not use randomization_scale=1.0"
        )
    if (
        float(quality["successful_close_xy_error"])
        > MAX_SUCCESSFUL_CLOSE_XY_ERROR + 1e-08
    ):
        raise AssertionError(f"Episode {episode_index} closed too far from cube XY")
    if (
        float(quality["successful_close_z_above"])
        > MAX_SUCCESSFUL_CLOSE_Z_ABOVE + 1e-08
    ):
        raise AssertionError(
            f"Episode {episode_index} closed above the precision grasp plane"
        )
    if (
        float(quality["successful_close_z_above"])
        < MIN_SUCCESSFUL_CLOSE_Z_ABOVE - 1e-08
    ):
        raise AssertionError(
            f"Episode {episode_index} closed below the precision grasp band"
        )
    if strict:
        strict_quality = {
            "quality_schema_version",
            "first_attachment_has_two_pad_contact",
            "two_pad_contact_steps",
            "attachment_lost_before_release",
            "attachment_xy_error",
            "attachment_z_above",
            "attachment_cube_tilt_deg",
            "attachment_cube_angular_speed",
            "pregrasp_cube_displacement",
            "pregrasp_cube_tilt_deg",
            "held_cube_tilt_deg",
        }
        missing_strict = sorted(strict_quality - set(quality))
        if missing_strict:
            raise AssertionError(
                f"Episode {episode_index} is missing strict grasp fields: {missing_strict}"
            )
        if int(quality["quality_schema_version"]) < 3:
            raise AssertionError(
                f"Episode {episode_index} uses an obsolete quality schema"
            )
        strict_checks = {
            "first_attachment_has_two_pad_contact": bool(
                quality["first_attachment_has_two_pad_contact"]
            ),
            "two_pad_contact_steps": int(quality["two_pad_contact_steps"]) >= 2,
            "attachment_retained": not bool(quality["attachment_lost_before_release"]),
            "attachment_xy_error": float(quality["attachment_xy_error"])
            <= MAX_SUCCESSFUL_CLOSE_XY_ERROR + 1e-08,
            "attachment_z_lower": float(quality["attachment_z_above"])
            >= MIN_SUCCESSFUL_CLOSE_Z_ABOVE - 1e-08,
            "attachment_z_upper": float(quality["attachment_z_above"])
            <= MAX_SUCCESSFUL_CLOSE_Z_ABOVE + 1e-08,
            "pregrasp_cube_displacement": float(quality["pregrasp_cube_displacement"])
            <= MAX_PREGRASP_CUBE_DISPLACEMENT + 1e-08,
            "pregrasp_cube_tilt": float(quality["pregrasp_cube_tilt_deg"])
            <= MAX_PREGRASP_CUBE_TILT_DEG + 1e-08,
            "attachment_cube_tilt": float(quality["attachment_cube_tilt_deg"])
            <= MAX_ATTACH_CUBE_TILT_DEG + 1e-08,
            "held_cube_tilt": float(quality["held_cube_tilt_deg"])
            <= MAX_HELD_CUBE_TILT_DEG + 1e-08,
            "attachment_cube_angular_speed": float(
                quality["attachment_cube_angular_speed"]
            )
            <= MAX_ATTACH_CUBE_ANGULAR_SPEED + 1e-08,
        }
        failed_strict = [name for (name, passed) in strict_checks.items() if not passed]
        if failed_strict:
            raise AssertionError(
                f"Episode {episode_index} failed strict grasp checks: {failed_strict}"
            )


def validate_contact_quality(quality, task_number):
    """Require recorded physical trajectory evidence, beyond the final success flag."""
    if quality.get("correction_source") is not None:
        validate_correction_context(None, quality, task_number)
    if task_number not in (3, 4):
        raise AssertionError("Native contact schema requires Task3 or Task4")
    required = {
        "complete_motion",
        "two_finger_grasp",
        "object_lifted",
        "transport_retained",
        "no_robot_table_contact",
        "no_unexpected_robot_contact",
        "no_execution_external_force",
        "no_object_gravity_assistance",
        "robot_actuators_only",
        "stable_placement",
        "confirmed_release_and_support",
        "continuous_contact_monitoring",
        "physics_substep_placement_monitoring",
        "contact_gap_descent_monitoring",
    }
    if task_number == 4:
        required.update(
            {
                "horizontal_handle_approach",
                "handle_contact_through_pull",
                "no_wrong_drawer_opened",
                "correct_first_drawer",
                "presentation_complete",
                "presentation_robot_clear",
                "selection_within_history",
            }
        )
    if (task_number == 4 and (quality.get("correction_source") or {}).get("kind") == "object_empty_grasp"):
        history_checks = {"horizontal_handle_approach", "handle_contact_through_pull"}
        recorded = quality.get("policy_context_checks", {})
        if set(recorded) != history_checks or any(type(value) is not bool for value in recorded.values()):
            raise AssertionError("Object continuation must retain its excluded handle-stage diagnostics")
        required -= history_checks
    if (
        quality.get("schema") != CONTACT_QUALITY_SCHEMA
        or quality.get("schema_version") != CONTACT_QUALITY_VERSION
    ):
        raise AssertionError("Missing or unsupported native contact quality schema")
    if quality.get("task_id") != task_number or quality.get("success") is not True:
        raise AssertionError("Wrong task or unsuccessful native contact trajectory")
    checks = quality.get("checks", {})
    failed = sorted(name for name in required if checks.get(name) is not True)
    failed.extend(name for name, passed in checks.items() if passed is not True)
    if failed or quality.get("failed_checks") != []:
        raise AssertionError(
            f"Task{task_number} failed contact quality checks: {sorted(set(failed))}"
        )
    if (
        quality.get("motion_failures") != []
        or quality.get("transport_drop") is not False
    ):
        raise AssertionError("Trajectory has a motion failure or transport drop")
    if quality.get("timestamps_preserve_control_time") is not True:
        raise AssertionError("Trajectory does not preserve simulation timestamps")
    if task_number == 4 and not quality.get(
        "target_visible_history_slots_at_selection"
    ):
        raise AssertionError("Target is absent from the sampled decision history")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode("utf-8"))


def _atomic_jsonl(path, rows):
    _atomic_bytes(path, "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def feature_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def merge_feature_stats(rows, key):
    entries = [row["stats"][key] for row in rows]
    count = sum(entry["count"] for entry in entries)
    means = np.asarray([entry["mean"] for entry in entries], dtype=np.float64)
    stds = np.asarray([entry["std"] for entry in entries], dtype=np.float64)
    counts = np.asarray([entry["count"] for entry in entries], dtype=np.float64)
    mean = np.sum(means * counts[:, None], axis=0) / count
    second = np.sum((stds**2 + means**2) * counts[:, None], axis=0) / count
    return {
        "count": int(count),
        "min": np.min([entry["min"] for entry in entries], axis=0).tolist(),
        "max": np.max([entry["max"] for entry in entries], axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - mean**2, 0.0)).tolist(),
    }


def validate_correction_context(phases, quality, task_number, timestamps=None):
    """Only a certified, contiguous policy prefix can be history-only data."""
    source = quality.get("correction_source")
    if task_number == 4 and source is not None:
        return _validate_handle_correction_context(phases, quality, timestamps)
    if source is not None and (
        task_number != 3 or not isinstance(source, dict)
        or not isinstance(source.get("recorded_empty_grasp_sim_time"), (int, float))
        or not np.isfinite(source["recorded_empty_grasp_sim_time"])
        or not isinstance(source.get("recorded_response_seconds"), (int, float))
        or not np.isfinite(source["recorded_response_seconds"])
        or source["recorded_response_seconds"] < 1.0 - 1e-5
    ):
        raise ValueError("Correction history must include the recorded response after the empty grasp")
    if phases is None:
        return
    prefix = [i for i, phase in enumerate(phases) if phase == "policy_context"]
    if not prefix and source is None:
        return
    if (task_number != 3 or not prefix or prefix != list(range(len(prefix)))
            or len(prefix) >= len(phases) or phases[len(prefix)] != "recover"
            or not isinstance(source, dict)
            or source.get("history_only_frames") != len(prefix)
            or source.get("safe_empty_grasp_reproduced") is not True
            or len(str(source.get("sha256", ""))) != 64
            or quality.get("natural_corrections", 0) < 1):
        raise ValueError("Invalid recorded-policy correction context")
    if timestamps is not None and source is not None:
        elapsed = timestamps[len(prefix)] - source["recorded_empty_grasp_sim_time"]
        if not np.isclose(elapsed, source["recorded_response_seconds"], atol=1e-4, rtol=0):
            raise ValueError("Correction response duration disagrees with physical timestamps")


def _validate_handle_correction_context(phases, quality, timestamps):
    source = quality["correction_source"]
    if not isinstance(source, dict):
        raise ValueError("Invalid handle correction source")
    policy_state = source.get("kind") == "handle_policy_state"
    object_failure = source.get("kind") == "object_empty_grasp"
    if object_failure:
        _validate_object_failure_source(source, quality)
    elif policy_state:
        intervention_time = source.get("recorded_intervention_sim_time")
        replay_error = source.get("recorded_replay_max_state_error")
        opening = source.get("recorded_max_drawer_open_m")
        counts = [source.get(name) for name in (
            "intervention_step", "recorded_prefix_physics_substeps",
            "recorded_physics_substeps_per_control")]
        values = [intervention_time, replay_error, opening]
        if (source.get("safe_handle_deviation_reproduced") is not False
                or source.get("policy_state_prefix_verified") is not True
                or source.get("confirmed_failure_claim") is not False
                or "recorded_first_selected_drawer" not in source
                or source["recorded_first_selected_drawer"] is not None
                or source.get("recorded_bilateral_ever") is not False
                or any(source.get(name) is True for name in (
                    "safe_handle_contact_loss_reproduced", "recorded_bilateral_before_loss",
                    "recorded_bilateral_at_intervention", "recorded_closed_at_intervention",
                    "safe_empty_grasp_reproduced"))
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) for v in values)
                or intervention_time < 0
                or not 0 <= replay_error <= 1e-4 or not 0 <= opening <= .005
                or any(type(v) is not int or v <= 0 for v in counts)
                or counts[0] < 2 or counts[1] != counts[0] * counts[2]
                or len(str(source.get("sha256", ""))) != 64
                or not source.get("run_id")
                or type(quality.get("policy_state_continuations")) is not int
                or quality["policy_state_continuations"] < 1):
            raise ValueError("Uncertified recorded handle policy state")
    else:
        _validate_handle_failure_source(source, quality)
    if phases is None:
        return
    start = 0
    while start < len(phases) and phases[start] in {"observe_target", "presentation_close"}:
        start += 1
    prefix = [i for i, phase in enumerate(phases) if phase == "policy_context"]
    end = start + len(prefix)
    if (start == 0 or not prefix or prefix != list(range(start, end))
            or len(prefix) != source["intervention_step"]
            or end >= len(phases) or phases[end] != "recover"
            or source.get("history_only_frames") != end):
        raise ValueError("Handle correction must follow presentation and a contiguous policy prefix")
    time_field = "recorded_intervention_sim_time" if policy_state or object_failure else "recorded_deviation_sim_time"
    if timestamps is not None and not np.isclose(
        timestamps[end], source[time_field], atol=1e-4, rtol=0
    ):
        raise ValueError("Handle intervention time disagrees with physical observations")


def _validate_object_failure_source(source, quality):
    values = [source.get(name) for name in (
        "recorded_intervention_sim_time", "recorded_replay_max_state_error",
        "recorded_drawer_open_m", "recorded_aperture_m", "recorded_gripper_command")]
    if (source.get("safe_object_empty_grasp_reproduced") is not True
            or source.get("recorded_object_contact") is not False
            or type(source.get("recorded_selected_drawer")) is not int
            or source["recorded_selected_drawer"] != quality.get("target_drawer_diagnostic_only")
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) for v in values)
            or values[0] < 0 or not 0 <= values[1] <= 1e-4
            or values[2] < .165 or not -.001 <= values[3] < .006
            or not 0 <= values[4] <= .2
            or type(source.get("intervention_step")) is not int
            or source["intervention_step"] < 2
            or len(str(source.get("sha256", ""))) != 64
            or not source.get("run_id") or quality.get("natural_corrections", 0) < 1):
        raise ValueError("Uncertified recorded drawer object empty grasp")


def _validate_handle_failure_source(source, quality):
    """Preserve the two existing failure certificates independently of policy states."""
    numbers = [source.get(name) for name in (
        "recorded_handle_deviation_m", "recorded_deviation_sim_time",
        "recorded_replay_max_state_error")]
    numbers.append(source.get("recorded_handle_orientation_error_rad", 0.0))
    contact_loss = source.get("kind") == "handle_contact_loss"
    if (source.get("kind") not in {"handle_approach_deviation", "handle_contact_loss"}
            or source.get("safe_handle_deviation_reproduced") is not True
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                   for v in numbers)
            or numbers[0] < 0 or not 0 <= numbers[3] <= np.pi + 1e-6
            or (not contact_loss and numbers[0] < 0.005 and numbers[3] < np.deg2rad(2.0))
            or not 0 <= numbers[2] <= 1e-4
            or type(source.get("intervention_step")) is not int
            or source["intervention_step"] < 2
            or len(str(source.get("sha256", ""))) != 64
            or not source.get("run_id") or quality.get("natural_corrections", 0) < 1):
        raise ValueError("Uncertified recorded handle correction")
    if contact_loss:
        opening = source.get("recorded_max_drawer_open_m")
        selected = source.get("recorded_first_selected_drawer")
        target = quality.get("target_drawer_diagnostic_only")
        if (source.get("safe_handle_contact_loss_reproduced") is not True
                or source.get("recorded_bilateral_before_loss") is not True
                or source.get("recorded_bilateral_at_intervention") is not False
                or source.get("recorded_closed_at_intervention") is not True
                or type(selected) is not int or selected not in range(4)
                or type(target) is not int or target not in range(4) or selected != target
                or isinstance(opening, bool) or not isinstance(opening, (int, float))
                or not np.isfinite(opening) or not 0 <= opening <= .005):
            raise ValueError("Uncertified loss of a correctly selected, unopened handle")


def validate_collection_snapshot(metadata):
    """Saved episodes are not a finalized training snapshot during collection.

    Older datasets without an attempt report remain explicitly unreported.
    An interrupted/attempt-limited run may contain a valid finalized subset;
    a running attempt must never be silently called a completed collection.
    """
    report = metadata.get("last_collection_report")
    if report is None:
        return {"status": "unreported"}
    if report.get("status") == "quarantined":
        raise ValueError(
            "Quarantined collection snapshot is not eligible for checking/training: "
            + str(report.get("reason", "physical qualification failed"))
        )
    if report.get("status") == "running" or any(
        row.get("status") == "running" for row in report.get("attempt_results", [])
    ):
        raise ValueError(
            "Unfinished collection snapshot: last_collection_report is running "
            f"(saved={report.get('saved')}, rejected={report.get('rejected')}, "
            f"attempts={report.get('attempts')}). Saved episodes do not prove "
            "that collection completed; finalize collection before checking/training."
        )
    if report.get("status") not in {"completed", "attempt_limit", "interrupted", "error"}:
        raise ValueError(f"Unknown collection snapshot status: {report.get('status')!r}")
    return {key: report.get(key) for key in ("run_id", "status", "saved", "rejected", "attempts")}


class EpisodeDatasetWriter:

    def __init__(
        self,
        root,
        fps=5.0,
        chunk_size=1000,
        image_size=448,
        collection_config=None,
        task_description=DEFAULT_TASK_DESCRIPTION,
        source_policy=DEFAULT_SOURCE_POLICY,
        source_policy_version=SOURCE_POLICY_VERSION,
        robot="handwritten-panda7",
        environment_contract=None,
    ):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        if self.chunk_size < 1 or self.image_size < 1:
            raise ValueError("chunk_size and image_size must be positive")
        self.collection_config = dict(collection_config or {})
        self.video_encoding = dict(CAUSAL_VIDEO_ENCODING)
        self.task_description = str(task_description)
        self.source_policy = str(source_policy)
        self.source_policy_version = str(source_policy_version)
        self.robot = str(robot)
        self.environment_contract = environment_contract
        validate_recorded_environment_contract(dict(
            environment_contract=environment_contract, collection_config=self.collection_config,
            control_period_seconds=1.0 / self.fps))
        if not self.task_description.strip():
            raise ValueError("task_description cannot be empty")
        if not self.source_policy_version.strip():
            raise ValueError("source_policy_version cannot be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.meta_dir / "dataset.json"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.episode_stats_path = self.meta_dir / "episodes_stats.jsonl"
        self.stats_path = self.meta_dir / "stats.json"
        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self._initialize_metadata()

    def _initialize_metadata(self):
        if self.dataset_path.exists():
            with self.dataset_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            validate_video_encoding(metadata)
            if metadata.get("format") != FORMAT_NAME:
                raise ValueError(f"{self.root} contains a different dataset format")
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"Unsupported dataset version in {self.root}")
            if metadata.get("source_policy_version") != self.source_policy_version:
                raise ValueError(
                    f"{self.root} was collected by a different expert policy; "
                    "do not mix it with strict stable-grasp episodes"
                )
            if metadata.get("task") != self.task_description:
                raise ValueError(
                    f"{self.root} contains task {metadata.get('task')!r}, expected "
                    f"{self.task_description!r}"
                )
            if metadata.get("collection_config", {}) != self.collection_config:
                raise ValueError(
                    f"{self.root} collection_config does not match this run; "
                    "use --overwrite or keep collection settings identical"
                )
            if self.environment_contract is not None and metadata.get("environment_contract") != self.environment_contract:
                raise ValueError(f"{self.root} environment provenance differs or is missing; do not append mixed environments")
            if (metadata.get("fps") != self.fps
                    or metadata.get("chunk_size") != self.chunk_size
                    or any(metadata.get("cameras", {}).get(camera, {}).get("shape")
                           != [self.image_size, self.image_size, 3] for camera in CAMERAS)
                    or any(metadata.get("cameras", {}).get(camera, {}).get("fps")
                           != self.fps for camera in CAMERAS)):
                raise ValueError(
                    f"{self.root} video resolution, frame rate or chunk size differs; "
                    "do not append incompatible episodes to this dataset"
                )
        else:
            now = _utc_now()
            metadata = {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "inspired_by": [
                    "LeRobot v2.1 episode files and metadata",
                    "LeRobot v3 chunk organization",
                    "LIBERO task and episode semantics",
                ],
                "official_lerobot_dataset": False,
                "robot": self.robot,
                "task": self.task_description,
                "created_at": now,
                "updated_at": now,
                "fps": self.fps,
                "physics_hz": 500.0,
                "control_period_seconds": 1.0 / self.fps,
                "chunk_size": self.chunk_size,
                "total_episodes": 0,
                "total_frames": 0,
                "source_policy": self.source_policy,
                "source_policy_version": self.source_policy_version,
                "collection_config": self.collection_config,
                "environment_contract": self.environment_contract,
                "layout": {
                    "data": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
                    "video": "videos/chunk-{chunk_index:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4",
                },
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(STATE_NAMES)],
                        "names": STATE_NAMES,
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [len(ACTION_NAMES)],
                        "names": ACTION_NAMES,
                    },
                    "video_timestamp": {
                        "dtype": "float64",
                        "unit": "seconds",
                        "origin": "first_video_frame",
                    },
                    "expert.phase": {"dtype": "string"},
                    "next.done": {"dtype": "bool"},
                },
                "cameras": {
                    camera: {
                        "dtype": "video",
                        "shape": [self.image_size, self.image_size, 3],
                        "fps": self.fps,
                        "codec": "h264",
                    }
                    for camera in CAMERAS
                },
            }
            metadata["video_encoding"] = self.video_encoding
            _atomic_json(self.dataset_path, metadata)

        if not self.tasks_path.exists():
            _atomic_jsonl(
                self.tasks_path,
                [
                    {
                        "task_index": 0,
                        "task": self.task_description,
                    }
                ],
            )

    @property
    def next_episode_index(self):
        rows = _read_jsonl(self.episodes_path)
        return max((row["episode_index"] for row in rows), default=-1) + 1

    def _paths(self, episode_index):
        chunk_index = episode_index // self.chunk_size
        stem = f"episode_{episode_index:06d}"
        data = self.root / "data" / f"chunk-{chunk_index:03d}" / f"{stem}.parquet"
        videos = {
            camera: self.root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / f"observation.images.{camera}"
            / f"{stem}.mp4"
            for camera in CAMERAS
        }
        return chunk_index, data, videos

    def record_collection_report(self, report):
        """Keep rejected seeds/reasons in existing metadata, outside accepted episodes."""
        metadata = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        metadata["last_collection_report"] = report
        metadata["updated_at"] = _utc_now()
        _atomic_json(self.dataset_path, metadata)

    def write_episode(
        self,
        states,
        actions,
        images,
        phases,
        dones,
        seed,
        quality,
        success=True,
        timestamps=None,
    ):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        length = len(states)
        if timestamps is None:
            timestamps = np.arange(length, dtype=np.float64) / self.fps
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64)
        if self.collection_config.get("task_id") in (3, 4):
            validate_contact_quality(quality, self.collection_config["task_id"])
            validate_correction_context(phases, quality, self.collection_config["task_id"], timestamps)
        if not success:
            raise ValueError("Unsuccessful episodes must not be written")
        if (
            length == 0
            or len(actions) != length
            or len(phases) != length
            or len(dones) != length
        ):
            raise ValueError("State, action, phase, and done lengths must match")
        if (states.shape != (length, len(STATE_NAMES))
                or actions.shape != (length, len(ACTION_NAMES))):
            raise ValueError("Episode states/actions must have native dimensions 8/7")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Episode states and actions must be finite")
        if dones.shape != (length,) or not dones[-1] or dones[:-1].any():
            raise ValueError("Only the final episode row must be done")
        if timestamps.shape != (length,):
            raise ValueError("Timestamp length must match state length")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                "Timestamps must be finite and strictly increasing"
            )
        for camera in CAMERAS:
            if camera not in images or len(images[camera]) != length:
                raise ValueError(f"Camera {camera} does not have {length} frames")
            if any(np.asarray(image).shape != (self.image_size, self.image_size, 3)
                   or np.asarray(image).dtype != np.uint8 for image in images[camera]):
                raise ValueError(f"Camera {camera} frames must match the declared uint8 RGB resolution")

        episode_index = self.next_episode_index
        chunk_index, data_path, video_paths = self._paths(episode_index)
        all_final_paths = [data_path, *video_paths.values()]
        if any(path.exists() for path in all_final_paths):
            raise FileExistsError(f"Episode {episode_index} already has output files")

        frame = pd.DataFrame(
            {
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "timestamp": timestamps.astype(np.float32),
                # MP4 starts at frame zero even when simulation starts after settling.
                "video_timestamp": np.arange(length, dtype=np.float64) / self.fps,
                "task_index": np.zeros(length, dtype=np.int64),
                "seed": np.full(length, int(seed), dtype=np.int64),
                "observation.state": [row.tolist() for row in states],
                "action": [row.tolist() for row in actions],
                "expert.phase": list(phases),
                "next.done": dones,
            }
        )

        metadata_before = {
            path: path.read_bytes() if path.exists() else None
            for path in (self.episodes_path, self.episode_stats_path, self.stats_path, self.dataset_path)
        }
        metadata_started = False
        temporary_paths = []
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_temp = data_path.with_name(
                f".{data_path.stem}.{uuid.uuid4().hex}.tmp.parquet"
            )
            temporary_paths.append((data_temp, data_path))
            frame.to_parquet(data_temp, index=False)

            for camera, video_path in video_paths.items():
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_temp = video_path.with_name(
                    f".{video_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
                )
                temporary_paths.append((video_temp, video_path))
                write_rgb_video(video_temp, images[camera], self.fps)

            for temporary, final in temporary_paths:
                os.replace(temporary, final)

            state_stats = feature_stats(states)
            supervised = np.asarray([
                phase not in {"observe_target", "presentation_close", "policy_context"}
                for phase in phases
            ])
            action_stats = feature_stats(actions[supervised])
            episode_row = {
                "episode_index": episode_index,
                "chunk_index": chunk_index,
                "length": length,
                "seed": int(seed),
                "success": True,
                "task_index": 0,
                "task": self.task_description,
                "data_path": data_path.relative_to(self.root).as_posix(),
                "video_paths": {
                    camera: path.relative_to(self.root).as_posix()
                    for camera, path in video_paths.items()
                },
                "quality": quality,
            }
            stats_row = {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": state_stats,
                    "action": action_stats,
                },
            }
            episode_rows = _read_jsonl(self.episodes_path) + [episode_row]
            stats_rows = _read_jsonl(self.episode_stats_path) + [stats_row]
            metadata_started = True
            _atomic_jsonl(self.episodes_path, episode_rows)
            _atomic_jsonl(self.episode_stats_path, stats_rows)
            _atomic_json(
                self.stats_path,
                {
                    "observation.state": merge_feature_stats(
                        stats_rows, "observation.state"
                    ),
                    "action": merge_feature_stats(stats_rows, "action"),
                },
            )

            with self.dataset_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            metadata["updated_at"] = _utc_now()
            metadata["total_episodes"] = len(episode_rows)
            metadata["total_frames"] = sum(row["length"] for row in episode_rows)
            _atomic_json(self.dataset_path, metadata)
            return episode_row
        except BaseException:
            # Collection interrupts must leave the accepted manifest, stats and
            # files in agreement. All final paths were absent before this try;
            # only this attempt's outputs are removed. Existing episodes remain.
            for temporary, final in temporary_paths:
                temporary.unlink(missing_ok=True)
                final.unlink(missing_ok=True)
            if metadata_started:
                for path, payload in metadata_before.items():
                    if payload is None:
                        path.unlink(missing_ok=True)
                    else:
                        _atomic_bytes(path, payload)
            raise

    def validate_episode(self, episode_index):
        rows = _read_jsonl(self.episodes_path)
        row = next(item for item in rows if item["episode_index"] == episode_index)
        parquet_path = self.root / row["data_path"]
        table = pd.read_parquet(parquet_path)
        if len(table) != row["length"]:
            raise ValueError("Parquet length does not match episode metadata")
        for relative_path in row["video_paths"].values():
            path = self.root / relative_path
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty video: {path}")
        return {
            "episode_index": episode_index,
            "frames": len(table),
            "columns": list(table.columns),
            "videos": row["video_paths"],
        }
