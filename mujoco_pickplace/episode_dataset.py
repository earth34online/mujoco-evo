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
    }
    if task_number == 4:
        required.update(
            {
                "horizontal_handle_approach",
                "handle_contact_through_pull",
                "no_wrong_drawer_opened",
                "correct_first_drawer",
                "presentation_complete",
                "selection_within_history",
            }
        )
    if (
        quality.get("schema") != "mujoco-panda-contact"
        or quality.get("schema_version") != 1
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


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    os.replace(temporary, path)


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
    ):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        self.collection_config = dict(collection_config or {})
        self.task_description = str(task_description)
        self.source_policy = str(source_policy)
        self.source_policy_version = str(source_policy_version)
        self.robot = str(robot)
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
        if not success:
            raise ValueError("Unsuccessful episodes must not be written")
        if (
            length == 0
            or len(actions) != length
            or len(phases) != length
            or len(dones) != length
        ):
            raise ValueError("State, action, phase, and done lengths must match")
        if len(timestamps) != length:
            raise ValueError("Timestamp length must match state length")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError(
                "Timestamps must be finite and monotonically nondecreasing"
            )
        for camera in CAMERAS:
            if camera not in images or len(images[camera]) != length:
                raise ValueError(f"Camera {camera} does not have {length} frames")

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

        temporary_paths = []
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_temp = data_path.with_name(
                f".{data_path.stem}.{uuid.uuid4().hex}.tmp.parquet"
            )
            frame.to_parquet(data_temp, index=False)
            temporary_paths.append((data_temp, data_path))

            for camera, video_path in video_paths.items():
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_temp = video_path.with_name(
                    f".{video_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
                )
                imageio.mimsave(
                    video_temp,
                    np.asarray(images[camera], dtype=np.uint8),
                    fps=self.fps,
                    macro_block_size=1,
                )
                temporary_paths.append((video_temp, video_path))

            for temporary, final in temporary_paths:
                os.replace(temporary, final)

            state_stats = feature_stats(states)
            action_stats = feature_stats(actions)
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
        except Exception:
            for temporary, final in temporary_paths:
                temporary.unlink(missing_ok=True)
                final.unlink(missing_ok=True)
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
