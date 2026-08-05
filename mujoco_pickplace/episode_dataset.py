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
STATE_NAMES = [
    "eef_x", "eef_y", "eef_z",
    "cube_x", "cube_y", "cube_z",
    "goal_x", "goal_y", "goal_z",
    "gripper",
]
ACTION_NAMES = ["eef_dx", "eef_dy", "eef_dz", "gripper"]
CAMERAS = ("front", "overhead", "wrist")


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
    second = np.sum((stds ** 2 + means ** 2) * counts[:, None], axis=0) / count
    return {
        "count": int(count),
        "min": np.min([entry["min"] for entry in entries], axis=0).tolist(),
        "max": np.max([entry["max"] for entry in entries], axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - mean ** 2, 0.0)).tolist(),
    }


class EpisodeDatasetWriter:

    def __init__(self, root, fps=5.0, chunk_size=1000, image_size=448):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        self.fps = float(fps)
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
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
                "robot": "handwritten-panda7",
                "task": "pick up the blue cube and place it on the green target",
                "created_at": now,
                "updated_at": now,
                "fps": self.fps,
                "physics_hz": 500.0,
                "control_period_seconds": 1.0 / self.fps,
                "chunk_size": self.chunk_size,
                "total_episodes": 0,
                "total_frames": 0,
                "source_policy": "stateful contact-aware scripted expert",
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
            _atomic_jsonl(self.tasks_path, [{
                "task_index": 0,
                "task": "pick up the blue cube and place it on the green target",
            }])

    @property
    def next_episode_index(self):
        rows = _read_jsonl(self.episodes_path)
        return max((row["episode_index"] for row in rows), default=-1) + 1

    def _paths(self, episode_index):
        chunk_index = episode_index // self.chunk_size
        stem = f"episode_{episode_index:06d}"
        data = self.root / "data" / f"chunk-{chunk_index:03d}" / f"{stem}.parquet"
        videos = {
            camera: self.root / "videos" / f"chunk-{chunk_index:03d}" / f"observation.images.{camera}" / f"{stem}.mp4"
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
    ):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        length = len(states)
        if not success:
            raise ValueError("Unsuccessful episodes must not be written")
        if length == 0 or len(actions) != length or len(phases) != length or len(dones) != length:
            raise ValueError("State, action, phase, and done lengths must match")
        for camera in CAMERAS:
            if camera not in images or len(images[camera]) != length:
                raise ValueError(f"Camera {camera} does not have {length} frames")

        episode_index = self.next_episode_index
        chunk_index, data_path, video_paths = self._paths(episode_index)
        all_final_paths = [data_path, *video_paths.values()]
        if any(path.exists() for path in all_final_paths):
            raise FileExistsError(f"Episode {episode_index} already has output files")

        frame = pd.DataFrame({
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / self.fps,
            "task_index": np.zeros(length, dtype=np.int64),
            "seed": np.full(length, int(seed), dtype=np.int64),
            "observation.state": [row.tolist() for row in states],
            "action": [row.tolist() for row in actions],
            "expert.phase": list(phases),
            "next.done": dones,
        })

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
                "task": "pick up the blue cube and place it on the green target",
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
            _atomic_json(self.stats_path, {
                "observation.state": merge_feature_stats(stats_rows, "observation.state"),
                "action": merge_feature_stats(stats_rows, "action"),
            })

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
