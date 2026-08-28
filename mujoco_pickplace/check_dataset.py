import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "Mujoco_training_dataset" / "cache" / "mujoco_pickplace"
)
EVO_ROOT = PROJECT_ROOT / "Evo-1" / "Evo_1"

MAX_ACTION_DIM = 24
MAX_STATE_DIM = 24
MAX_VIEWS = 3
ACTIVE_ACTION_MASK = [True, True, True, False, False, False, True]
ACTION_HORIZON = 14
IMAGE_SIZE = 448

REQUIRED_META_FILES = (
    "dataset.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "stats.json",
)
REQUIRED_PARQUET_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "seed",
    "observation.state",
    "action",
    "expert.phase",
    "next.done",
)


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
    return rows


def _require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty {description}: {path}")


def _check_vector_column(series, expected_dim, name, parquet_path):
    for row_index, value in enumerate(series):
        array = np.asarray(value)
        if array.shape != (expected_dim,):
            raise AssertionError(
                f"{parquet_path}: row {row_index} {name} has shape "
                f"{array.shape}, expected {(expected_dim,)}"
            )
        if not np.all(np.isfinite(array)):
            raise AssertionError(
                f"{parquet_path}: row {row_index} {name} contains non-finite values"
            )


def check_raw_dataset(dataset_dir):
    dataset_dir = dataset_dir.resolve()
    meta_dir = dataset_dir / "meta"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    for filename in REQUIRED_META_FILES:
        _require_file(meta_dir / filename, f"metadata file {filename}")

    dataset_info = _read_json(meta_dir / "dataset.json")
    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    episode_stats = _read_jsonl(meta_dir / "episodes_stats.jsonl")
    stats = _read_json(meta_dir / "stats.json")

    if not episodes:
        raise AssertionError(f"No episodes listed in {meta_dir / 'episodes.jsonl'}")

    state_dim = int(dataset_info["features"]["observation.state"]["shape"][0])
    action_dim = int(dataset_info["features"]["action"]["shape"][0])
    expected_episodes = int(dataset_info["total_episodes"])
    expected_frames = int(dataset_info["total_frames"])
    expected_fps = float(dataset_info["fps"])

    if expected_episodes != len(episodes):
        raise AssertionError(
            f"dataset.json total_episodes={expected_episodes}, but episodes.jsonl "
            f"contains {len(episodes)} rows"
        )
    if len(episode_stats) != len(episodes):
        raise AssertionError(
            f"episodes_stats.jsonl contains {len(episode_stats)} rows, expected "
            f"{len(episodes)}"
        )
    if not tasks:
        raise AssertionError("tasks.jsonl does not contain any task")

    episode_indices = [int(row["episode_index"]) for row in episodes]
    if episode_indices != list(range(len(episodes))):
        raise AssertionError(
            "Episode indices are not contiguous and ordered from zero: "
            f"first={episode_indices[:5]}, last={episode_indices[-5:]}"
        )

    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_files) != len(episodes):
        raise AssertionError(
            f"Found {len(parquet_files)} parquet files, expected {len(episodes)}"
        )

    total_frames = 0
    video_files = set()
    required_columns = set(REQUIRED_PARQUET_COLUMNS)

    for position, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        parquet_path = dataset_dir / episode["data_path"]

        _require_file(parquet_path, f"episode {episode_index} parquet")
        for camera, relative_path in episode.get("video_paths", {}).items():
            video_path = dataset_dir / relative_path
            _require_file(video_path, f"episode {episode_index} {camera} video")
            video_files.add(video_path.resolve())

        frame = pd.read_parquet(parquet_path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise AssertionError(
                f"{parquet_path} is missing columns: {missing_columns}"
            )
        if len(frame) != expected_length:
            raise AssertionError(
                f"{parquet_path} contains {len(frame)} rows, expected {expected_length}"
            )
        if not np.all(frame["episode_index"].to_numpy() == episode_index):
            raise AssertionError(f"{parquet_path} contains the wrong episode_index")
        if not np.array_equal(
            frame["frame_index"].to_numpy(), np.arange(expected_length)
        ):
            raise AssertionError(f"{parquet_path} has non-contiguous frame_index values")

        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(timestamps)):
            raise AssertionError(f"{parquet_path} contains non-finite timestamps")
        if len(timestamps) > 1:
            timestamp_steps = np.diff(timestamps)
            if not np.all(timestamp_steps > 0):
                raise AssertionError(f"{parquet_path} timestamps are not increasing")
            expected_period = 1.0 / expected_fps
            if not np.allclose(
                timestamp_steps,
                expected_period,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise AssertionError(
                    f"{parquet_path} timestamp period does not match fps={expected_fps}"
                )

        _check_vector_column(
            frame["observation.state"],
            state_dim,
            "observation.state",
            parquet_path,
        )
        _check_vector_column(frame["action"], action_dim, "action", parquet_path)

        if not bool(frame["next.done"].iloc[-1]):
            raise AssertionError(f"{parquet_path} final next.done is not true")
        if bool(frame["next.done"].iloc[:-1].any()):
            raise AssertionError(f"{parquet_path} has next.done before the final row")

        total_frames += len(frame)
        if position == 1 or position % 50 == 0 or position == len(episodes):
            print(f"  checked raw episodes: {position}/{len(episodes)}", flush=True)

    if total_frames != expected_frames:
        raise AssertionError(
            f"Validated {total_frames} frames, dataset.json reports {expected_frames}"
        )

    for feature_name, expected_dim in (
        ("observation.state", state_dim),
        ("action", action_dim),
    ):
        feature_stats = stats.get(feature_name)
        if feature_stats is None:
            raise AssertionError(f"stats.json is missing {feature_name}")
        for statistic in ("min", "max", "mean", "std"):
            values = np.asarray(feature_stats.get(statistic))
            if values.shape != (expected_dim,):
                raise AssertionError(
                    f"stats.json {feature_name}.{statistic} has shape {values.shape}, "
                    f"expected {(expected_dim,)}"
                )
            if not np.all(np.isfinite(values)):
                raise AssertionError(
                    f"stats.json {feature_name}.{statistic} contains non-finite values"
                )

    print("[PASS] raw dataset is internally consistent")
    print(f"  dataset: {dataset_dir}")
    print(f"  episodes: {len(episodes)}")
    print(f"  frames: {total_frames}")
    print(f"  parquet files: {len(parquet_files)}")
    print(f"  video files: {len(video_files)}")
    print(f"  native state/action dims: {state_dim}/{action_dim}")
    return dataset_info


def _missing_evo_dependencies():
    required = ("torch", "torchvision", "PIL", "av")
    return [name for name in required if importlib.util.find_spec(name) is None]


def check_evo_interface(dataset_dir, dataset_info, image_size, action_horizon):
    missing = _missing_evo_dependencies()
    if missing:
        raise ModuleNotFoundError(
            "Missing Evo interface dependencies: " + ", ".join(missing)
        )
    if not EVO_ROOT.is_dir():
        raise FileNotFoundError(f"Evo-1 source directory does not exist: {EVO_ROOT}")

    sys.path.insert(0, str(EVO_ROOT))
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset

    cameras = list(dataset_info.get("cameras", {}).keys())
    if not cameras:
        raise AssertionError("dataset.json does not define any camera")

    view_map = {
        f"image_{index + 1}": f"observation.images.{camera}"
        for index, camera in enumerate(cameras)
    }
    config = {
        "max_action_dim": MAX_ACTION_DIM,
        "max_state_dim": MAX_STATE_DIM,
        "max_views": MAX_VIEWS,
        "active_action_mask": [int(value) for value in ACTIVE_ACTION_MASK],
        "data_groups": {
            "mujoco_pickplace": {
                "MuJoCo_PickPlace_Dataset": {
                    "path": str(dataset_dir.resolve()),
                    "view_map": view_map,
                }
            }
        },
    }

    dataset = LeRobotDataset(
        config=config,
        image_size=image_size,
        action_horizon=action_horizon,
        max_samples_per_file=None,
        use_augmentation=False,
        overwrite_horizon_cache=False,
    )
    if len(dataset) <= 0:
        raise AssertionError("Evo LeRobotDataset did not produce any samples")

    print(f"Evo dataset length: {len(dataset)}")
    item = dataset[0]
    for key, value in item.items():
        if hasattr(value, "shape"):
            print(key, value.shape, value.dtype)
        else:
            print(key, value)

    expected_views = min(len(cameras), MAX_VIEWS)
    expected_image_mask = [True] * expected_views + [False] * (
        MAX_VIEWS - expected_views
    )
    expected_action_mask = ACTIVE_ACTION_MASK + [False] * (
        MAX_ACTION_DIM - len(ACTIVE_ACTION_MASK)
    )

    assert item["images"].shape == (MAX_VIEWS, 3, image_size, image_size)
    assert item["state"].shape == (MAX_STATE_DIM,)
    assert item["action"].shape == (action_horizon, MAX_ACTION_DIM)
    assert item["image_mask"].tolist() == expected_image_mask
    assert item["action_mask"].shape == (action_horizon, MAX_ACTION_DIM)
    assert item["action_mask"][0].tolist() == expected_action_mask

    print("[PASS] Evo dataset interface is correct")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check the raw MuJoCo dataset and, when available, its Evo interface."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory containing data/, videos/, and meta/.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Check metadata, parquet files, and video paths without loading Evo.",
    )
    parser.add_argument(
        "--require-evo",
        action="store_true",
        help="Fail instead of skipping when Evo interface dependencies are unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_info = check_raw_dataset(args.dataset_dir)

    if args.raw_only:
        print("[SKIP] Evo interface check disabled by --raw-only")
        return

    missing = _missing_evo_dependencies()
    if missing and not args.require_evo:
        print(
            "[SKIP] Evo interface check requires: "
            + ", ".join(missing)
            + ". Run this script in the Evo1 environment for the full interface check."
        )
        return

    check_evo_interface(
        args.dataset_dir,
        dataset_info,
        image_size=args.image_size,
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
