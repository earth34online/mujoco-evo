import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mujoco_pickplace.episode_dataset import (
    validate_contact_quality as check_contact_quality,
    validate_precision_grasp_quality,
    MAX_SUCCESSFUL_CLOSE_XY_ERROR,
    MIN_SUCCESSFUL_CLOSE_Z_ABOVE,
    MAX_SUCCESSFUL_CLOSE_Z_ABOVE,
    MAX_PREGRASP_CUBE_DISPLACEMENT,
    MAX_PREGRASP_CUBE_TILT_DEG,
    MAX_ATTACH_CUBE_TILT_DEG,
    MAX_HELD_CUBE_TILT_DEG,
    MAX_ATTACH_CUBE_ANGULAR_SPEED,
)

EVO_ROOT = PROJECT_ROOT / "Evo-1" / "Evo_1"

MAX_ACTION_DIM = 24
MAX_STATE_DIM = 24
MAX_VIEWS = 3
ACTION_HORIZON = 14
IMAGE_SIZE = 448
LEGACY_PRECISION_POLICY_VERSION = "precision-grasp-recovery-v2"
STRICT_PRECISION_POLICY_VERSION = "precision-grasp-stable-v3"
PRECISION_POLICY_VERSIONS = {
    LEGACY_PRECISION_POLICY_VERSION,
    STRICT_PRECISION_POLICY_VERSION,
}
EXPECTED_CUBE_SIDE_M = 0.050
EXPECTED_CUBE_SUPPORT_Z_M = 0.055
EXPECTED_EXPERT_GRASP_X_BIAS_M = 0.004

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


def check_raw_dataset(
    dataset_dir,
    require_precision_grasp=False,
    require_strict_grasp_quality=False,
    task_number=1,
):
    dataset_dir = dataset_dir.resolve()
    precision_checks = task_number == 1 and (
        require_precision_grasp or require_strict_grasp_quality
    )
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

    collection_config = dataset_info.get("collection_config", {})
    source_policy_version = dataset_info.get("source_policy_version")
    if task_number in (3, 4):
        if source_policy_version != f"task{task_number}-contact-v1":
            raise AssertionError(
                f"Task{task_number} requires its native contact expert dataset"
            )
        if collection_config.get("task_id") != task_number:
            raise AssertionError("Dataset task_id does not match the selected task")
        if collection_config.get("compact_static_frames") is not False:
            raise AssertionError(
                "Memory datasets must preserve the dense simulation timeline"
            )
        if state_dim != 8 or action_dim != 7:
            raise AssertionError("Expected the shared Panda 8-state/7-action contract")
    if precision_checks:
        if source_policy_version not in PRECISION_POLICY_VERSIONS:
            raise AssertionError(
                "Dataset was not collected by the precision-grasp recovery "
                f"expert ({sorted(PRECISION_POLICY_VERSIONS)})"
            )
        if (
            require_strict_grasp_quality
            and source_policy_version != STRICT_PRECISION_POLICY_VERSION
        ):
            raise AssertionError(
                "Dataset predates contact-time stable-grasp validation; expected "
                f"{STRICT_PRECISION_POLICY_VERSION}, got {source_policy_version}"
            )
        if collection_config.get("randomize_task") is not True:
            raise AssertionError(
                "Precision dataset must keep randomized cube/goal positions"
            )
        if not np.isclose(
            float(collection_config.get("randomization_scale", np.nan)), 1.0
        ):
            raise AssertionError(
                "Precision dataset must use the full randomization range (1.0)"
            )
        if collection_config.get("compact_static_frames") is not False:
            raise AssertionError(
                "π-MEM precision dataset must keep the dense, uncompacted timeline"
            )
        geometry_contract = {
            "cube_side_m": EXPECTED_CUBE_SIDE_M,
            "cube_support_z_m": EXPECTED_CUBE_SUPPORT_Z_M,
            "expert_grasp_x_bias_m": EXPECTED_EXPERT_GRASP_X_BIAS_M,
        }
        for key, expected in geometry_contract.items():
            actual = collection_config.get(key)
            if actual is None or not np.isclose(float(actual), expected):
                raise AssertionError(
                    "Dataset geometry does not match the current 50 mm task: "
                    f"{key} expected {expected}, got {actual}. Recollect with "
                    "--overwrite; do not append to the old 60 mm dataset."
                )

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
    initial_cube_positions = []
    initial_goal_positions = []

    for position, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        parquet_path = dataset_dir / episode["data_path"]
        if task_number in (3, 4):
            check_contact_quality(episode.get("quality", {}), task_number)

        if precision_checks:
            quality = episode.get("quality", {})
            validate_precision_grasp_quality(
                quality,
                episode_index,
                strict=source_policy_version == STRICT_PRECISION_POLICY_VERSION,
            )
            initial_cube_positions.append(quality["initial_cube_xy"])
            initial_goal_positions.append(quality["initial_goal_xy"])

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
            raise AssertionError(
                f"{parquet_path} has non-contiguous frame_index values"
            )

        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(timestamps)):
            raise AssertionError(f"{parquet_path} contains non-finite timestamps")
        if len(timestamps) > 1:
            timestamp_steps = np.diff(timestamps)
            if not np.all(timestamp_steps > 0):
                raise AssertionError(f"{parquet_path} timestamps are not increasing")
            expected_period = 1.0 / expected_fps
            # Static-frame compaction preserves real collection timestamps, so
            # adjacent retained frames may span multiple control periods.
            period_multiples = timestamp_steps / expected_period
            if not np.allclose(
                period_multiples,
                np.round(period_multiples),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise AssertionError(
                    f"{parquet_path} timestamps are not aligned to fps={expected_fps}"
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

    if precision_checks:
        if len(episodes) > 1:
            if len(np.unique(np.asarray(initial_cube_positions), axis=0)) < 2:
                raise AssertionError("Cube positions are fixed across the dataset")
            if len(np.unique(np.asarray(initial_goal_positions), axis=0)) < 2:
                raise AssertionError("Goal positions are fixed across the dataset")
        if source_policy_version == STRICT_PRECISION_POLICY_VERSION:
            print(
                "  strict contact-time grasp/randomization checks passed",
                flush=True,
            )
        else:
            print(
                "  legacy precision checks passed; contact-time tilt/displacement "
                "quality is unavailable",
                flush=True,
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


def _check_native_video_metadata(dataset_path):
    """Check every native MP4 header without substituting frames or decoding a rollout."""
    import av

    info = json.loads((dataset_path / "meta/dataset.json").read_text(encoding="utf-8"))
    if info.get("format") != "mujoco-evo-episodes":
        return
    for episode in _read_jsonl(dataset_path / "meta/episodes.jsonl"):
        for camera, relative in episode["video_paths"].items():
            path = dataset_path / relative
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                count = int(stream.frames)
                if count <= 0:
                    count = sum(1 for _ in container.decode(video=0))
                if count != int(episode["length"]):
                    raise AssertionError(
                        f"{path} contains {count} video frames, expected {episode['length']}"
                    )
                rate = float(stream.average_rate or 0)
                if not np.isfinite(rate) or abs(rate - float(info["fps"])) > 1e-6:
                    raise AssertionError(
                        f"{path} video fps={rate} disagrees with dataset fps={info['fps']}"
                    )
                expected = info["cameras"][camera]
                if (stream.width, stream.height) != (
                    int(expected["shape"][1]),
                    int(expected["shape"][0]),
                ):
                    raise AssertionError(
                        f"{path} video resolution disagrees with dataset camera metadata"
                    )


def check_evo_interface(
    datasets,
    image_size,
    action_horizon,
    memory_frames,
    memory_stride_steps,
    memory_stride_seconds,
    training_cache_dir=None,
):
    missing = _missing_evo_dependencies()
    if missing:
        raise ModuleNotFoundError(
            "Missing Evo interface dependencies: " + ", ".join(missing)
        )
    sys.path.insert(0, str(EVO_ROOT))
    import yaml
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset

    # Use the actual training configuration, including each dataset's action mask.
    with (EVO_ROOT / "dataset/config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    group = config["data_groups"]["mujoco_panda"]
    selected = {}
    for task, path, info in datasets:
        _check_native_video_metadata(path)
        entry = dict(group[task.key])
        entry["path"] = str(path.resolve())
        selected[task.key] = entry
        if tuple(entry["active_action_mask"]) != task.action_mask:
            raise AssertionError(f"Task{task.number} training/action contract mismatch")
    config["data_groups"] = {"mujoco_panda": selected}
    # Interface probes must not replace the full training manifest/cache.
    config["cache_namespace"] += "_check"
    dataset = LeRobotDataset(
        config=config,
        image_size=image_size,
        action_horizon=action_horizon,
        max_samples_per_file=1,
        use_augmentation=False,
        overwrite_horizon_cache=False,
        memory_frames=memory_frames,
        memory_stride_steps=memory_stride_steps,
        memory_stride_seconds=memory_stride_seconds,
        cache_dir=training_cache_dir,
        video_backend="av",
    )
    if len(dataset) <= 0:
        raise AssertionError("Evo LeRobotDataset did not produce any samples")
    print(f"Evo dataset length: {len(dataset)} (one interface window per episode)")
    for task, path, info in datasets:
        index = next(
            (
                i
                for i, cached in enumerate(dataset.data)
                if Path(cached).relative_to(dataset.cache_dir).parts[1] == task.key
            ),
            None,
        )
        if index is None:
            raise AssertionError(f"Task{task.number} has no Evo interface sample")
        item = dataset[index]
        cameras = list(info.get("cameras", {}))
        if not cameras:
            raise AssertionError("dataset.json does not define any camera")
        expected_views = min(len(cameras), MAX_VIEWS)
        expected_image_mask = [True] * expected_views + [False] * (
            MAX_VIEWS - expected_views
        )
        expected_action_mask = list(map(bool, task.action_mask)) + [False] * (
            MAX_ACTION_DIM - len(task.action_mask)
        )
        assert item["images"].shape == (
            memory_frames,
            MAX_VIEWS,
            3,
            image_size,
            image_size,
        )
        assert item["state"].shape == (memory_frames, MAX_STATE_DIM)
        assert item["state_mask"].shape == (memory_frames, MAX_STATE_DIM)
        assert item["history_mask"].shape == (memory_frames,)
        assert bool(item["history_mask"][-1])
        assert item["action"].shape == (action_horizon, MAX_ACTION_DIM)
        assert item["image_mask"].tolist() == expected_image_mask
        assert item["action_mask"].shape == (action_horizon, MAX_ACTION_DIM)
        assert item["action_mask"][0].tolist() == expected_action_mask
        assert int(item["embodiment_id"]) == 0
        print(
            f"[PASS] Task{task.number} Evo interface: images={tuple(item['images'].shape)}, "
            f"state={tuple(item['state'].shape)}, action={tuple(item['action'].shape)}, "
            f"active actions={item['action_mask'][0].tolist()[:7]}"
        )
    print("[PASS] all selected tasks share one Panda training group")


def _replay_drawer_contact(task, episode, frame, predictions):
    """Isolate pose errors; no expert actions are substituted in normal evaluation."""
    from mujoco_pickplace.eval_policy_client import GripperCommandFilter
    import mujoco

    indices = frame.index[
        frame["expert.phase"].isin(
            ["approach", "handle_approach", "handle_close", "pull"]
        )
    ].tolist()
    results = []
    for mode in ("expert", "model_translation", "model_translation_rotation"):
        env = task.make_env(seed=episode["seed"], image_size=128, capture_frames=False)
        try:
            list(env.presentation())
            grip = GripperCommandFilter()
            initial = float(
                env.data.qpos[env.model.joint(f"drawer_slide{env.target}").qposadr[0]]
            )
            for index in indices:
                action = np.asarray(
                    frame.iloc[index]["action"], dtype=np.float32
                ).copy()
                if mode != "expert":
                    action[:3] = predictions[index][:3]
                if mode == "model_translation_rotation":
                    action[3:6] = predictions[index][3:6]
                env.phase = str(frame.iloc[index]["expert.phase"])
                env.step(task.prepare_action(action, grip))
            pull = [row for row in env.rows if row["phase"] == "pull"]
            position = float(
                env.data.qpos[env.model.joint(f"drawer_slide{env.target}").qposadr[0]]
            )
            results.append(
                dict(
                    mode=mode,
                    mujoco_version=mujoco.__version__,
                    steps=len(indices),
                    drawer_displacement_mm=(position - initial) * 1000,
                    two_finger_handle_contact=bool(
                        env.contacts(env.model.geom(f"handle{env.target}").id)[0]
                    ),
                    unsafe_contact=bool(env.unsafe_execution_contact),
                    pull_contact_fraction=(
                        float(
                            np.mean(
                                [
                                    row["handle_two_contact_physics_steps"]
                                    / env.CONTROL_NSTEP
                                    for row in pull
                                ]
                            )
                        )
                        if pull
                        else None
                    ),
                    longest_pull_contact_gap_seconds=max(
                        [row["longest_handle_contact_gap_seconds"] for row in pull],
                        default=0.0,
                    ),
                    maximum_pull_normal_error_mm=max(
                        [row["handle_normal_tracking_error_m"] * 1000 for row in pull],
                        default=0.0,
                    ),
                    maximum_pull_inclination_degrees=max(
                        [row["peak_tool_axis_inclination_deg"] for row in pull],
                        default=0.0,
                    ),
                )
            )
        finally:
            env.close()
    if results[0]["drawer_displacement_mm"] < 160 or results[0]["unsafe_contact"]:
        raise AssertionError(
            "Contact reference replay no longer matches the accepted expert"
        )
    return results


async def _check_drawer_history_response(socket, task, path, memory_frames, stride):
    """Hold current vision and all proprioception fixed; vary visual history only.

    Target labels annotate results after inference. This is a controlled history
    response check, not evidence of successful drawer selection or manipulation.
    """
    import av
    import copy
    from collections import deque
    from mujoco_pickplace.tasks import ObservationHistory
    from dataset.lerobot_dataset_pretrain_mp import select_history_indices

    episodes = {}
    for episode in _read_jsonl(path / "meta/episodes.jsonl"):
        target = episode["quality"].get("target_drawer_diagnostic_only")
        if target is not None:
            episodes.setdefault(int(target), episode)
    if len(episodes) < 2:
        return dict(checked=False, available_targets=sorted(episodes))
    cases, anchor = [], None
    for target, episode in sorted(episodes.items()):
        frame = pd.read_parquet(path / episode["data_path"])
        phases = frame["expert.phase"].astype(str).tolist()
        index = next(i for i, phase in enumerate(phases) if phase == "approach")
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        rows, valid = select_history_indices(
            timestamps, index, memory_frames, 5, stride
        )
        needed = {0, *rows}
        images = {}
        with av.open(str(path / episode["video_paths"]["front"])) as container:
            for j, decoded in enumerate(container.decode(video=0)):
                if j in needed:
                    images[j] = decoded.to_ndarray(format="rgb24")
                if j >= max(needed):
                    break
        if set(images) != needed:
            raise ValueError("Missing source frames for drawer history response check")
        history = ObservationHistory(memory_frames, stride)
        history.rows = deque(
            (
                float(timestamps[j]),
                np.asarray(frame.iloc[j]["observation.state"]),
                images[j],
            )
            for j in sorted(needed)
        )
        payload = history.payload(task)
        if anchor is None:
            anchor = copy.deepcopy(payload)
        if payload["history_mask"] != anchor["history_mask"]:
            raise ValueError(
                "Drawer presentation windows have different validity masks"
            )
        payload["state"] = copy.deepcopy(anchor["state"])
        payload["memory_images"][-1] = copy.deepcopy(anchor["memory_images"][-1])
        for slot, valid in enumerate(payload["history_mask"]):
            if not valid:
                payload["memory_images"][slot] = copy.deepcopy(
                    anchor["memory_images"][slot]
                )
        payload["flow_seed"] = 0
        await socket.send(json.dumps(payload))
        predicted = np.asarray(json.loads(await socket.recv()), dtype=np.float64)
        if (
            predicted.ndim != 2
            or predicted.shape[1] != 24
            or not len(predicted)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                "Invalid policy actions during drawer history response check"
            )
        expert = np.asarray(frame.iloc[index]["action"])
        cases.append(
            dict(
                target_annotation_only=target,
                seed=episode["seed"],
                expert_first_action=expert.tolist(),
                policy_first_action=predicted[0, :7].tolist(),
                first_translation_error_mm=float(
                    np.linalg.norm(predicted[0, :3] - expert[:3]) * 1000
                ),
                simulation_time=float(timestamps[index]),
                history_sim_times=timestamps[rows].tolist(),
            )
        )
    current_only = copy.deepcopy(anchor)
    current_only["history_mask"] = [False] * (memory_frames - 1) + [True]
    current_only["flow_seed"] = 0
    await socket.send(json.dumps(current_only))
    without_history = np.asarray(json.loads(await socket.recv()), dtype=np.float64)
    if (
        without_history.ndim != 2
        or without_history.shape[1] != 24
        or not len(without_history)
        or not np.isfinite(without_history).all()
    ):
        raise ValueError("Invalid policy actions during current-only drawer check")
    return dict(
        checked=True,
        controlled_counterfactual=True,
        current_image_identical=True,
        all_state_history_identical=True,
        history_mask=anchor["history_mask"],
        task_selection_success_not_measured=True,
        cases=cases,
        current_only_first_action=without_history[0, :7].tolist(),
        maximum_pairwise_first_translation_difference_mm=float(
            max(
                np.linalg.norm(
                    np.asarray(a["policy_first_action"][:3])
                    - b["policy_first_action"][:3]
                )
                * 1000
                for a in cases
                for b in cases
            )
        ),
    )


async def check_policy_contact_precision(datasets, policy_url, report_path):
    """Bounded teacher-forced checks of the first episode of each selected task.

    The policy receives exactly the public image/state/history contract. Phases
    and expert labels select diagnostic samples and measure errors only.
    """
    import av
    import websockets
    from collections import deque
    from mujoco_pickplace.tasks import ObservationHistory
    from mujoco_pickplace.eval_policy_client import request_model_metadata
    from mujoco_pickplace.task_env import rotation_matrix, rotation_vector

    sys.path.insert(0, str(EVO_ROOT))
    from dataset.lerobot_dataset_pretrain_mp import (
        select_history_indices,
        verified_target_history_indices,
    )

    result = dict(policy_url=policy_url, flow_seed=0, teacher_forced=True, tasks=[])
    async with websockets.connect(policy_url, max_size=100_000_000) as socket:
        metadata = await request_model_metadata(socket)
        result["checkpoint_metadata"] = metadata
        memory_frames = int(metadata["memory_frames"])
        stride = float(metadata["memory_stride_seconds"])
        required = {task.key for task, _, _ in datasets}
        if not required.issubset(set(metadata["normalization_keys"])):
            raise ValueError(
                "Policy checkpoint lacks selected task normalization statistics"
            )
        for task, path, info in datasets:
            episode = _read_jsonl(path / "meta/episodes.jsonl")[0]
            frame = pd.read_parquet(path / episode["data_path"])
            phases = frame["expert.phase"].astype(str).tolist()
            certified_visible = (
                verified_target_history_indices(frame, path)
                if task.number == 4
                else set()
            )
            checked_phases = (
                [
                    "approach",
                    "handle_approach",
                    "handle_close",
                    "pull",
                    "safe_wrist_arc",
                    "topdown_orientation",
                    "object_approach",
                    "object_descend",
                    "object_close",
                    "object_lift",
                ]
                if task.number == 4
                else [
                    "approach",
                    "descend",
                    "close",
                    "lift",
                    "transfer" if task.number == 1 else "transport",
                ]
            )
            indices = []
            for phase in checked_phases:
                candidates = [i for i, value in enumerate(phases) if value == phase]
                if task.number == 4 and phase in {
                    "approach",
                    "handle_approach",
                    "handle_close",
                    "pull",
                }:
                    indices.extend(candidates)
                elif candidates:
                    indices.extend(
                        candidates[i]
                        for i in np.unique(
                            np.linspace(
                                0,
                                len(candidates) - 1,
                                min(3, len(candidates)),
                                dtype=int,
                            )
                        )
                    )
            indices = sorted(set(indices))
            timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
            histories = {
                i: select_history_indices(timestamps, i, memory_frames, 5, stride)
                for i in indices
            }
            needed = {0} | {j for rows, _ in histories.values() for j in rows}
            video = path / episode["video_paths"]["front"]
            images = {}
            with av.open(str(video)) as container:
                for index, decoded in enumerate(container.decode(video=0)):
                    if index in needed:
                        images[index] = decoded.to_ndarray(format="rgb24")
                    if index >= max(needed):
                        break
            if needed != set(images):
                raise ValueError(f"Missing requested MP4 frames in {video}")
            predictions, measurements = {}, []
            for index in indices:
                rows, valid = histories[index]
                history = ObservationHistory(memory_frames, stride)
                history.rows = deque(
                    (
                        float(timestamps[j]),
                        np.asarray(frame.iloc[j]["observation.state"]),
                        images[j],
                    )
                    for j in sorted({0, *rows})
                )
                selected_rows, selected_valid = history.selected()
                if [row[0] for row in selected_rows] != timestamps[
                    rows
                ].tolist() or selected_valid != valid:
                    raise AssertionError(
                        "Policy and training disagree on sampled history frames"
                    )
                payload = history.payload(task)
                if payload["history_mask"] != valid:
                    raise AssertionError(
                        "Policy and training disagree on history validity"
                    )
                payload["flow_seed"] = (
                    0  # Diagnostic noise only; evaluation start_seed is unchanged.
                )
                await socket.send(json.dumps(payload))
                actions = np.asarray(json.loads(await socket.recv()), dtype=np.float64)
                if (
                    actions.ndim != 2
                    or not len(actions)
                    or actions.shape[1] != 24
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError(
                        "Policy returned invalid actions during precision check"
                    )
                predicted = actions[0, :7].copy()
                predicted *= np.asarray(task.action_mask)
                expert = np.asarray(frame.iloc[index]["action"])
                predictions[index] = predicted
                difference = predicted - expert
                rotation_error = np.linalg.norm(
                    rotation_vector(
                        rotation_matrix(predicted[3:6]) @ rotation_matrix(expert[3:6]).T
                    )
                )
                measurements.append(
                    dict(
                        frame_index=index,
                        phase=phases[index],
                        simulation_time=float(timestamps[index]),
                        translation_error_mm=float(
                            np.linalg.norm(difference[:3]) * 1000
                        ),
                        translation_error_xyz_mm=(difference[:3] * 1000).tolist(),
                        rotation_error_degrees=float(np.rad2deg(rotation_error)),
                        gripper_absolute_error=float(abs(difference[6])),
                        expert_action=expert.tolist(),
                        policy_action=predicted.tolist(),
                        valid_history_sim_times=[
                            float(timestamps[j]) for j, ok in zip(rows, valid) if ok
                        ],
                        presentation_in_valid_history=any(
                            (phases[j] == "observe_target" or j in certified_visible)
                            and ok
                            for j, ok in zip(rows, valid)
                        ),
                    )
                )
            metrics = {}
            for phase in checked_phases:
                selected = [row for row in measurements if row["phase"] == phase]
                if not selected:
                    continue
                metrics[phase] = {"samples": len(selected)}
                for key in [
                    "translation_error_mm",
                    "rotation_error_degrees",
                    "gripper_absolute_error",
                ]:
                    values = [row[key] for row in selected]
                    metrics[phase][key] = dict(
                        mean=float(np.mean(values)),
                        p95=float(np.percentile(values, 95)),
                        maximum=float(max(values)),
                    )
            task_result = dict(
                task=task.number,
                dataset=str(path),
                seed=episode["seed"],
                episode_index=episode["episode_index"],
                phases=metrics,
                measurements=measurements,
            )
            if task.number == 4:
                task_result["controlled_contact_replay"] = _replay_drawer_contact(
                    task, episode, frame, predictions
                )
                task_result["visual_history_response"] = (
                    await _check_drawer_history_response(
                        socket, task, path, memory_frames, stride
                    )
                )
            result["tasks"].append(task_result)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from mujoco_pickplace.tasks import TASKS

    parser = argparse.ArgumentParser(
        description="Check the MuJoCo task suite and its shared Evo interface."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", type=int, nargs="+", choices=list(TASKS))
    selection.add_argument("--task", type=int, choices=list(TASKS))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Suite data root, or the dataset directory when selecting one task.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--memory-frames", type=int, default=6)
    parser.add_argument("--memory-stride-steps", type=int, default=5)
    parser.add_argument("--memory-stride-seconds", type=float, default=1.0)
    parser.add_argument(
        "--training-cache-dir",
        type=Path,
        default=None,
        help="Optional derived-window cache for this interface check.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Check every episode without loading Evo.",
    )
    parser.add_argument(
        "--require-evo",
        action="store_true",
        help="Fail when Evo interface dependencies are unavailable.",
    )
    parser.add_argument(
        "--require-precision-grasp",
        action="store_true",
        help="Require Task1 precision grasp evidence and new tasks' contact evidence.",
    )
    parser.add_argument(
        "--require-strict-grasp-quality",
        action="store_true",
        help="Require Task1 stable-grasp-v3 evidence and new tasks' strict contact evidence.",
    )
    parser.add_argument(
        "--policy-url",
        default=None,
        help="Optional running EVO server: check contact precision on one recorded episode per selected task.",
    )
    parser.add_argument(
        "--policy-report",
        type=Path,
        default=PROJECT_ROOT / "mujoco_pickplace/outputs/contact_precision.json",
        help="File for phase errors and controlled drawer contact replay; normal terminal output is unchanged.",
    )
    args = parser.parse_args(argv)
    args.tasks = args.tasks or ([args.task] if args.task is not None else list(TASKS))
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Tasks must be unique")
    if (
        min(
            args.image_size,
            args.action_horizon,
            args.memory_frames,
            args.memory_stride_steps,
        )
        < 1
    ):
        parser.error(
            "Image size, horizon, memory frames and stride steps must be positive"
        )
    if not np.isfinite(args.memory_stride_seconds) or args.memory_stride_seconds <= 0:
        parser.error("Memory stride seconds must be finite and positive")
    if args.raw_only and args.require_evo:
        parser.error("--raw-only and --require-evo cannot be combined")
    return args


def main():
    args = parse_args()
    from mujoco_pickplace.tasks import get_task

    datasets = []
    for number in args.tasks:
        task = get_task(number)
        path = (
            task.dataset
            if args.dataset_dir is None
            else (
                args.dataset_dir
                if len(args.tasks) == 1
                else args.dataset_dir / task.dataset.name
            )
        )
        print(f"Checking Task{number}: {path}", flush=True)
        info = check_raw_dataset(
            path,
            require_precision_grasp=args.require_precision_grasp,
            require_strict_grasp_quality=args.require_strict_grasp_quality,
            task_number=number,
        )
        datasets.append((task, path, info))
    if args.policy_url:
        import asyncio

        asyncio.run(
            check_policy_contact_precision(
                datasets, args.policy_url, args.policy_report
            )
        )
    if args.raw_only:
        print("[SKIP] Evo interface check disabled by --raw-only")
        return
    missing = _missing_evo_dependencies()
    if missing and not args.require_evo:
        print(
            "[SKIP] Evo interface check requires: "
            + ", ".join(missing)
            + ". Run in the Evo1 environment for the full interface check."
        )
        return
    check_evo_interface(
        datasets,
        image_size=args.image_size,
        action_horizon=args.action_horizon,
        memory_frames=args.memory_frames,
        memory_stride_steps=args.memory_stride_steps,
        memory_stride_seconds=args.memory_stride_seconds,
        training_cache_dir=args.training_cache_dir,
    )


if __name__ == "__main__":
    main()
