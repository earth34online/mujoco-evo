#use lerobot_dataset_pretrain_mp.py for multithreading load dataset
import os
import sys
import io
import hashlib
import torch
import random
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm  
from typing import List, Union, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToTensor
from collections.abc import Iterable
import multiprocessing as mp
import logging
import pickle
from collections import Counter

CACHE_INDEX_VERSION = 7

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEMORY_EVENT_ROUTINE = "routine"
MEMORY_EVENT_GRASP_ALIGNMENT = "grasp_alignment"
MEMORY_EVENT_POST_FAILURE_CORRECTION = "post_failure_correction"
MEMORY_EVENT_PROBE = "memory_probe"
GRASP_PHASES = {
    "approach", "descend", "close", "handle_approach", "handle_close", "pull",
    "object_descend", "object_close",
}
CONTACT_PRECISION_PHASES = {
    "descend", "close", "handle_approach", "handle_close", "pull",
    "object_descend", "object_close",
}
CORRECTION_PHASES = {"recover", "approach", "descend", "close"}


def build_training_augmentation(image_size, preserve_spatial_calibration=False):
    """Build augmentation while preserving metric image/action alignment.

    The MINT defaults include small crops and rotations.  Those transforms are
    suitable only when the corresponding Cartesian labels are transformed as
    well.  A fixed calibrated camera therefore keeps geometry unchanged and
    uses photometric augmentation only.
    """
    steps = []
    if not preserve_spatial_calibration:
        steps.extend([
            T.RandomResizedCrop(
                image_size,
                scale=(0.95, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomRotation(
                degrees=(-5, 5),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ])
    else:
        steps.append(T.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
        ))
    steps.extend([
        T.ColorJitter(
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08,
        ),
        T.ToTensor(),
    ])
    return T.Compose(steps)


def select_history_indices(
    timestamps,
    current_index: int,
    memory_frames: int,
    memory_stride_steps: int = 1,
    memory_stride_seconds: Union[float, None] = None,
):
    """Select oldest-to-current history indices without crossing an episode.

    Invalid left context is clamped to the first frame and marked false.  When
    usable timestamps and ``memory_stride_seconds`` are supplied, selection is
    based on elapsed time; otherwise it uses explicit frame strides.
    """
    if memory_frames < 1:
        raise ValueError("memory_frames must be at least 1")
    if memory_stride_steps < 1:
        raise ValueError("memory_stride_steps must be at least 1")
    if current_index < 0 or current_index >= len(timestamps):
        raise IndexError(f"current_index {current_index} is outside the episode")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    use_seconds = (
        memory_stride_seconds is not None
        and float(memory_stride_seconds) > 0
        and np.isfinite(timestamps[: current_index + 1]).all()
        and np.all(np.diff(timestamps[: current_index + 1]) >= 0)
    )
    indices, valid = [], []
    for slot in range(memory_frames):
        lag = memory_frames - 1 - slot
        if use_seconds:
            target_time = timestamps[current_index] - lag * float(memory_stride_seconds)
            # Episode timestamps are stored as float32. Preserve the intended
            # control tick when subtraction lands a few ulps below that tick.
            tolerance = max(1e-8, 2 * np.finfo(np.float32).eps * max(1., abs(timestamps[current_index])))
            is_valid = target_time >= timestamps[0] - tolerance
            index = int(
                np.searchsorted(
                    timestamps[: current_index + 1], target_time + tolerance, side="right"
                ) - 1
            )
        else:
            raw_index = current_index - lag * memory_stride_steps
            is_valid = raw_index >= 0
            index = raw_index
        indices.append(max(0, min(current_index, index)))
        valid.append(bool(is_valid))
    valid[-1] = True
    return indices, valid


def classify_memory_event(
    phases, history_indices, current_index, history_valid=None, verified_visible_indices=()
):
    """Label windows whose current action can use visible short-term history.

    A correction window is not defined by a requested dataset quota.  It is
    detected from the expert state sequence: a recovery must already be inside
    the sampled history and the current target must still be part of the retry.
    This mirrors the pi-MEM intervention setup where the failed attempt remains
    in short-term memory while the corrected strategy is supervised.
    """
    if not phases:
        return MEMORY_EVENT_ROUTINE
    if current_index < 0 or current_index >= len(phases):
        raise IndexError(f"current_index {current_index} is outside phase history")
    current_phase = str(phases[current_index])
    if history_valid is None:
        history_valid = [True] * len(history_indices)
    if len(history_valid) != len(history_indices):
        raise ValueError("History indices and validity mask must have equal length")
    visible_phases = {
        str(phases[int(index)])
        for index, valid in zip(history_indices, history_valid)
        if valid and 0 <= int(index) <= current_index
    }
    if current_phase in CORRECTION_PHASES and "recover" in visible_phases:
        return MEMORY_EVENT_POST_FAILURE_CORRECTION
    # Drawer selection uses history during approach. Once the fingers close or
    # pull, contact precision takes precedence even if presentation is visible.
    target_visible = "observe_target" in visible_phases or any(
        valid and int(index) in verified_visible_indices
        for index, valid in zip(history_indices, history_valid)
    )
    if current_phase in {"approach", "handle_approach"} and target_visible:
        return MEMORY_EVENT_PROBE
    if current_phase in GRASP_PHASES:
        # Short contact phases must not be diluted by long approach trajectories.
        # Keep the existing inverse-frequency sampler; these are training labels,
        # never privileged phase inputs to the policy.
        if current_phase in CONTACT_PRECISION_PHASES:
            return f"{MEMORY_EVENT_GRASP_ALIGNMENT}/{current_phase}"
        return MEMORY_EVENT_GRASP_ALIGNMENT
    return MEMORY_EVENT_ROUTINE


def adaptive_memory_event_weights(events):
    """Return inverse-square-root frequency weights without a fixed class quota."""
    events = [str(event) for event in events]
    if not events:
        raise ValueError("events cannot be empty")
    counts = Counter(events)
    largest_class = max(counts.values())
    class_weights = {
        event: float(np.sqrt(largest_class / count))
        for event, count in counts.items()
    }
    weights = np.asarray([class_weights[event] for event in events], dtype=np.float64)
    weights /= float(np.mean(weights))
    return weights, dict(counts)

def compute_lerobot_normalization_stats_from_minmax(jsonl_path):
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Extracting min/max"):
            obj = json.loads(line)
            stats = obj.get("stats", {})
            try:
                state_mins.append(stats["observation.state"]["min"])
                state_maxs.append(stats["observation.state"]["max"])
                action_mins.append(stats["action"]["min"])
                action_maxs.append(stats["action"]["max"])
            except Exception as e:
                print(f"skipping abnormal line: {e}")


    state_min_global = np.min(np.array(state_mins), axis=0).tolist()
    state_max_global = np.max(np.array(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.array(action_mins), axis=0).tolist()
    action_max_global = np.max(np.array(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }

def merge_lerobot_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict:
    state_mins = [np.array(d["observation.state"]["min"]) for d in stats_list]
    state_maxs = [np.array(d["observation.state"]["max"]) for d in stats_list]
    action_mins = [np.array(d["action"]["min"]) for d in stats_list]
    action_maxs = [np.array(d["action"]["max"]) for d in stats_list]
    state_min_global = np.min(np.stack(state_mins), axis=0).tolist()
    state_max_global = np.max(np.stack(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.stack(action_mins), axis=0).tolist()
    action_max_global = np.max(np.stack(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }


def episode_video_timestamps(frame, dataset_path):
    """Map pre-action records to MP4 frames, keeping simulation time separate."""
    metadata_path = Path(dataset_path) / "meta" / "dataset.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") == "mujoco-evo-episodes":
            fps = float(metadata["fps"])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Dataset video fps must be finite and positive")
            indices = frame["frame_index"].to_numpy()
            if not np.array_equal(indices, np.arange(len(frame))):
                raise ValueError("Episode frame indices must be contiguous from zero")
            times = indices.astype(np.float64) / fps
            if "video_timestamp" in frame and not np.allclose(
                frame["video_timestamp"].to_numpy(), times, atol=1e-6, rtol=0
            ):
                raise ValueError("Recorded video timestamps disagree with MP4 frame indices")
            return times
    # Other LeRobot datasets retain their original timestamp convention.
    return (
        frame["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in frame
        else np.arange(len(frame), dtype=np.float64)
    )


def verified_target_history_indices(frame, dataset_path):
    """Recover certified visible history slots, including during drawer closure.

    This annotation only affects sampling. It never enters image/state/action
    tensors, prompts or inference requests, and contains no target drawer ID.
    """
    path = Path(dataset_path) / "meta/episodes.jsonl"
    if not path.is_file() or "episode_index" not in frame:
        return set()
    episode_index = int(frame.iloc[0]["episode_index"])
    times = frame["timestamp"].to_numpy(dtype=np.float64)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode["episode_index"] != episode_index:
            continue
        quality = episode.get("quality", {})
        certified_times = quality.get("sampled_history_sim_times_at_selection", [])
        if not certified_times:
            return set()
        indices = set()
        for slot in quality.get("target_visible_history_slots_at_selection", []):
            if not 0 <= int(slot) < len(certified_times):
                raise ValueError("Target visibility annotation refers to an invalid history slot")
            visible_time = float(certified_times[int(slot)])
            tolerance = max(1e-6, 2 * np.finfo(np.float32).eps * max(1., abs(visible_time)))
            matches = np.flatnonzero(np.abs(times - visible_time) <= tolerance)
            indices.update(int(index) for index in matches)
        return indices
    return set()


def _process_parquet_file_worker(args):
    (
        parquet_path,
        arm_name,
        dataset_name,
        dataset_config,
        dataset_path,
        task_mapping,
        action_horizon,
        max_samples_per_file,
        cache_dir,
        memory_frames,
        memory_stride_steps,
        memory_stride_seconds,
    ) = args
    
    try:
        view_map = dataset_config.get('view_map', None)
        if not view_map:
            logging.info(f"did not find view_map for '{arm_name}-{dataset_name}', use default mapping")
            default_keys = ["image_1", "image_2", "image_3"]
            view_map = {key: f"observation.images.{key}" for key in default_keys}

        source_df = pd.read_parquet(parquet_path)
        if source_df.empty:
            raise ValueError("episode parquet is empty")

        metadata_path = Path(dataset_path) / "meta" / "dataset.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file() else {}
        )
        if metadata.get("format") == "mujoco-evo-episodes":
            episode_path = Path(dataset_path) / "meta/episodes.jsonl"
            records = [
                json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records = [
                row for row in records
                if (Path(dataset_path) / row["data_path"]).resolve() == Path(parquet_path).resolve()
            ]
            if len(records) != 1:
                raise ValueError(f"Native parquet has no unique accepted manifest record: {parquet_path}")
            record = records[0]
            if (len(source_df) != int(record["length"])
                    or not np.all(
                        source_df["episode_index"].to_numpy() == int(record["episode_index"])
                    )
                    or not np.array_equal(source_df["frame_index"].to_numpy(), np.arange(len(source_df)))
                    or not np.all(source_df["task_index"].to_numpy() == int(record["task_index"]))):
                raise ValueError(f"Native parquet rows do not match the accepted episode record: {parquet_path}")

        if action_horizon < 1:
            raise ValueError("action_horizon must be at least 1")

        if "timestamp" in source_df:
            source_timestamps = source_df["timestamp"].to_numpy(dtype=np.float64)
        else:
            source_timestamps = np.arange(len(source_df), dtype=np.float64)

        video_timestamps = episode_video_timestamps(source_df, dataset_path)

        df = source_df
        last_row = df.iloc[-1:]
        padding_count = action_horizon - 1
        if padding_count:
            padding_rows = pd.concat([last_row] * padding_count, ignore_index=True)
            df = pd.concat([df, padding_rows], ignore_index=True)

        source_phases = (
            source_df["expert.phase"].astype(str).tolist()
            if "expert.phase" in source_df
            else []
        )
        sample_indices = list(range(len(source_df)))
        verified_visible_indices = set()
        if metadata_path.is_file():
            if (metadata.get("format") == "mujoco-evo-episodes"
                    and metadata.get("collection_config", {}).get("task_id") == 4):
                # The simulated human owns this interval. Keep its observations
                # in history, but do not train policy actions that evaluation
                # never requests from the model.
                verified_visible_indices = verified_target_history_indices(source_df, dataset_path)
                sample_indices = [
                    i for i in sample_indices
                    if source_phases[i] not in {"observe_target", "presentation_close"}
                ]
        if max_samples_per_file is not None:
            sample_indices = sample_indices[:int(max_samples_per_file)]
        episode_files = []
        episode_events = []
        for i in sample_indices:
            start_idx = i
            end_idx = i + action_horizon

      
            cache_subdir = cache_dir / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem
            cache_filename = f"{start_idx}_{end_idx}.pkl"
            cache_filepath = cache_subdir / cache_filename
            history_indices, history_valid = select_history_indices(
                source_timestamps,
                current_index=i,
                memory_frames=memory_frames,
                memory_stride_steps=memory_stride_steps,
                memory_stride_seconds=memory_stride_seconds,
            )

            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                episode_events.append(
                    classify_memory_event(source_phases, history_indices, i, history_valid, verified_visible_indices)
                    if source_phases
                    else MEMORY_EVENT_ROUTINE
                )
                continue

            logging.info(f"build {cache_filename}")
            sub_df = df.iloc[i: i + action_horizon]
            history_df = source_df.iloc[history_indices]
            memory_event = (
                classify_memory_event(source_phases, history_indices, i, history_valid, verified_visible_indices)
                if source_phases
                else MEMORY_EVENT_ROUTINE
            )
            video_paths = {}
            base_video_path = dataset_path / "videos" / parquet_path.parent.name

            for view_key, view_folder in view_map.items():
                full_path = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
                logging.info(f"full_path {full_path}")
                if full_path.exists():
                    video_paths[view_key] = str(full_path)
                else:
                    raise FileNotFoundError(f"Configured camera video is missing: {full_path}")
            
            
            task_index = sub_df.iloc[0].get("task_index", None)
            if task_index is not None and task_index in task_mapping:
                prompt = task_mapping[task_index]
            else:
                raise ValueError(f"No language instruction for task_index={task_index} in {parquet_path}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty language instruction for task_index={task_index} in {parquet_path}")

            episode = {
                "arm_key": arm_name,
                "dataset_key": dataset_name,
                "prompt": prompt,
                "states": [
                    row.get("observation.state", None)
                    for _, row in history_df.iterrows()
                ],
                "action": [row["action"] for _, row in sub_df.iterrows()],
                "video_paths": video_paths,
                "timestamps": video_timestamps[history_indices].tolist(),
                "simulation_timestamps": source_timestamps[history_indices].tolist(),
                "history_mask": history_valid,
                "memory_event": memory_event,
            }
            
            cache_subdir.mkdir(parents=True, exist_ok=True)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(episode, f)
            
            episode_files.append(str(cache_filepath))
            episode_events.append(memory_event)
        return episode_files, episode_events, None
        
    except Exception as e:
        error_msg = f"Error processing file {parquet_path}: {str(e)}"
        logging.error(error_msg)
        return [], [], error_msg

class LeRobotDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av", # TODO: 
        action_horizon: int = 14,
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,  
        use_augmentation: bool = False,
        overwrite_horizon_cache: bool = False,
        memory_frames: int = 6,
        memory_stride_steps: int = 5,
        memory_stride_seconds: Union[float, None] = 1.0,
    ):
        self.config = config

        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']
        self.active_action_mask = config.get("active_action_mask", None)
        self.dataset_action_masks = {}
        for arm_config in config['data_groups'].values():
            for dataset_name, dataset_config in arm_config.items():
                mask = dataset_config.get('active_action_mask')
                if mask is not None:
                    if len(mask) > self.max_action_dim or not any(mask):
                        raise ValueError(f'Invalid action mask for {dataset_name}')
                    self.dataset_action_masks[dataset_name] = torch.tensor(mask, dtype=torch.bool)
        if self.active_action_mask is not None:
            if len(self.active_action_mask) > self.max_action_dim:
                raise ValueError("active_action_mask cannot be longer than max_action_dim")
            self.active_action_mask = torch.tensor(self.active_action_mask, dtype=torch.bool)

        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation
        self.preserve_spatial_calibration = bool(
            self.config.get("preserve_spatial_calibration", False)
        )
        self.normalization_scope = self.config.get("normalization_scope", "arm")
        if self.normalization_scope not in {"arm", "dataset"}:
            raise ValueError("normalization_scope must be 'arm' or 'dataset'")
        if self.dataset_action_masks and self.normalization_scope != 'dataset':
            raise ValueError('Per-dataset action masks require dataset normalization')
        self.memory_frames = int(memory_frames)
        self.memory_stride_steps = int(memory_stride_steps)
        self.memory_stride_seconds = memory_stride_seconds
        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride_steps < 1:
            raise ValueError("memory_stride_steps must be at least 1")

        cache_name = (
            f"horizon_{action_horizon}_mem_{self.memory_frames}_"
            f"stride_{self.memory_stride_steps}_events_v1"
        )
        if self.memory_stride_seconds is not None:
            seconds_tag = str(float(self.memory_stride_seconds)).replace(".", "p")
            cache_name += f"_seconds_{seconds_tag}"
        cache_namespace = str(config.get("cache_namespace", "")).strip()
        if cache_namespace:
            if not cache_namespace.replace("_", "").isalnum():
                raise ValueError("cache_namespace must be alphanumeric or underscores")
            cache_name += f"_{cache_namespace}"
        self.cache_name = cache_name

        if cache_dir is None:
            self.cache_dir = (
                Path(__file__).resolve().parents[1] / "training_data_cache" / cache_name
            )
        else:
            self.cache_dir = Path(cache_dir)
        if overwrite_horizon_cache and self.cache_dir.exists():
            self._overwrite_horizon_cache(cache_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []  
        self.memory_events = []
        self.arm2stats_dict = {}
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}  

        if self.video_backend == "decord" and not self.video_backend_kwargs:
            self.video_backend_kwargs = {"ctx": "cpu"}  

        self._load_metadata()
        self.source_signature = self._compute_source_signature()
        self._load_trajectories()

        self.basic_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = build_training_augmentation(
            self.image_size,
            preserve_spatial_calibration=self.preserve_spatial_calibration,
        )

    def _overwrite_horizon_cache(self, expected_name: str):
        """Clear the generated cache for the current action horizon.

        This cache contains derived .pkl windows built from parquet/video data.
        It is safe to rebuild and should be overwritten when the MuJoCo dataset
        is recollected in place; otherwise training can silently reuse stale
        horizon_14 samples from an older expert policy.
        """
        cache_path = self.cache_dir.resolve()
        if cache_path.name != expected_name:
            raise ValueError(
                f"Refusing to overwrite cache '{cache_path}': expected directory "
                f"name '{expected_name}'"
            )
        if cache_path.parent.name != "training_data_cache":
            raise ValueError(
                f"Refusing to overwrite cache outside training_data_cache: {cache_path}"
            )
        logging.info("Overwriting generated horizon cache: %s", cache_path)
        shutil.rmtree(cache_path)

    def _load_metadata(self):
     
        self.episodes = []
        self.tasks = {}
        norm_stats_list = []

        # for arms
        for arm_name, arm_config in self.config['data_groups'].items():
            print(f"  -- Processing arm group: '{arm_name}'")

            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                print(f"    -- Processing dataset: '{dataset_name}'")
                print(f"    -- Dataset config: {dataset_config}")
                dataset_tasks = []
                path_str = dataset_config['path']
                dataset_path = Path(path_str)
                required_collection_config = dataset_config.get(
                    "required_collection_config", {}
                )
                required_source_policy_version = dataset_config.get(
                    "required_source_policy_version"
                )
                dataset_metadata = {}
                dataset_metadata_path = dataset_path / "meta" / "dataset.json"
                if dataset_metadata_path.is_file():
                    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
                if required_collection_config or required_source_policy_version:
                    dataset_metadata_path = dataset_path / "meta" / "dataset.json"
                    if not dataset_metadata_path.is_file():
                        raise FileNotFoundError(
                            f"dataset metadata file not found: {dataset_metadata_path}"
                        )
                    if (
                        required_source_policy_version
                        and dataset_metadata.get("source_policy_version")
                        != required_source_policy_version
                    ):
                        raise ValueError(
                            f"{dataset_path} expert policy version must be "
                            f"{required_source_policy_version!r}; got "
                            f"{dataset_metadata.get('source_policy_version')!r}"
                        )
                    actual_collection_config = dataset_metadata.get(
                        "collection_config", {}
                    )
                    mismatches = {}
                    for key, expected in required_collection_config.items():
                        actual = actual_collection_config.get(key)
                        if actual is None:
                            mismatches[key] = {"expected": expected, "actual": None}
                            continue
                        if isinstance(expected, (int, float)) and not isinstance(
                            expected, bool
                        ):
                            matches = bool(np.isclose(float(actual), float(expected)))
                        else:
                            matches = actual == expected
                        if not matches:
                            mismatches[key] = {
                                "expected": expected,
                                "actual": actual,
                            }
                    if mismatches:
                        raise ValueError(
                            f"{dataset_path} collection geometry does not match "
                            f"the training contract: {mismatches}. Recollect the "
                            "dataset with --overwrite instead of mixing geometries."
                        )
                tasks_path = dataset_path / "meta" / "tasks.jsonl"
                if tasks_path.exists():
                    dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict("records")
                    task_index_to_task = {
                        task_obj["task_index"]: task_obj["task"]
                        for task_obj in dataset_tasks
                        if "task_index" in task_obj and "task" in task_obj
                    }
                    self.tasks[arm_name][dataset_name] = task_index_to_task
                else:
                    raise FileNotFoundError(f"tasks file not found: {tasks_path}")
                
                episodes_path = dataset_path / "meta" / "episodes.jsonl"
                if episodes_path.exists():
                    dataset_episodes = pd.read_json(
                        episodes_path, lines=True
                    ).to_dict("records")
                    required_quality_schema = dataset_config.get(
                        "required_quality_schema"
                    )
                    if required_quality_schema:
                        expected_version = dataset_config.get(
                            "required_quality_schema_version"
                        )
                        for episode in dataset_episodes:
                            quality = episode.get("quality") or {}
                            if required_quality_schema == "precision-grasp-stable-v3":
                                project_root = Path(__file__).resolve().parents[3]
                                if str(project_root) not in sys.path:
                                    sys.path.insert(0, str(project_root))
                                from mujoco_pickplace.episode_dataset import validate_precision_grasp_quality
                                try:
                                    validate_precision_grasp_quality(quality, episode.get("episode_index"))
                                    if (episode.get("success") is not True
                                            or quality.get("success") is not True
                                            or int(quality["quality_schema_version"]) < int(expected_version)):
                                        raise AssertionError("Unsuccessful or obsolete Task1 expert episode")
                                except (AssertionError, TypeError, ValueError, KeyError) as exc:
                                    raise ValueError(
                                        f"{dataset_path} episode {episode.get('episode_index')} "
                                        f"fails precision grasp quality: {exc}"
                                    ) from exc
                                continue
                            if required_quality_schema == "mujoco-panda-contact":
                                project_root = Path(__file__).resolve().parents[3]
                                if str(project_root) not in sys.path:
                                    sys.path.insert(0, str(project_root))
                                from mujoco_pickplace.episode_dataset import (
                                    validate_contact_quality,
                                )
                                try:
                                    validate_contact_quality(
                                        quality, required_collection_config.get("task_id")
                                    )
                                except (AssertionError, TypeError, AttributeError) as exc:
                                    raise ValueError(
                                        f"{dataset_path} episode {episode.get('episode_index')} "
                                        f"fails contact quality: {exc}"
                                    ) from exc
                            checks = quality.get("checks") or {}
                            if (
                                not bool(episode.get("success"))
                                or not bool(quality.get("success"))
                                or quality.get("schema") != required_quality_schema
                                or quality.get("schema_version") != expected_version
                                or not checks
                                or quality.get("failed_checks")
                                or not all(value is True for value in checks.values())
                            ):
                                raise ValueError(
                                    f"{dataset_path} episode "
                                    f"{episode.get('episode_index')} fails required "
                                    f"expert quality {required_quality_schema!r}"
                                )
                    self.episodes += dataset_episodes
                    if dataset_metadata.get("format") == "mujoco-evo-episodes":
                        expected_files = []
                        episode_indices = []
                        for episode in dataset_episodes:
                            relative = Path(episode["data_path"])
                            resolved = (dataset_path / relative).resolve()
                            if relative.is_absolute() or not resolved.is_relative_to(dataset_path.resolve()):
                                raise ValueError(f"Native episode path escapes dataset: {relative}")
                            expected_files.append(resolved)
                            episode_indices.append(int(episode["episode_index"]))
                        actual_files = {path.resolve() for path in dataset_path.glob("data/*/*.parquet")}
                        if (not expected_files or len(set(expected_files)) != len(expected_files)
                                or episode_indices != list(range(len(dataset_episodes)))
                                or set(expected_files) != actual_files
                                or dataset_metadata.get("total_episodes") != len(dataset_episodes)
                                or dataset_metadata.get("total_frames") != sum(int(row["length"]) for row in dataset_episodes)):
                            raise ValueError(
                                f"{dataset_path}: native episode manifest does not match the parquet files and dataset totals"
                            )
                elif dataset_config.get("required_quality_schema"):
                    raise FileNotFoundError(
                        f"episode quality records not found: {episodes_path}"
                    )

     
                stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
                stats_path_after_compute = dataset_path / "meta" / "stats.json"
                if stats_path_after_compute.exists():
                    print(f"already have stats file: {stats_path_after_compute}")
                    with open(stats_path_after_compute, "r") as f:
                        stats = json.load(f)
                    norm_arm_list.append(stats)
                elif stats_path.exists():
                    stats = compute_lerobot_normalization_stats_from_minmax(stats_path)
                   
                    with open(stats_path_after_compute, "w") as f:
                        json.dump(stats, f, indent=4)
               
                    print(f"computed stats and saved to: {stats_path_after_compute}")
                    norm_arm_list.append(stats)
                else:
                    raise FileNotFoundError(f"normalization stats file not found: {stats_path}")
                if self.normalization_scope == "dataset":
                    if dataset_name in self.arm2stats_dict:
                        raise ValueError(
                            f"Dataset normalization key is ambiguous: {dataset_name}"
                        )
                    self.arm2stats_dict[dataset_name] = merge_lerobot_stats([stats])
            
            if self.normalization_scope == "dataset":
                stats_targets = [
                    self.arm2stats_dict[dataset_name]
                    for dataset_name in arm_config
                ]
            else:
                stats_targets = [merge_lerobot_stats(norm_arm_list)]
            
            for dataset_name, stats_target in zip(arm_config, stats_targets):
                active_mask = self.dataset_action_masks.get(dataset_name, self.active_action_mask)
                if active_mask is not None:
                    action_min = stats_target["action"]["min"]
                    action_max = stats_target["action"]["max"]
                    for dim in range(len(action_min)):
                        if dim >= len(active_mask) or not bool(active_mask[dim]):
                            # Raw inactive action is zero.
                            # [-1, +1] makes normalized zero exactly zero.
                            action_min[dim] = -1.0
                            action_max[dim] = 1.0

            if self.normalization_scope == "arm":
                self.arm2stats_dict[arm_name] = stats_targets[0]

    def _compute_source_signature(self) -> str:
        """Fingerprint source metadata, parquet and videos using cheap stat data."""
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "video_alignment_version": CACHE_INDEX_VERSION,
                    "window_contract": {
                        "action_horizon": getattr(self, "action_horizon", None),
                        "memory_frames": getattr(self, "memory_frames", None),
                        "memory_stride_steps": getattr(self, "memory_stride_steps", None),
                        "memory_stride_seconds": getattr(self, "memory_stride_seconds", None),
                    },
                    "phase_classification": {
                        "verified_target_visibility": 1,
                        "grasp": sorted(GRASP_PHASES),
                        "contact_precision": sorted(CONTACT_PRECISION_PHASES),
                        "correction": sorted(CORRECTION_PHASES),
                    },
                    "data_config": self.config,
                    "max_samples_per_file": getattr(
                        self, "max_samples_per_file", None
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        for arm_name, arm_config in sorted(self.config["data_groups"].items()):
            for dataset_name, dataset_config in sorted(arm_config.items()):
                dataset_path = Path(dataset_config["path"]).resolve()
                digest.update(f"{arm_name}/{dataset_name}\n".encode("utf-8"))
                candidates = []
                for relative in (
                    "meta/dataset.json",
                    "meta/tasks.jsonl",
                    "meta/episodes.jsonl",
                    "meta/episodes_stats.jsonl",
                    "meta/stats.json",
                ):
                    path = dataset_path / relative
                    if path.is_file():
                        candidates.append(path)
                candidates.extend(dataset_path.glob("data/*/*.parquet"))
                candidates.extend(dataset_path.glob("videos/*/*/*.mp4"))
                for path in sorted(candidates, key=lambda value: str(value)):
                    stat = path.stat()
                    relative = path.relative_to(dataset_path)
                    digest.update(
                        f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                            "utf-8"
                        )
                    )
        return digest.hexdigest()

    def _write_cache_index(self) -> None:
        relative_files = [
            str(Path(path).resolve().relative_to(self.cache_dir.resolve()))
            for path in self.data
        ]
        index_path = self.cache_dir / "cache_index.json"
        temporary_index = index_path.with_suffix(".json.tmp")
        with open(temporary_index, "w", encoding="utf-8") as index_file:
            json.dump(
                {
                    "version": CACHE_INDEX_VERSION,
                    "source_signature": self.source_signature,
                    "files": relative_files,
                    "memory_events": self.memory_events,
                },
                index_file,
                ensure_ascii=False,
            )
        os.replace(temporary_index, index_path)


    def _load_trajectories(self):
        index_path = self.cache_dir / "cache_index.json"
        if index_path.is_file():
            try:
                with open(index_path, "r", encoding="utf-8") as index_file:
                    index_data = json.load(index_file)
                version = index_data.get("version")
                if version != CACHE_INDEX_VERSION:
                    raise ValueError("unsupported cache index version")
                relative_files = index_data["files"]
                self.data = [self.cache_dir / value for value in relative_files]
                self.memory_events = [
                    str(value) for value in index_data["memory_events"]
                ]
                if not self.data:
                    raise ValueError("cache index is empty")
                if len(self.memory_events) != len(self.data):
                    raise ValueError("cache index memory-event count does not match files")
                missing_files = [path for path in self.data if not path.is_file()]
                if missing_files:
                    raise ValueError(
                        f"cache index references {len(missing_files)} missing files"
                    )
                if (
                    index_data.get("source_signature") != self.source_signature
                ):
                    raise ValueError("source dataset fingerprint changed")
                print(
                    f"Loaded {len(self.data)} cached windows from {index_path}"
                )
                return
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logging.warning("Rebuilding invalid cache index %s: %s", index_path, exc)
                self._overwrite_horizon_cache(self.cache_name)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.data = []
                self.memory_events = []
        elif any(self.cache_dir.rglob("*.pkl")):
            logging.warning(
                "Rebuilding unindexed derived cache under %s", self.cache_dir
            )
            self._overwrite_horizon_cache(self.cache_name)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        parquet_process_units = []
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = dataset_config.get('path', None)
                if dataset_path is None:
                    raise ValueError(f"Dataset path for '{arm_name}-{dataset_name}' is not configured, please check the config")
                dataset_path = Path(dataset_path)
                parquet_files = list(dataset_path.glob("data/*/*.parquet"))
                
                task_mapping = self.tasks[arm_name][dataset_name]
                
                for parquet_path in parquet_files:
                    parquet_process_units.append((
                        parquet_path, 
                        arm_name, 
                        dataset_name, 
                        dataset_config, 
                        dataset_path,
                        task_mapping,  
                        self.action_horizon,
                        self.max_samples_per_file,
                        self.cache_dir,
                        self.memory_frames,
                        self.memory_stride_steps,
                        self.memory_stride_seconds,
                    ))

       
        print(f"total {len(parquet_process_units)} parquet files to process")
        if not parquet_process_units:
            raise FileNotFoundError(
                "No episode parquet files matched data/*/*.parquet in the configured datasets"
            )
        
   
        num_processes = min(16, len(parquet_process_units))

        print(f"Using {num_processes} processes for concurrent processing")
        
 
        with mp.Pool(processes=num_processes) as pool:
            
            total_episodes = 0
            failed_files = []
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, episode_events, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                        failed_files.append(error)
                    else:
                        self.data.extend(episode_files)  
                        self.memory_events.extend(episode_events)
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)
        if failed_files:
            raise RuntimeError(
                f"Failed to build {len(failed_files)} episode windows; "
                f"first error: {failed_files[0]}"
            )
        
        print(f"Data processing completed, total {len(self.data)} files generated")
        self._write_cache_index()

    def memory_event_sampling_weights(self):
        # Balance the naturally occurring event types inside each dataset, not
        # just globally.  Otherwise a large Task1 dataset can drown out every
        # Task3 window even though both belong to the same Panda embodiment.
        # This changes no collection quota and duplicates no trajectories.
        labels = []
        cache_root = self.cache_dir.resolve()
        for cache_file, event in zip(self.data, self.memory_events):
            relative = Path(cache_file).resolve().relative_to(cache_root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"Unexpected cache layout for sampling: {cache_file}"
                )
            dataset_key = relative.parts[1]
            labels.append(f"{dataset_key}/{event}")
        weights, counts = adaptive_memory_event_weights(labels)
        return torch.as_tensor(weights, dtype=torch.double), counts


    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):

        source_dim = source_tensor.shape[-1]
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask


    def _load_video_frames(self, video_paths: dict, timestamps) -> List[List[Image.Image]]:
        """Decode all requested timestamps while opening each camera video once."""
        timestamps = [float(value) for value in timestamps]
        if not video_paths or not timestamps:
            raise ValueError("Video paths and requested timestamps must not be empty")
        if not np.isfinite(timestamps).all() or min(timestamps) < 0 or np.any(np.diff(timestamps) < 0):
            raise ValueError("Video timestamps must be finite, nonnegative and ordered")
        frames = [[None for _ in video_paths] for _ in timestamps]
        for view_index, (view, path) in enumerate(video_paths.items()):
            if not os.path.exists(path):
                raise FileNotFoundError(f"video file not found: {path}")
            
            if self.video_backend == "decord":
                import decord

                try:
                    ctx = self.video_backend_kwargs.get("ctx", "cpu")
                    if ctx == "cpu":
                        ctx = decord.cpu(0)
                    elif ctx == "gpu":
                        ctx = decord.gpu(0)
                    logging.info(f"Using video backend {self.video_backend}, context: {ctx}")
                    vr = decord.VideoReader(path, ctx=ctx)
                    logging.info(f"Successfully opened video file: {path}")
                    fps = vr.get_avg_fps()
                    logging.info(f"Video {path} FPS: {fps}")
                    if fps is None or not np.isfinite(fps) or fps <= 0 or not len(vr):
                        raise ValueError(f"Unable to read FPS, video may be corrupted: {path}")

                    for time_index, timestamp in enumerate(timestamps):
                        # Timestamps generated from decimal control periods can
                        # land just below an integer frame index in binary
                        # floating point.  Nearest-frame selection avoids a
                        # systematic one-frame shift into the past.
                        frame_idx = int(round(timestamp * fps))
                        if frame_idx >= len(vr):
                            raise ValueError(
                                f"Requested video frame {frame_idx} at {timestamp:.6f}s is missing from {path}; "
                                f"video has {len(vr)} frames"
                            )
                        frames[time_index][view_index] = Image.fromarray(
                            vr[frame_idx].asnumpy()
                        )

                except Exception as e:
                    logging.info(f"Failed to read video file: {path}")
                    logging.info(f"Error message: {str(e)}")
                    raise

            elif self.video_backend == "av":
                import av
                try:
                    with av.open(path) as container:
                        target_index = 0
                        last_image = None
                        for frame in container.decode(video=0):
                            last_image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                            frame_time = float(frame.time or 0.0)
                            while (
                                target_index < len(timestamps)
                                and frame_time + 1e-6 >= timestamps[target_index]
                            ):
                                frames[target_index][view_index] = last_image.copy()
                                target_index += 1
                        if last_image is None:
                            raise ValueError(f"Video contains no decodable frames: {path}")
                        if target_index < len(timestamps):
                            raise ValueError(
                                f"Requested video timestamp {timestamps[target_index]:.6f}s is missing from {path}; "
                                f"last decoded timestamp is {frame_time:.6f}s"
                            )

                except Exception as e:
                    print(f"Failed to read video file: {path}")
                    print(f"Error message: {str(e)}")
                    raise
            else:
                raise NotImplementedError(f"Video backend {self.video_backend} not implemented")
        
        if any(image is None for timestep in frames for image in timestep):
            raise ValueError("Video decoder did not fill every requested history frame")
        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cache_filepath = self.data[idx]
        
        try:
            with open(cache_filepath, 'rb') as f:
                item = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load dataset cache file {cache_filepath}"
            ) from e
 
        
        arm_key = item["arm_key"]
        dataset_key = item["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]

 
        try:
            frames = self._load_video_frames(item["video_paths"], item["timestamps"])
        except Exception as e:
            raise RuntimeError(
                f"Cannot decode video frames for dataset sample {self.data[idx]}"
            ) from e

        apply_augmentation = self.use_augmentation and random.random() < 0.5
        augmentation_seed = random.randrange(2**31)
        images = []
        for timestep_frames in frames:
            transformed_timestep = []
            for image in timestep_frames:
                if apply_augmentation:
                    # Reset the transform RNG so crop/rotation/color jitter are
                    # identical across time and cannot manufacture fake motion.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(augmentation_seed)
                        transformed = self.aug_transform(image)
                else:
                    transformed = self.basic_transform(image)
                transformed_timestep.append(transformed)
            images.append(transformed_timestep)

        num_real_views = len(images[-1])
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True 

        for timestep_images in images:
            while len(timestep_images) < self.max_views:
                dummy_image = torch.zeros_like(timestep_images[0])
                timestep_images.append(dummy_image)
        images = torch.stack([torch.stack(timestep) for timestep in images])


        if any(state is None for state in item["states"]):
            raise ValueError("missing observation.state, please check data integrity")
        
    

        try:
            stats_key = dataset_key if self.normalization_scope == "dataset" else arm_key
            norm_stats = self.arm2stats_dict[stats_key]
        except KeyError:
        
            raise KeyError(f"Normalization stats not found for arm_key={arm_key} and dataset_key={dataset_key}")

        

        state = torch.tensor(np.stack(item["states"]), dtype=torch.float32)
        device = state.device
        state_min = torch.tensor(norm_stats["observation.state"]["min"], dtype=torch.float32, device=device)
        state_max = torch.tensor(norm_stats["observation.state"]["max"], dtype=torch.float32, device=device)
        
        state = 2 * (state - state_min) / (state_max - state_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)  

        state_padded, state_mask = self._pad_tensor(
            state, self.max_state_dim
        )


        if item["action"] is None:
            raise ValueError("missing action, please check data integrity")

  
        action = torch.from_numpy(np.stack(item["action"])).float()
        device = action.device
        action_min = torch.tensor(norm_stats["action"]["min"], dtype=torch.float32, device=device)
        action_max = torch.tensor(norm_stats["action"]["max"], dtype=torch.float32, device=device)
        action = 2 * (action - action_min.unsqueeze(0)) / (action_max.unsqueeze(0) - action_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)

        action_padded, action_mask = self._pad_tensor(
            action, self.max_action_dim
        )
        active_mask = self.dataset_action_masks.get(dataset_key, self.active_action_mask)
        if active_mask is not None:
            active = torch.zeros(self.max_action_dim, dtype=torch.bool, device=action_mask.device)
            active[:len(active_mask)] = active_mask.to(action_mask.device)
            action_mask = action_mask & active

        prompt = item["prompt"] if item["prompt"] is not None else ""
        
        return {
            "images": images,
            "image_mask": image_mask,
            "prompt": prompt,
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "history_mask": torch.tensor(item["history_mask"], dtype=torch.bool),
            "memory_event": item.get("memory_event", MEMORY_EVENT_ROUTINE),
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long)
        }
