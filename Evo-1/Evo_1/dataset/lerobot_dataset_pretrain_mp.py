#use lerobot_dataset_pretrain_mp.py for multithreading load dataset
import os
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEMORY_EVENT_ROUTINE = "routine"
MEMORY_EVENT_GRASP_ALIGNMENT = "grasp_alignment"
MEMORY_EVENT_POST_FAILURE_CORRECTION = "post_failure_correction"
GRASP_PHASES = {"approach", "descend", "close"}
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
            is_valid = target_time >= timestamps[0] - 1e-8
            index = int(
                np.searchsorted(
                    timestamps[: current_index + 1], target_time + 1e-8, side="right"
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


def classify_memory_event(phases, history_indices, current_index):
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
    visible_phases = {
        str(phases[int(index)])
        for index in history_indices
        if 0 <= int(index) <= current_index
    }
    if current_phase in CORRECTION_PHASES and "recover" in visible_phases:
        return MEMORY_EVENT_POST_FAILURE_CORRECTION
    if current_phase in GRASP_PHASES:
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

        if action_horizon < 1:
            raise ValueError("action_horizon must be at least 1")

        sample_count = len(source_df)
        if max_samples_per_file is not None:
            sample_count = min(sample_count, int(max_samples_per_file))

        if "timestamp" in source_df:
            source_timestamps = source_df["timestamp"].to_numpy(dtype=np.float64)
        else:
            source_timestamps = np.arange(len(source_df), dtype=np.float64)

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
        episode_files = []
        episode_events = []
        for i in range(sample_count):
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
                    classify_memory_event(source_phases, history_indices, i)
                    if source_phases
                    else MEMORY_EVENT_ROUTINE
                )
                continue

            logging.info(f"build {cache_filename}")
            sub_df = df.iloc[i: i + action_horizon]
            history_df = source_df.iloc[history_indices]
            memory_event = (
                classify_memory_event(source_phases, history_indices, i)
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
                    logging.warning(f"missing video file: {full_path}")
            
            
            task_index = sub_df.iloc[0].get("task_index", None)
            if task_index is not None and task_index in task_mapping:
                prompt = task_mapping[task_index]
            else:
                logging.info(f"cannot find task description from task_index={task_index}")
                prompt = ""

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
                "timestamps": source_timestamps[history_indices].tolist(),
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
                if required_collection_config:
                    dataset_metadata_path = dataset_path / "meta" / "dataset.json"
                    if not dataset_metadata_path.is_file():
                        raise FileNotFoundError(
                            f"dataset metadata file not found: {dataset_metadata_path}"
                        )
                    with open(dataset_metadata_path, "r", encoding="utf-8") as stream:
                        dataset_metadata = json.load(stream)
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
                    self.episodes += pd.read_json(episodes_path, lines=True).to_dict("records")

     
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
            
            merged_states = merge_lerobot_stats(norm_arm_list)
            
            if self.active_action_mask is not None:
                action_min = merged_states["action"]["min"]
                action_max = merged_states["action"]['max']
                native_action_dim = len(action_min)
                
                for dim in range(native_action_dim):
                    if not bool(self.active_action_mask[dim]):
                         # Raw inactive action is zero.
                        # [-1, +1] makes normalized zero exactly zero.
                        action_min[dim] = -1.0
                        action_max[dim] = 1.0
                        
            self.arm2stats_dict[arm_name] = merged_states

    def _compute_source_signature(self) -> str:
        """Fingerprint source metadata, parquet and videos using cheap stat data."""
        digest = hashlib.sha256()
        for arm_name, arm_config in sorted(self.config["data_groups"].items()):
            for dataset_name, dataset_config in sorted(arm_config.items()):
                dataset_path = Path(dataset_config["path"]).resolve()
                digest.update(f"{arm_name}/{dataset_name}\n".encode("utf-8"))
                candidates = []
                for relative in (
                    "meta/tasks.jsonl",
                    "meta/episodes.jsonl",
                    "meta/episodes_stats.jsonl",
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
                    "version": 3,
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
                if version != 3:
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
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, episode_events, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)  
                        self.memory_events.extend(episode_events)
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)
        
        print(f"Data processing completed, total {len(self.data)} files generated")
        self._write_cache_index()

    def memory_event_sampling_weights(self):
        weights, counts = adaptive_memory_event_weights(self.memory_events)
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
                    if fps is None or np.isnan(fps):
                        raise ValueError(f"Unable to read FPS, video may be corrupted: {path}")

                    for time_index, timestamp in enumerate(timestamps):
                        # Timestamps generated from decimal control periods can
                        # land just below an integer frame index in binary
                        # floating point.  Nearest-frame selection avoids a
                        # systematic one-frame shift into the past.
                        frame_idx = min(
                            max(int(round(timestamp * fps)), 0), len(vr) - 1
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
                        while target_index < len(timestamps):
                            frames[target_index][view_index] = last_image.copy()
                            target_index += 1

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
            norm_stats = self.arm2stats_dict[arm_key]
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
        if self.active_action_mask is not None:
            active = torch.zeros(self.max_action_dim, dtype=torch.bool, device=action_mask.device)
            active[:len(self.active_action_mask)] = self.active_action_mask.to(action_mask.device)
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
