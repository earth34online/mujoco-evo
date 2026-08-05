from pathlib import Path
import argparse
import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_panda7_multiview_small"
OUT_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "MuJoCo_Panda7_Multiview_Small"
# One saved frame corresponds to one 0.2-second environment action.
FPS = 5
TASK = "pick up the blue cube and place it on the green target"
ROBOT_TYPE = "panda"
CAMERAS = {
    "front": "images_front",
    "overhead": "images_overhead",
    "wrist": "images_wrist",
}
STATE_NAMES = ["eef_x", "eef_y", "eef_z",
               "cube_x", "cube_y", "cube_z",
               "goal_x", "goal_y", "goal_z",
               "gripper"]
ACTION_NAMES = ["eef_dx", "eef_dy", "eef_dz", "gripper"]
IMAGE_NAMES_21 = ["height", "width", "channel"]
IMAGE_NAMES_30 = ["height", "width", "channels"]
VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def build_feature_map():
    features = {
        "observation.state": {"dtype": "float32", "shape": [10], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [4], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        features[key] = {
            "dtype": "video",
            "shape": [448, 448, 3],
            "names": IMAGE_NAMES_21,
            "video_info": {
                "video.fps": float(FPS),
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def load_raw_episodes(raw_dir):
    npz_files = sorted(raw_dir.glob("episode_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No raw episodes found in {raw_dir}")
    episodes = []
    for npz_path in npz_files:
        ep_idx = int(npz_path.stem.split("_")[-1])
        raw = np.load(npz_path)
        episodes.append({
            "index": ep_idx,
            "name": f"episode_{ep_idx:06d}",
            "states": raw["states"].astype(np.float32),
            "actions": raw["actions"].astype(np.float32),
            "cameras": {cam: raw[raw_key].astype(np.uint8) for cam, raw_key in CAMERAS.items()},
        })
    episodes.sort(key=lambda e: e["index"])
    return episodes

def write_v21(episodes, out_dir):
    data_dir = out_dir / "data" / "chunk-000"
    meta_dir = out_dir / "meta"
    video_dirs = {cam: out_dir / "videos" / "chunk-000" / f"observation.images.{cam}" for cam in CAMERAS}
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for vd in video_dirs.values():
        vd.mkdir(parents=True, exist_ok=True)

    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    episode_rows, episode_stats_rows = [], []
    all_states, all_actions = [], []

    for ep in tqdm(episodes, desc="v2.1 write"):
        ep_idx, name = ep["index"], ep["name"]
        states, actions = ep["states"], ep["actions"]
        length = len(states)

        frame = pd.DataFrame({
            "index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep_idx, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / FPS,
            "task_index": np.zeros(length, dtype=np.int64),
            "observation.state": [v.astype(np.float32).tolist() for v in states],
            "action": [v.astype(np.float32).tolist() for v in actions],
        })
        frame.to_parquet(data_dir / f"{name}.parquet")

        for cam, frames in ep["cameras"].items():
            imageio.mimsave(video_dirs[cam] / f"{name}.mp4", frames, fps=FPS, macro_block_size=1)

        episode_rows.append({"episode_index": ep_idx, "tasks": [TASK], "length": int(length)})
        episode_stats_rows.append({
            "episode_index": ep_idx,
            "stats": {
                "observation.state": {"min": states.min(axis=0).tolist(), "max": states.max(axis=0).tolist()},
                "action": {"min": actions.min(axis=0).tolist(), "max": actions.max(axis=0).tolist()},
            },
        })
        all_states.append(states)
        all_actions.append(actions)

    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_rows)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "observation.state": {"min": all_states.min(axis=0).tolist(), "max": all_states.max(axis=0).tolist()},
            "action": {"min": all_actions.min(axis=0).tolist(), "max": all_actions.max(axis=0).tolist()},
        }, f, indent=2)

    total_frames = sum(len(e["states"]) for e in episodes)
    n_cameras = len(CAMERAS)
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": len(episodes) * n_cameras,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_feature_map(),
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out_dir


def write_v30(episodes, out_dir):
    data_dir = out_dir / "data" / "chunk-000"
    meta_dir = out_dir / "meta"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    # --- concatenated data parquet (all episodes in one file) ---
    global_index = 0
    data_rows = []
    ep_meta_rows = []
    all_states, all_actions = [], []

    for ep in tqdm(episodes, desc="v3.0 write"):
        ep_idx = ep["index"]
        states, actions = ep["states"], ep["actions"]
        length = len(states)
        ds_from, ds_to = global_index, global_index + length
        ep_meta_rows.append({
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": ds_from,
            "dataset_to_index": ds_to,
            "tasks": [TASK],
            "length": int(length),
        })
        n = length
        data_rows.append(pd.DataFrame({
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "episode_index": np.full(n, ep_idx, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": np.arange(n, dtype=np.float32) / FPS,
            "task_index": np.zeros(n, dtype=np.int64),
            "observation.state": [v.astype(np.float32).tolist() for v in states],
            "action": [v.astype(np.float32).tolist() for v in actions],
        }))
        global_index += length
        all_states.append(states)
        all_actions.append(actions)

    pd.concat(data_rows, ignore_index=True).to_parquet(data_dir / "file-000.parquet")

    duration = 0.0
    for cam, frames_key in CAMERAS.items():
        video_dir = out_dir / "videos" / f"observation.images.{cam}" / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        duration = 0.0
        for ep in tqdm(episodes, desc=f"v3.0 concat {cam}"):
            ep_frames = ep["cameras"][cam]
            ep_dur = len(ep_frames) / FPS
            row = next(r for r in ep_meta_rows if r["episode_index"] == ep["index"])
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = 0
            row[f"videos/{cam}/from_timestamp"] = round(duration, 6)
            row[f"videos/{cam}/to_timestamp"] = round(duration + ep_dur, 6)
            duration += ep_dur
            frames.extend(ep_frames)
        imageio.mimsave(video_dir / "file-000.mp4", frames, fps=FPS, macro_block_size=1)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    for ep in episodes:
        s, a = ep["states"], ep["actions"]
        row = next(r for r in ep_meta_rows if r["episode_index"] == ep["index"])
        row["stats/observation.state/min"] = s.min(axis=0).tolist()
        row["stats/observation.state/max"] = s.max(axis=0).tolist()
        row["stats/action/min"] = a.min(axis=0).tolist()
        row["stats/action/max"] = a.max(axis=0).tolist()

    pd.DataFrame(ep_meta_rows).to_parquet(ep_meta_dir / "file-000.parquet")

    # meta/tasks.parquet
    pd.DataFrame({"task_index": [0], "task": [TASK]}).to_parquet(meta_dir / "tasks.parquet")

    # meta/stats.json
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "observation.state": {"min": all_states.min(axis=0).tolist(), "max": all_states.max(axis=0).tolist()},
            "action": {"min": all_actions.min(axis=0).tolist(), "max": all_actions.max(axis=0).tolist()},
        }, f, indent=2)

    features = {
        "observation.state": {"dtype": "float32", "shape": [10], "names": STATE_NAMES, "fps": FPS},
        "action": {"dtype": "float32", "shape": [4], "names": ACTION_NAMES, "fps": FPS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
    }
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        features[key] = {
            "dtype": "video",
            "shape": [448, 448, 3],
            "names": IMAGE_NAMES_30,
            "info": {
                "video.fps": float(FPS),
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "video.height": 448,
                "video.width": 448,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    total_frames = int(global_index)
    info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Convert raw MuJoCo npz episodes to LeRobot datasets.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--format", choices=["v21", "v30", "both"], default="both",
                        help="v21 = LeRobot v2.1 (required by the Evo-1 loader); "
                             "v30 = consolidated LeRobot v3.0; both = write both.")
    args = parser.parse_args()

    episodes = load_raw_episodes(Path(args.raw_dir))
    print(f"Loaded {len(episodes)} raw episodes "
          f"(lengths {min(len(e['states']) for e in episodes)}..{max(len(e['states']) for e in episodes)})")

    if args.format in ("v21", "both"):
        out21 = write_v21(episodes, Path(args.out_dir))
        print(f"v2.1 dataset written to {out21}")

    if args.format in ("v30", "both"):
        out30 = write_v30(episodes, Path(args.out_dir).with_name(Path(args.out_dir).name + "_v3"))
        print(f"v3.0 dataset written to {out30}")


if __name__ == "__main__":
    main()
