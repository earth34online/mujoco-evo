from pathlib import Path
import json
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_pickplace"
OUT_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "MuJoCo_PickPlace_Dataset"
FPS = 20
TASK = "pick up the blue cube and place it on the green target"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    data_dir = OUT_DIR / "data" / "chunk-000"
    video_dir = OUT_DIR / "videos" / "chunk-000" / "observation.images.image"
    meta_dir = OUT_DIR / "meta"

    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": TASK}])

    episode_rows, episode_stats_rows, all_states, all_actions = [], [], [], []

    npz_files = sorted(RAW_DIR.glob("episode_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No raw episodes found in {RAW_DIR}")

    for npz_path in tqdm(npz_files):
        ep_idx = int(npz_path.stem.split("_")[-1])
        episode_name = f"episode_{ep_idx:06d}"

        raw = np.load(npz_path)
        images = raw["images"]
        states = raw["states"]
        actions = raw["actions"]
        length = len(states)

        df = pd.DataFrame({
            "index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, ep_idx, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / FPS,
            "task_index": np.zeros(length, dtype=np.int64),
            "observation.state": [x.astype(np.float32).tolist() for x in states],
            "action": [x.astype(np.float32).tolist() for x in actions],
        })

        df.to_parquet(data_dir / f"{episode_name}.parquet")
        imageio.mimsave(
            video_dir / f"{episode_name}.mp4",
            images,
            fps=FPS,
            macro_block_size=1,
        )

        episode_rows.append({
            "episode_index": ep_idx,
            "tasks": [TASK],
            "length": int(length),
        })

        episode_stats_rows.append({
            "episode_index": ep_idx,
            "stats": {
                "observation.state": {
                    "min": states.min(axis=0).tolist(),
                    "max": states.max(axis=0).tolist(),
                },
                "action": {
                    "min": actions.min(axis=0).tolist(),
                    "max": actions.max(axis=0).tolist(),
                },
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
            "observation.state": {
                "min": all_states.min(axis=0).tolist(),
                "max": all_states.max(axis=0).tolist(),
            },
            "action": {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
            },
        }, f, indent=2)

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump({
            "fps": FPS,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [10]},
                "action": {"dtype": "float32", "shape": [4]},
                "observation.images.image": {"dtype": "video", "shape": [448, 448, 3]},
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()

