# merge_and_reconvert.py
#
# Merge the second batch of raw episodes into the main raw dir (renumbering to
# keep episode indices unique), then reconvert the combined raw data into a
# fresh LeRobot v2.1 dataset (MuJoCo_Panda7_Multiview_Clean) for retraining.
#
# Run in WSL with the mujoco env:
#   /home/user/miniconda3/envs/mujoco/bin/python merge_and_reconvert.py
#
import glob
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MAIN = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_panda7_multiview_fixed"
RAW_BATCH = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_panda7_multiview_fixed_b"
OUT_CLEAN = PROJECT_ROOT / "Mujoco_training_dataset" / "MuJoCo_Panda7_Multiview_Clean"


def merge():
    existing = sorted(RAW_MAIN.glob("episode_*.npz"))
    next_idx = len(existing)
    batch = sorted(RAW_BATCH.glob("episode_*.npz"))
    print(f"main={len(existing)} batch={len(batch)} -> merged into {RAW_MAIN}")
    for f in batch:
        src_idx = int(f.stem.split("_")[-1])
        dst = RAW_MAIN / f"episode_{next_idx + src_idx:06d}.npz"
        shutil.move(str(f), str(dst))
    print(f"total after merge: {len(list(RAW_MAIN.glob('episode_*.npz')))}")


def reconvert():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_to_evo_lerobot as conv
    episodes = conv.load_raw_episodes(RAW_MAIN)
    print(f"reconverting {len(episodes)} episodes -> {OUT_CLEAN}")
    conv.write_v21(episodes, OUT_CLEAN)
    print(f"done: {OUT_CLEAN}")


if __name__ == "__main__":
    merge()
    reconvert()
