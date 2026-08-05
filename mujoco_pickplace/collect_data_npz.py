from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm

from pick_place_env import PickPlaceEnv, scripted_expert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_panda7_multiview_small"
NUM_EPISODES = 50
MAX_STEPS = 80
MIN_LEN = 10          
MAX_LEN = 120         
MAX_VEL_FLIP_RATIO = 0.25 


def _quality_ok(states: np.ndarray, actions: np.ndarray) -> bool:
    length = len(states)
    if length < MIN_LEN or length > MAX_LEN:
        return False

    vel = np.diff(states[:, :3], axis=0)  
    if length <= 1 or np.linalg.norm(vel, axis=1).max() < 1e-6:
        return False 
    sgn = np.sign(vel[:, 0])
    flips = int((np.diff(sgn) != 0).sum())
    if flips / len(vel) > MAX_VEL_FLIP_RATIO:
        return False

    # No wild action spikes beyond the expert's clip range.
    if np.abs(actions[:, :3]).max() > 0.026 + 1e-6:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Collect scripted pick-and-place demonstrations.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv()
    saved = 0
    seed = 0
    rejected = 0

    with tqdm(total=args.num_episodes) as progress:
        while saved < args.num_episodes:
            obs = env.reset(seed=seed)
            seed += 1
            front_images, overhead_images, wrist_images = [], [], []
            states, actions = [], []
            done = False

            for _ in range(args.max_steps):
                action = scripted_expert(obs)
                front_images.append(obs["image_front"])
                overhead_images.append(obs["image_overhead"])
                wrist_images.append(obs["image_wrist"])
                states.append(obs["state"])
                actions.append(action)

                obs, done = env.step(action)
                if done and len(actions) > MIN_LEN:
                    break

            states = np.asarray(states, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)

            if not done:
                rejected += 1
                continue
            if not _quality_ok(states, actions):
                rejected += 1
                continue

            np.savez_compressed(
                raw_dir / f"episode_{saved:06d}.npz",
                images_front=np.asarray(front_images, dtype=np.uint8),
                images_overhead=np.asarray(overhead_images, dtype=np.uint8),
                images_wrist=np.asarray(wrist_images, dtype=np.uint8),
                states=states,
                actions=actions,
            )
            saved += 1
            progress.update(1)

    print(f"\nCollected {saved} episodes (rejected {rejected}) -> {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
