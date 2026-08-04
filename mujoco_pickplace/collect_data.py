from pathlib import Path

import numpy as np
from tqdm import tqdm

from pick_place_env import PickPlaceEnv, scripted_expert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_panda7_multiview_small"
NUM_EPISODES = 30
MAX_STEPS = 60


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv()
    saved = 0
    seed = 0

    with tqdm(total=NUM_EPISODES) as progress:
        while saved < NUM_EPISODES:
            obs = env.reset(seed=seed)
            seed += 1
            front_images, overhead_images, wrist_images = [], [], []
            states, actions = [], []
            done = False

            for _ in range(MAX_STEPS):
                action = scripted_expert(obs)
                front_images.append(obs["image_front"])
                overhead_images.append(obs["image_overhead"])
                wrist_images.append(obs["image_wrist"])
                states.append(obs["state"])
                actions.append(action)

                obs, done = env.step(action)
                if done and len(actions) > 30:
                    break

            if not done:
                continue

            np.savez_compressed(
                RAW_DIR / f"episode_{saved:06d}.npz",
                images_front=np.asarray(front_images, dtype=np.uint8),
                images_overhead=np.asarray(overhead_images, dtype=np.uint8),
                images_wrist=np.asarray(wrist_images, dtype=np.uint8),
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
            )
            saved += 1
            progress.update(1)


if __name__ == "__main__":
    main()
