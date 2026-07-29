from pathlib import Path
import numpy as np
from tqdm import trange
from pick_place_env import PickPlaceEnv, scripted_expert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "raw_mujoco_pickplace"
NUM_EPISODES = 100
MAX_STEPS = 300


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv()

    for ep in trange(NUM_EPISODES):
        obs = env.reset(seed=ep)
        images, states, actions = [], [], []

        for _ in range(MAX_STEPS):
            action = scripted_expert(obs)

            images.append(obs["image"])
            states.append(obs["state"])
            actions.append(action)

            obs, done = env.step(action)
            if done and len(actions) > 30:
                break

        np.savez_compressed(
            RAW_DIR / f"episode_{ep:06d}.npz",
            images=np.asarray(images, dtype=np.uint8),
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
        )

if __name__ == "__main__":
    main()

