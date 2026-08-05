import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import imageio.v2 as imageio
import numpy as np

from pick_place_env import PickPlaceEnv, scripted_expert


OUTPUT = Path("outputs/demo_videos/expert_libero_wrist_small_palm_seed0.mp4")


def compose(views):
    return np.hstack(views)


def main():
    env = PickPlaceEnv()
    obs = env.reset(seed=0)
    frames = [compose([obs["image_front"], obs["image_overhead"], obs["image_wrist"]])]
    had_two_pad_contact = False
    done = False

    for step in range(100):
        views, obs, done = env.step_video(scripted_expert(obs), frames_per_step=4)
        had_two_pad_contact |= env.attached
        for index in range(len(views["front"])):
            frames.append(compose([
                views["front"][index],
                views["overhead"][index],
                views["wrist"][index],
            ]))
        if done:
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUTPUT, frames, fps=20, macro_block_size=1)
    env.renderer.close()
    print(
        f"path={OUTPUT.resolve()} success={done} steps={step + 1} "
        f"two_pad_contact={had_two_pad_contact} frames={len(frames)} "
        f"shape={frames[0].shape}",
        flush=True,
    )
    if not done or not had_two_pad_contact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
