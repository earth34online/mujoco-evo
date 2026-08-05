# render_demo_video.py
#
# Roll out the scripted expert and save a natural-speed, three-camera video.
# Useful to eyeball the clean (non-jittery) arm motion after the control fix,
# and to validate the step_video rendering pipeline used by eval_policy_client.
#
# Usage (WSL mujoco env):
#   MUJOCO_GL=egl /home/user/miniconda3/envs/mujoco/bin/python render_demo_video.py
#
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ["MUJOCO_GL"] = "egl"
from pick_place_env import PickPlaceEnv, ScriptedExpertPolicy


def compose(views):
    return np.hstack(views)


def main():
    env = PickPlaceEnv()
    out_dir = Path("outputs") / "demo_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(3):
        expert = ScriptedExpertPolicy(env)
        obs = env.reset(seed=seed)
        frames = [compose([obs["image_front"], obs["image_overhead"], obs["image_wrist"]])]
        done = False
        step = 0
        while not done and step < 140:
            action = expert(obs)
            views, obs, done = env.step_video(action, frames_per_step=4)
            for k in range(len(views["front"])):
                frames.append(compose([views["front"][k], views["overhead"][k], views["wrist"][k]]))
            step += 1
            if done:
                break
        path = out_dir / f"demo_seed{seed}.mp4"
        imageio.mimsave(path, frames, fps=20, macro_block_size=1)
        print(f"demo video saved: {path} ({len(frames)} frames, success={done})", flush=True)


if __name__ == "__main__":
    main()
