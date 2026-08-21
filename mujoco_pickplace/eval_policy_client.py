import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets
from pick_place_env import PickPlaceEnv


SERVER_URL = "ws://127.0.0.1:9000"
PROMPT = "pick up the blue cube and place it on the green target"
NUM_EPISODES = 20
MAX_STEPS = 200
ACTION_HORIZON = 14
ACTIVE_ACTION_MASK = [1, 1, 1, 0, 0, 0, 1] + [0] * 17
TASK_ID = 1
TASK_NAME = "task1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_VIDEO_DIR = Path("outputs/eval_videos") / RUN_ID
VIDEO_FPS = 20
FRAMES_PER_STEP = 4

CKPT_NAME = "Evo1_mujoco_pickplace"
LOG_FILE = f"./log_file/{CKPT_NAME}.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def obs_to_payload(obs):
    state = obs["robot_state"].astype(np.float32)
    if state.shape != (8,):
        raise ValueError(
            "Expected 8-D robot proprioception, got "
            f"{state.shape}"
        )
    front = obs["image_front"]
    blank = np.zeros_like(front)
    images = [front, blank, blank]
    return {
        "image": [image[..., ::-1].astype(np.uint8).tolist() for image in images],
        "image_mask": [1, 0, 0],
        "state": state.astype(float).tolist(),
        "action_mask": ACTIVE_ACTION_MASK,
        "prompt": PROMPT,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 policy in the MuJoCo Panda7 pick-place env.")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON,
                        help="Actions of each received chunk to execute before re-inferring.")
    parser.add_argument("--render", action="store_true", help="Show the front view.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    return parser.parse_args()


def maybe_show(frame, enabled):
    if not enabled:
        return False
    try:
        import cv2

        cv2.imshow("MuJoCo Panda7 PickPlace", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return True
    except Exception as exc:
        print(f"render disabled: {exc}", flush=True)
        return False


def save_video(frames, path, fps=VIDEO_FPS):
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"Video saved: {path} ({len(frames)} frames)", flush=True)


async def main():
    args = parse_args()
    env = PickPlaceEnv()
    success_count, total_steps = 0, 0
    render_enabled = args.render
    video_root = Path(args.video_dir)

    log.info(f"\n========= Start task{TASK_ID}: {PROMPT} =========")

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for ep in range(args.num_episodes):
            print(f"\n===== Task {TASK_ID - 1} | Episode {ep + 1} =====", flush=True)
            print(PROMPT, flush=True)

            obs = env.reset(seed=1000 + ep)
            done = False
            executed_steps = 0
            step = 0
            frames = [obs["image_front"].copy()]
            video_path = video_root / TASK_NAME / f"episode_{ep + 1:03d}.mp4"
            render_enabled = maybe_show(frames[0], render_enabled)

            try:
                while executed_steps < args.max_steps:
                    payload = obs_to_payload(obs)
                    flow_seed = ((1000 + ep) * 10000 + executed_steps)
                    payload["flow_seed"] = int(flow_seed)
                    print(f"[Step {step}] Send observation", flush=True)
                    await ws.send(json.dumps(payload))

                    result = await ws.recv()
                    try:
                        action_chunk = np.asarray(json.loads(result), dtype=np.float32)
                        if (
                            action_chunk.ndim != 2
                            or action_chunk.shape[0] < 1
                            or action_chunk.shape[1] < 7
                        ):
                            raise ValueError(
                                "Expected action chunk shaped [horizon, >=7], got "
                                f"{action_chunk.shape}"
                            )
                        if not np.isfinite(action_chunk).all():
                            raise ValueError("Action chunk contains NaN or Inf")
                        print(f"[Step {step}] recivied actions (shape={action_chunk.shape})", flush=True)
                    except Exception as exc:
                        print(f"Action parsing failed: {exc}, content: {result}", flush=True)
                        break

                    expected_shape = (ACTION_HORIZON, 24, )

                    if action_chunk.shape != expected_shape:
                        raise ValueError(
                                f"Expected action chunk "
                                f"{expected_shape}, "
                                f"got {action_chunk.shape}"
                        )
                    for action_index in range(ACTION_HORIZON):
                        action = np.zeros(7, dtype=np.float32)
                        available = min(7, action_chunk.shape[1])
                        action[:available] = action_chunk[action_index, :available]
                        print(action[:7])
                        action[6] = 1.0 if action[6] >= 0.5 else 0.0
                        print(f"gripper action", action[6])

                        frames_in, obs, done = env.step_video(
                            action,
                            frames_per_step=FRAMES_PER_STEP,
                        )
                        for k in range(len(frames_in["front"])):
                            frames.append(frames_in["front"][k])

                        executed_steps += 1
                        step += 1
                        render_enabled = maybe_show(frames[-1], render_enabled)
                        reward = 1.0 if done else 0.0
                        print(f"[Step {step}] reward={reward:.2f}, done={done}", flush=True)
                        if done or executed_steps >= args.max_steps:
                            break

                    if done:
                        print("Task completed", flush=True)
                        break
            finally:
                save_video(frames, video_path)

            success_count += int(done)
            total_steps += executed_steps
            result_text = "✅ Success" if done else "❌ Fail"
            log.info(f"Task {TASK_ID - 1} | Episode {ep + 1}: {result_text}")

        log.info(f"========= Task {TASK_ID} Summary: {success_count}/{args.num_episodes} Successful =========")
        log.info("\n========= Overall Task Summary =========")
        log.info(f"✅ Total Successful Episodes: {success_count}/{args.num_episodes}")
        log.info(f"📊 Average Steps: {total_steps / max(args.num_episodes, 1):.2f}")
        log.info(f"success_rate={success_count / max(args.num_episodes, 1):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
