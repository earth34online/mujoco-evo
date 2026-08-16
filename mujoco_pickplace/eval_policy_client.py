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
MAX_STEPS = 150
ACTION_HORIZON = 14
ACTIVE_ACTION_MASK = [1, 1, 1, 0, 0, 0, 1] + [0] * 17
MAX_POSITION_DELTA = 0.012
MAX_DELTA_CHANGE = 0.004
REVERSAL_DEADBAND = 0.0005
GRIPPER_OPEN_THRESHOLD = 0.8
GRIPPER_CLOSE_THRESHOLD = 0.25
GRIPPER_CLOSE_CONFIRM = 2
GRIPPER_OPEN_CONFIRM = 4
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
    images = [
        obs["image_front"],
        obs["image_overhead"],
        obs["image_wrist"],
    ]
    return {
        "image": [image[..., ::-1].astype(np.uint8).tolist() for image in images],
        "image_mask": [1, 1, 1],
        "state": state.astype(float).tolist(),
        "action_mask": ACTIVE_ACTION_MASK,
        "prompt": PROMPT,
    }


def compose_video_frame(views):
    return np.hstack(views)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 policy in the MuJoCo Panda7 pick-place env.")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON,
                        help="Actions of each received chunk to execute before re-inferring.")
    parser.add_argument("--smooth", action="store_true", default=False,
                        help="Exponentially smooth the executed position deltas.")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.add_argument("--smooth-alpha", type=float, default=0.25,
                        help="Causal action low-pass factor; lower is smoother.")
    parser.add_argument("--max-position-delta", type=float, default=MAX_POSITION_DELTA,
                        help="Per-axis action bound, matched to the collected expert data.")
    parser.add_argument("--max-delta-change", type=float, default=MAX_DELTA_CHANGE,
                        help="Maximum per-axis change between consecutive executed deltas.")
    parser.add_argument("--render", action="store_true", help="Show the composed three-camera view.")
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
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


def update_gripper_state(raw_gripper, prev_gripper, open_count, close_count):
    if raw_gripper > GRIPPER_OPEN_THRESHOLD:
        open_count += 1
        close_count = 0
    elif raw_gripper < GRIPPER_CLOSE_THRESHOLD:
        close_count += 1
        open_count = 0
    else:
        open_count = 0
        close_count = 0

    if prev_gripper > 0.5:
        if close_count >= GRIPPER_CLOSE_CONFIRM:
            prev_gripper = 0.0
    else:
        if open_count >= GRIPPER_OPEN_CONFIRM:
            prev_gripper = 1.0

    return prev_gripper, open_count, close_count



def smooth_positions(
    raw,
    prev,
    alpha=0.25,
    max_position_delta=MAX_POSITION_DELTA,
    max_delta_change=MAX_DELTA_CHANGE,
    reversal_deadband=REVERSAL_DEADBAND,
):
    bounded = np.clip(
        np.asarray(raw, dtype=np.float32),
        -max_position_delta,
        max_position_delta,
    )
    if prev is None:
        prev = np.zeros(3, dtype=np.float32)
    low_pass = alpha * bounded + (1.0 - alpha) * prev
    filtered = prev + np.clip(
        low_pass - prev,
        -max_delta_change,
        max_delta_change,
    )

    prev_norm = np.linalg.norm(prev)
    if prev_norm > reversal_deadband:
        prev_direction = prev / prev_norm
        reverse_component = float(np.dot(filtered, prev_direction))
        if reverse_component < 0.0:
            filtered = filtered - reverse_component * prev_direction

    return filtered.astype(np.float32), filtered.astype(np.float32)


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
            gripper_state = 1.0
            gripper_open_count = 0
            gripper_close_count = 0
            prev_smooth = np.zeros(3, dtype=np.float32)
            frames = [compose_video_frame([obs["image_front"], obs["image_overhead"], obs["image_wrist"]])]
            render_enabled = maybe_show(frames[0], render_enabled)

            while executed_steps < args.max_steps:
                payload = obs_to_payload(obs)
                print(f"[Step {step}] Send observation", flush=True)
                await ws.send(json.dumps(payload))

                result = await ws.recv()
                try:
                    action_chunk = np.asarray(json.loads(result), dtype=np.float32)
                    print(f"[Step {step}] recivied actions (shape={action_chunk.shape})", flush=True)
                except Exception as exc:
                    print(f"Action parsing failed: {exc}, content: {result}", flush=True)
                    break

                horizon = min(args.horizon, len(action_chunk))
                for action_index in range(horizon):
                    action = np.zeros(7, dtype=np.float32)
                    available = min(7, action_chunk.shape[1])
                    action[:available] = action_chunk[action_index, :available]
                    print(action[:7])
                    gripper_state, gripper_open_count, gripper_close_count = update_gripper_state(
                        float(action[6]),
                        gripper_state,
                        gripper_open_count,
                        gripper_close_count,
                    )
                    action[6] = gripper_state
                    print(f"gripper action", action[6])
                    if args.smooth:
                        action[:3], prev_smooth = smooth_positions(
                            action[:3],
                            prev_smooth,
                            alpha=args.smooth_alpha,
                            max_position_delta=args.max_position_delta,
                            max_delta_change=args.max_delta_change,
                        )
                    else:
                        action[:3] = np.clip(
                            action[:3],
                            -args.max_position_delta,
                            args.max_position_delta,
                        )

                    if args.save_video:
                        frames_in, obs, done = env.step_video(action, frames_per_step=FRAMES_PER_STEP)
                        for k in range(len(frames_in["front"])):
                            frames.append(compose_video_frame([
                                frames_in["front"][k],
                                frames_in["overhead"][k],
                                frames_in["wrist"][k],
                            ]))
                    else:
                        obs, done = env.step(action)

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

            if args.save_video:
                video_path = video_root / TASK_NAME / f"episode_{ep + 1:03d}.mp4"
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
