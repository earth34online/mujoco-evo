import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import websockets
from pick_place_env import PickPlaceEnv


SERVER_URL = "ws://127.0.0.1:9000"
PROMPT = "pick up the blue cube and place it on the green target"
NUM_EPISODES = 10
MAX_STEPS = 100
# How many actions of each received chunk to execute before asking the server
# again. Must be <= the model's horizon (10 for the h10 checkpoint; set to the
# model horizon for maximum closed-loop stability, or 1 for tightest feedback).
ACTION_HORIZON = 10
TASK_ID = 1
TASK_NAME = "task1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_VIDEO_DIR = Path("outputs/eval_videos") / RUN_ID
VIDEO_FPS = 20
# Frames rendered per action for a natural-speed video. Each env.step advances
# CONTROL_NSTEP * timestep = 0.2 s of simulation; spreading it over this many
# frames and saving at VIDEO_FPS gives ~real-time playback.
FRAMES_PER_STEP = 4


def obs_to_payload(obs):
    state = obs["state"].astype(np.float32)
    images = [
        obs["image_front"],
        obs["image_overhead"],
        obs["image_wrist"],
    ]
    return {
        # Mirror what the LIBERO client does: MuJoCo/renderer gives RGB, the
        # server decodes with cv2.cvtColor(BGR2RGB), so we send the flipped
        # (BGR) array and let the server restore RGB.
        "image": [image[..., ::-1].astype(np.uint8).tolist() for image in images],
        "image_mask": [1, 1, 1],
        "state": state.astype(float).tolist(),
        "action_mask": [1, 1, 1, 1] + [0] * 20,
        "prompt": PROMPT,
    }


def compose_video_frame(views):
    """Stack camera views side-by-side (LIBERO-style video layout)."""
    return np.hstack(views)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 policy in the MuJoCo Panda7 pick-place env.")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON,
                        help="Actions of each received chunk to execute before re-inferring.")
    parser.add_argument("--smooth", action="store_true", default=True,
                        help="Exponentially smooth the executed position deltas.")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.add_argument("--smooth-alpha", type=float, default=0.6,
                        help="Smoothing factor: 1.0 = no smoothing, lower = more smoothing.")
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
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"Video saved: {path} ({len(frames)} frames)", flush=True)


def apply_gripper_hysteresis(raw_gripper, prev_gripper):
    """Convert the model's noisy gripper into a stable open/close command.

    The flow-matching head outputs a continuous gripper that flaps around 0.5
    (measured mean|dd| ~0.2-0.36), which would otherwise open/close the fingers
    every step. Only switch on a strong signal; otherwise hold the last state.
    """
    if raw_gripper > 0.7:
        return 1.0
    if raw_gripper < 0.3:
        return 0.0
    return prev_gripper




def smooth_positions(raw, prev, alpha=0.5):
    """Light exponential smoothing of the position deltas to damp model jitter.

    ``raw`` is the model's dpos (delta). If ``prev`` is None (first action of the
    episode), the raw delta is used unchanged.
    """
    if prev is None:
        return raw, raw
    smoothed = alpha * raw + (1.0 - alpha) * prev
    return smoothed, smoothed


async def main():
    args = parse_args()
    env = PickPlaceEnv()
    success_count, total_steps = 0, 0
    render_enabled = args.render
    video_root = Path(args.video_dir)

    print(f"========= Start task{TASK_ID}: {PROMPT} =========", flush=True)

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for ep in range(args.num_episodes):
            print(f"\n===== Task {TASK_ID - 1} | Episode {ep + 1} =====", flush=True)
            print(PROMPT, flush=True)

            obs = env.reset(seed=1000 + ep)
            done = False
            executed_steps = 0
            gripper_state = 1.0
            prev_smooth = None  # previous smoothed dpos (delta, not absolute)
            frames = [compose_video_frame([obs["image_front"], obs["image_overhead"], obs["image_wrist"]])]
            render_enabled = maybe_show(frames[0], render_enabled)

            while executed_steps < args.max_steps:
                payload = obs_to_payload(obs)
                print(f"[Step {executed_steps}] Send observation", flush=True)
                await ws.send(json.dumps(payload))

                result = await ws.recv()
                try:
                    action_chunk = np.asarray(json.loads(result), dtype=np.float32)
                    print(
                        f"[Step {executed_steps}] recivied actions "
                        f"(shape={action_chunk.shape}, gripper={float(action_chunk[0][3])})",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"Action parsing failed: {exc}, content: {result}", flush=True)
                    break

                horizon = min(args.horizon, len(action_chunk))
                for action_index in range(horizon):
                    action = action_chunk[action_index, :4].astype(np.float32)
                    # gripper hysteresis: hold last state unless the model is decisive
                    gripper_state = apply_gripper_hysteresis(float(action[3]), gripper_state)
                    action[3] = gripper_state
                    # light position smoothing to damp residual model jitter
                    if args.smooth:
                        action[:3], prev_smooth = smooth_positions(action[:3], prev_smooth, alpha=args.smooth_alpha)

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
                    render_enabled = maybe_show(frames[-1], render_enabled)
                    reward = 1.0 if done else 0.0
                    print(f"[Step {executed_steps}] reward={reward:.2f}, done={done}", flush=True)
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
            result_text = "Success" if done else "Fail"
            print(f"Task {TASK_ID - 1} | Episode {ep + 1}: {result_text}", flush=True)

        print("\n========= Overall Task Summary =========", flush=True)
        print(f"Total Successful Episodes: {success_count}/{args.num_episodes}", flush=True)
        print(f"Average Steps: {total_steps / max(args.num_episodes, 1):.2f}", flush=True)
        print(f"success_rate={success_count / max(args.num_episodes, 1):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
