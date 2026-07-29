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
MAX_STEPS = 300
ACTION_HORIZON = 1


def obs_to_payload(obs):
    image = obs["image"].astype(np.uint8)
    zero_image = np.zeros_like(image)
    state = obs["state"].astype(np.float32)

    return {
        "image": [
            image[..., ::-1].tolist(),
            zero_image.tolist(),
            zero_image.tolist(),
        ],
        "image_mask": [1, 0, 0],
        "state": state.astype(float).tolist(),
        "action_mask": [1, 1, 1, 1] + [0] * 20,
        "prompt": PROMPT,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 policy in the MuJoCo pick-place env.")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--render", action="store_true", help="Show a realtime OpenCV window.")
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True, help="Save each rollout as an mp4 video (default).")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="Disable rollout video saving.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    return parser.parse_args()


def maybe_show(frame, enabled):
    if not enabled:
        return False
    try:
        import cv2

        cv2.imshow("MuJoCo PickPlace", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return True
    except Exception as exc:
        print(f"render disabled: {exc}", flush=True)
        return False


def save_video(frames, path, fps=20):
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved: {path} ({len(frames)} frames)", flush=True)


async def main():
    args = parse_args()
    env = PickPlaceEnv()
    success_count, total_steps = 0, 0
    render_enabled = args.render
    video_root = Path(args.video_dir)

    print(f"========= Start {TASK_NAME}: {PROMPT} =========", flush=True)
    print(f"Video root: {video_root.resolve()}", flush=True)

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for ep in range(args.num_episodes):
            print(f"\n===== Task 0 | Episode {ep + 1} =====", flush=True)
            print(PROMPT, flush=True)

            obs = env.reset(seed=1000 + ep)
            done = False
            executed_steps = 0
            frames = [obs["image"]]
            render_enabled = maybe_show(obs["image"], render_enabled)

            while executed_steps < args.max_steps:
                payload = obs_to_payload(obs)
                print(f"[Step {executed_steps}] Send observation", flush=True)
                await ws.send(json.dumps(payload))

                result = await ws.recv()
                try:
                    action_list = json.loads(result)
                    action_chunk = np.asarray(action_list, dtype=np.float32)
                    print(f"[Step {executed_steps}] recivied actions (shape={action_chunk.shape}, gripper={float(action_chunk[0][3])})", flush=True)
                except Exception as exc:
                    print(f"Action parsing failed: {exc}, content: {result}", flush=True)
                    break

                action = action_chunk[0, :4].astype(np.float32)
                print(action.astype(float).tolist(), flush=True)
                print("gripper action", float(action[3]), flush=True)

                obs, done = env.step(action)
                executed_steps += 1
                reward = 1.0 if done else 0.0
                frames.append(obs["image"])
                render_enabled = maybe_show(obs["image"], render_enabled)
                print(f"[Step {executed_steps}] reward={reward:.2f}, done={done}", flush=True)

                if done:
                    print("Task completed", flush=True)
                    break

            if args.save_video:
                video_path = video_root / TASK_NAME / f"episode_{ep + 1:03d}.mp4"
                save_video(frames, video_path)

            success_count += int(done)
            total_steps += executed_steps
            if done:
                print(f"Task 0 | Episode {ep + 1}: Success", flush=True)
            else:
                print(f"Task 0 | Episode {ep + 1}: Fail", flush=True)

        print("\n========= Overall Task Summary =========", flush=True)
        print(f"Total Successful Episodes: {success_count}/{args.num_episodes}", flush=True)
        print(f"Average Steps: {total_steps / max(args.num_episodes, 1):.2f}", flush=True)
        print(f"success_rate={success_count / args.num_episodes:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())