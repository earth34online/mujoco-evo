import argparse
import asyncio
import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import websockets
from PIL import Image
from pick_place_env import PickPlaceEnv


SERVER_URL = "ws://127.0.0.1:9000"
PROMPT = "pick up the blue cube and place it on the green target"
NUM_EPISODES = 100
MAX_STEPS = 200
MODEL_ACTION_HORIZON = 14
DEFAULT_EXECUTION_HORIZON = 4
ACTIVE_ACTION_MASK = [1, 1, 1, 0, 0, 0, 1] + [0] * 17
TASK_ID = 1
TASK_NAME = "task1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_VIDEO_DIR = Path("outputs/eval_videos") / RUN_ID
VIDEO_FPS = 20
FRAMES_PER_STEP = 4
MEMORY_FRAMES = 6
MEMORY_STRIDE_STEPS = 5

CKPT_NAME = "Evo1_mujoco_pickplace"
LOG_FILE = f"./log_file/{CKPT_NAME}.txt"
log = logging.getLogger(__name__)


class GripperCommandFilter:
    """Apply hysteresis and debounce without reading simulator hidden state."""

    def __init__(
        self,
        required_steps=2,
        close_threshold=0.4,
        open_threshold=0.6,
        initial_command=1.0,
    ):
        if required_steps < 1:
            raise ValueError("required_steps must be at least 1")
        if not 0.0 <= close_threshold < open_threshold <= 1.0:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        self.required_steps = int(required_steps)
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.command = float(initial_command)
        self.candidate = self.command
        self.candidate_steps = 0

    def update(self, score):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError(f"Non-finite gripper score: {score}")
        if score <= self.close_threshold:
            requested = 0.0
        elif score >= self.open_threshold:
            requested = 1.0
        else:
            requested = self.command

        if requested == self.command:
            self.candidate = self.command
            self.candidate_steps = 0
            return self.command
        if requested != self.candidate:
            self.candidate = requested
            self.candidate_steps = 1
        else:
            self.candidate_steps += 1
        if self.candidate_steps >= self.required_steps:
            self.command = requested
            self.candidate_steps = 0
        return self.command


def configure_logging():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(),
        ],
    )


def encode_rgb_jpeg(image, quality=90):
    image = np.asarray(image, dtype=np.uint8)
    with io.BytesIO() as buffer:
        Image.fromarray(image, mode="RGB").save(
            buffer, format="JPEG", quality=int(quality), optimize=True
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def sample_memory_observations(history, memory_frames, stride_steps):
    """Return an oldest-to-current fixed-stride window and validity mask."""
    if memory_frames < 1 or stride_steps < 1:
        raise ValueError("memory_frames and stride_steps must be positive")
    history = list(history)
    if not history:
        raise ValueError("Observation history is empty")
    observations, valid = [], []
    for slot in range(memory_frames):
        lag = (memory_frames - 1 - slot) * stride_steps
        index = len(history) - 1 - lag
        valid.append(index >= 0)
        observations.append(history[max(index, 0)])
    valid[-1] = True
    return observations, valid


def snapshot_observation(obs):
    """Keep immutable memory fields; environments may reuse observation arrays."""
    return {
        "robot_state": np.asarray(obs["robot_state"], dtype=np.float32).copy(),
        "image_front": np.asarray(obs["image_front"], dtype=np.uint8).copy(),
    }


def obs_to_payload(history, memory_frames=MEMORY_FRAMES, stride_steps=MEMORY_STRIDE_STEPS):
    memory, history_mask = sample_memory_observations(
        history, memory_frames, stride_steps
    )
    memory_images = []
    states = []
    first_front = memory[0]["image_front"]
    for obs in memory:
        state = obs["robot_state"].astype(np.float32)
        if state.shape != (8,):
            raise ValueError(
                "Expected 8-D robot proprioception, got "
                f"{state.shape}"
            )
        front = obs["image_front"]
        if front.shape != first_front.shape:
            raise ValueError(
                "All memory images must share one shape, got "
                f"{first_front.shape} and {front.shape}"
            )
        # Send only physical cameras.  Dataset-side fixed-width padding is an
        # internal batching detail and should not consume websocket bandwidth
        # or server JPEG decode time.
        memory_images.append([encode_rgb_jpeg(front)])
        states.append(state.astype(float).tolist())
    return {
        "memory_images": memory_images,
        "image_encoding": "jpeg_base64",
        "image_mask": [1],
        "history_mask": [int(value) for value in history_mask],
        "state": states,
        "action_mask": ACTIVE_ACTION_MASK,
        "prompt": PROMPT,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Evo-1 policy in the MuJoCo Panda7 pick-place env.")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_EXECUTION_HORIZON,
        help=(
            "Actions executed before replanning (default: 4). The model still "
            "predicts 14; a shorter receding horizon reduces open-loop drift."
        ),
    )
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=MEMORY_FRAMES,
        help="Observation count including the current frame (default: 6).",
    )
    parser.add_argument(
        "--memory-stride-steps",
        type=int,
        default=MEMORY_STRIDE_STEPS,
        help="Spacing between memory observations in 5 Hz control steps (default: 5).",
    )
    parser.add_argument(
        "--gripper-debounce-steps",
        type=int,
        default=2,
        help=(
            "Consecutive strong open/close predictions required before the "
            "gripper command changes (default: 2; use 1 to disable debounce)."
        ),
    )
    parser.add_argument("--render", action="store_true", help="Show the front view.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate randomized cube/goal initial positions (default: enabled; "
            "use --no-randomize-task for fixed-task regression)."
        ),
    )
    parser.add_argument(
        "--randomization-scale",
        type=float,
        default=1.0,
        help="Fraction of the full task randomization range (default: 1.0).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=10000,
        help="First evaluation seed; keep it disjoint from collection seeds.",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.horizon > MODEL_ACTION_HORIZON:
        parser.error(
            f"--horizon cannot exceed model horizon {MODEL_ACTION_HORIZON}"
        )
    if args.memory_frames < 1:
        parser.error("--memory-frames must be at least 1")
    if args.memory_stride_steps < 1:
        parser.error("--memory-stride-steps must be at least 1")
    if args.gripper_debounce_steps < 1:
        parser.error("--gripper-debounce-steps must be at least 1")
    return args


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
    configure_logging()
    args = parse_args()
    env = PickPlaceEnv(
        image_size=448,
        randomize_task=args.randomize_task,
        randomization_scale=args.randomization_scale,
    )
    success_count, total_steps = 0, 0
    render_enabled = args.render
    video_root = Path(args.video_dir)

    log.info(f"\n========= Start task{TASK_ID}: {PROMPT} =========")

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for ep in range(args.num_episodes):
            print(f"\n===== Task {TASK_ID - 1} | Episode {ep + 1} =====", flush=True)
            print(PROMPT, flush=True)

            episode_seed = args.start_seed + ep
            obs = env.reset(seed=episode_seed)
            observation_history = deque(
                [snapshot_observation(obs)],
                maxlen=(args.memory_frames - 1) * args.memory_stride_steps + 1,
            )
            print(
                f"seed={episode_seed}, "
                f"cube_xy={env.initial_cube_xy.round(5).tolist()}, "
                f"goal_xy={env.initial_goal_xy.round(5).tolist()}",
                flush=True,
            )
            done = False
            gripper_filter = GripperCommandFilter(
                required_steps=args.gripper_debounce_steps
            )
            executed_steps = 0
            step = 0
            frames = [obs["image_front"].copy()]
            video_path = video_root / TASK_NAME / f"episode_{ep + 1:03d}.mp4"
            render_enabled = maybe_show(frames[0], render_enabled)

            try:
                while executed_steps < args.max_steps:
                    payload = obs_to_payload(
                        observation_history,
                        memory_frames=args.memory_frames,
                        stride_steps=args.memory_stride_steps,
                    )
                    flow_seed = episode_seed * 10000 + executed_steps
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

                    if action_chunk.shape[1] != len(ACTIVE_ACTION_MASK):
                        raise ValueError(
                            "Expected action dimension "
                            f"{len(ACTIVE_ACTION_MASK)}, got {action_chunk.shape[1]}"
                        )
                    if action_chunk.shape[0] < args.horizon:
                        raise ValueError(
                            f"Requested --horizon {args.horizon}, but server returned "
                            f"only {action_chunk.shape[0]} actions"
                        )
                    for action_index in range(args.horizon):
                        action = np.zeros(7, dtype=np.float32)
                        available = min(7, action_chunk.shape[1])
                        action[:available] = action_chunk[action_index, :available]
                        print(action[:7])
                        action[6] = gripper_filter.update(action[6])
                        print(f"gripper action", action[6])

                        frames_in, obs, done = env.step_video(
                            action,
                            frames_per_step=FRAMES_PER_STEP,
                        )
                        observation_history.append(snapshot_observation(obs))
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
