import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import websockets
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujoco_pickplace.pick_place_env import PickPlaceEnv
from mujoco_pickplace import release
from mujoco_pickplace.action_execution import (
    AlignedBinaryGripperEnsemble,
    GripperCommandFilter,
    PRECISION_REPLAN_Z,
    PrecisionExecutionController,
)

SERVER_URL = "ws://127.0.0.1:9000"
PROMPT = "pick up the blue cube and place it on the green target"
TASK_KEY = "mujoco_pickplace"
NUM_EPISODES = 10
MAX_STEPS = 200
MODEL_ACTION_HORIZON = 14
DEFAULT_EXECUTION_HORIZON = 4
LEGACY_EXECUTION_HORIZON = 14
ACTIVE_ACTION_MASK = [1, 1, 1, 0, 0, 0, 1] + [0] * 17
TASK_ID = 1
TASK_NAME = "task1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent / "outputs/eval_videos" / RUN_ID
VIDEO_FPS = 20
FRAMES_PER_STEP = 4
MEMORY_FRAMES = 6
MEMORY_STRIDE_STEPS = 5
log = logging.getLogger(__name__)


def configure_logging(output_dir=DEFAULT_VIDEO_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "client.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def append_diagnostic(path, record):
    if path is None:
        return
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def encode_rgb_png(image):
    """Encode an observation losslessly so pixel geometry is not perturbed."""
    image = np.asarray(image, dtype=np.uint8)
    with io.BytesIO() as buffer:
        Image.fromarray(image, mode="RGB").save(buffer, format="PNG", optimize=True)
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


def obs_to_payload(
    history, memory_frames=MEMORY_FRAMES, stride_steps=MEMORY_STRIDE_STEPS
):
    memory, history_mask = sample_memory_observations(
        history, memory_frames, stride_steps
    )
    memory_images = []
    states = []
    first_front = memory[0]["image_front"]
    for obs in memory:
        state = obs["robot_state"].astype(np.float32)
        if state.shape != (8,):
            raise ValueError("Expected 8-D robot proprioception, got " f"{state.shape}")
        front = obs["image_front"]
        if front.shape != first_front.shape:
            raise ValueError(
                "All memory images must share one shape, got "
                f"{first_front.shape} and {front.shape}"
            )
        # Send only physical cameras.  Dataset-side fixed-width padding is an
        # internal batching detail and should not consume websocket bandwidth
        # or server JPEG decode time.
        memory_images.append([encode_rgb_png(front)])
        states.append(state.astype(float).tolist())
    return {
        "memory_images": memory_images,
        "image_encoding": "png_base64",
        "image_mask": [1],
        "history_mask": [int(value) for value in history_mask],
        "state": states,
        "action_mask": ACTIVE_ACTION_MASK,
        "prompt": PROMPT,
        "task_key": TASK_KEY,
    }


def verify_checkpoint_identity(metadata, expected_checkpoint=None, expected_step=None):
    identity = metadata.get("checkpoint_identity") or {}
    path = identity.get("path")
    step = identity.get("step")
    if expected_checkpoint is not None and (
        not path or Path(path).expanduser().resolve() != Path(expected_checkpoint).expanduser().resolve()
    ):
        raise ValueError(f"Wrong server checkpoint: expected {expected_checkpoint}, received {path!r} (step {step!r})")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"Wrong server training step: expected {expected_step}, received {step!r} ({path!r})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate all registered MuJoCo tasks with one Evo-1 checkpoint."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", type=int, choices=(1, 2, 3))
    selection.add_argument("--tasks", type=int, nargs="+", choices=(1, 2, 3))
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--expected-checkpoint", type=Path,
                        help="Require this checkpoint directory on the connected server before evaluating.")
    parser.add_argument("--expected-step", type=int,
                        help="Require this checkpoint training step before evaluating.")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Control step limit; may shorten the Task1/3/4 limits of 200/250/600.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Actions executed before replanning. Omitted: use the checkpoint "
            "contract for Task1 (14 for legacy Stage1, 4 for π-MEM). "
            "Task2/Task3 require horizon=1 for contact feedback; conflicting overrides fail."
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
            "Consecutive strong open predictions required before release "
            "for Task1 (default: 2; closing remains immediate as in success_random). "
            "Task2 and Task3 use immediate binary transitions."
        ),
    )
    parser.add_argument(
        "--task1-gripper-open-threshold",
        type=float,
        default=None,
        help="Task1 open score required by the existing debounce filter (release: 0.45, other checkpoints: 0.6).",
    )
    parser.add_argument(
        "--precision-replan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Replan every control step near the grasp plane; gripper transition "
            "boundaries request a fresh observation. Omitted: disabled for "
            "legacy Stage1 and enabled for π-MEM."
        ),
    )
    parser.add_argument(
        "--task3-gripper-temporal-ensemble", action=argparse.BooleanOptionalAction, default=None,
        help="Combine aligned binary predictions for Task3 gripper commands.",
    )
    parser.add_argument("--render", action="store_true", help="Show the front view.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument(
        "--recovery-observation-steps", type=int, default=0,
        help=("Optional limit of N further steps after a quality failure; "
              "0 continues to physical completion or the task step budget. "
              "Strict failure remains recorded; deep/forceful collisions stop execution."),
    )
    parser.add_argument(
        "--diagnostics-jsonl",
        default=None,
        help=(
            "Optional per-decision/per-action diagnostics. It records model "
            "actions and simulator-only audit evidence but never feeds privileged "
            "state back to the policy."
        ),
    )
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
        default=None,
        help="Optional first evaluation seed; omitted means a random seed per run.",
    )
    raw_args = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(raw_args)
    args.explicit_memory_frames = any(
        a == "--memory-frames" or a.startswith("--memory-frames=") for a in raw_args
    )
    args.explicit_memory_stride = any(
        a == "--memory-stride-steps" or a.startswith("--memory-stride-steps=")
        for a in raw_args
    )
    args.tasks = args.tasks or ([args.task] if args.task is not None else [1, 2, 3])
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Each task may be selected only once")
    if (
        args.horizon is not None
        and args.horizon != 1
        and any(n in (2, 3) for n in args.tasks)
    ):
        parser.error(
            "Task2/Task3 require --horizon 1; use --task 1 for a different Task1 horizon"
        )
    if args.expected_step is not None and args.expected_step < 0:
        parser.error("--expected-step must be nonnegative")
    if args.num_episodes < 1:
        parser.error("--num-episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.recovery_observation_steps < 0:
        parser.error("--recovery-observation-steps must be nonnegative")
    if args.horizon is not None and args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.horizon is not None and args.horizon > MODEL_ACTION_HORIZON:
        parser.error(f"--horizon cannot exceed model horizon {MODEL_ACTION_HORIZON}")
    if args.memory_frames < 1:
        parser.error("--memory-frames must be at least 1")
    if args.memory_stride_steps < 1:
        parser.error("--memory-stride-steps must be at least 1")
    if args.server_url == SERVER_URL and args.expected_checkpoint is None and args.expected_step is None:
        args.expected_checkpoint = release.CHECKPOINT
        args.expected_step = release.TRAINING_STEP
    if args.task1_gripper_open_threshold is not None:
        if not 0.4 < args.task1_gripper_open_threshold <= 1.0:
            parser.error("--task1-gripper-open-threshold must be greater than 0.4 and at most 1")
        if 1 not in args.tasks:
            parser.error("--task1-gripper-open-threshold requires Task1")
    if args.task3_gripper_temporal_ensemble and 3 not in args.tasks:
        parser.error("--task3-gripper-temporal-ensemble requires Task3")
    if args.gripper_debounce_steps < 1:
        parser.error("--gripper-debounce-steps must be at least 1")
    return args


async def request_model_metadata(websocket):
    await websocket.send(json.dumps({"request": "model_metadata"}))
    metadata = json.loads(await websocket.recv())
    if not isinstance(metadata, dict) or metadata.get("type") != "model_metadata":
        raise RuntimeError("Server did not return valid model metadata")
    return metadata


def resolve_execution_policy(args, metadata):
    legacy = bool(metadata.get("legacy_inference_contract", False))
    recommended_horizon = int(
        metadata.get(
            "recommended_execution_horizon",
            LEGACY_EXECUTION_HORIZON if legacy else DEFAULT_EXECUTION_HORIZON,
        )
    )
    recommended_precision = bool(
        metadata.get("recommended_precision_replan", not legacy)
    )
    horizon = recommended_horizon if args.horizon is None else int(args.horizon)
    precision_replan = (
        recommended_precision
        if args.precision_replan is None
        else bool(args.precision_replan)
    )
    if not 1 <= horizon <= MODEL_ACTION_HORIZON:
        raise ValueError(
            f"Resolved execution horizon must be in 1..{MODEL_ACTION_HORIZON}, got {horizon}"
        )
    return horizon, precision_replan


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
    from mujoco_pickplace.episode_dataset import write_rgb_video

    write_rgb_video(path, frames, fps)


def clear_episode_videos(video_root, task_name=TASK_NAME):
    """Remove stale episode videos for this task before a new evaluation."""
    task_dir = Path(video_root) / task_name
    if not task_dir.is_dir():
        return 0

    removed = 0
    for video_path in task_dir.glob("episode_*.mp4"):
        if video_path.is_file():
            video_path.unlink()
            removed += 1
    return removed


async def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mujoco_pickplace.tasks import evaluate_tasks

    args = parse_args()
    configure_logging(args.video_dir)
    return await evaluate_tasks(args.tasks, args=args)


if __name__ == "__main__":
    asyncio.run(main())
