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

import numpy as np
import websockets
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujoco_pickplace.pick_place_env import PickPlaceEnv

SERVER_URL = "ws://127.0.0.1:9000"
PROMPT = "pick up the blue cube and place it on the green target"
TASK_KEY = "mujoco_pickplace"
NUM_EPISODES = 20
MAX_STEPS = 250
MODEL_ACTION_HORIZON = 14
DEFAULT_EXECUTION_HORIZON = 4
LEGACY_EXECUTION_HORIZON = 14
ACTIVE_ACTION_MASK = [1, 1, 1, 0, 0, 0, 1] + [0] * 17
TASK_ID = 1
TASK_NAME = "task1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_VIDEO_DIR = Path("outputs/eval_videos") / RUN_ID
VIDEO_FPS = 20
FRAMES_PER_STEP = 4
MEMORY_FRAMES = 6
MEMORY_STRIDE_STEPS = 5
PRECISION_REPLAN_Z = (
    PickPlaceEnv.CUBE_SUPPORT_Z + PickPlaceEnv.EXPERT_GRASP_OFFSET + 0.050
)

CKPT_NAME = "Evo1_mujoco_pickplace"
LOG_FILE = f"./log_file/{CKPT_NAME}.txt"
log = logging.getLogger(__name__)


class GripperCommandFilter:
    """Close immediately; debounce only the unsafe open transition."""

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
        self.close_required_steps = 1
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
        transition_steps = (
            self.close_required_steps if requested < 0.5 else self.required_steps
        )
        if self.candidate_steps >= transition_steps:
            self.command = requested
            self.candidate_steps = 0
        return self.command


class PrecisionExecutionController:
    """Replan grasp-sensitive actions without simulator privileged state.

    The controller only reads the policy-visible 8-D proprioception.  It keeps
    the normal action-chunk horizon for free-space motion and replans every
    step near the grasp plane.  A gripper transition is executed at its
    predicted position in the chunk, then ends that chunk so π-MEM receives a
    fresh visual/proprioceptive observation of the changed gripper state.

    Finger opening is deliberately not treated as grasp confirmation.  A
    corner collision and a stable two-pad grasp can produce nearly identical
    finger qpos, so latching either as success can disable the recovery loop
    precisely after a localization error.
    """

    def __init__(self, precision_z=PRECISION_REPLAN_Z):
        self.precision_z = float(precision_z)

    @staticmethod
    def _validate_robot_state(robot_state):
        robot_state = np.asarray(robot_state, dtype=np.float32)
        if robot_state.shape != (8,):
            raise ValueError(f"Expected 8-D robot_state, got {robot_state.shape}")
        if not np.isfinite(robot_state).all():
            raise ValueError("robot_state contains NaN or Inf")
        return robot_state

    def execution_horizon(
        self,
        action_chunk,
        requested_horizon,
        robot_state,
        gripper_filter,
        enabled=True,
    ):
        robot_state = self._validate_robot_state(robot_state)
        prefix = np.asarray(action_chunk, dtype=np.float32)[:requested_horizon]
        if prefix.ndim != 2 or prefix.shape[1] < 7:
            raise ValueError("action_chunk must have shape [horizon, >=7]")

        if not enabled:
            return int(requested_horizon)

        near_grasp_plane = float(robot_state[2]) <= self.precision_z
        if near_grasp_plane:
            return 1

        # Do not collapse to horizon=1 merely because a later action requests
        # closing: doing so repeatedly discards the transition before it can
        # execute.  Run through the first transition and then replan.  The same
        # boundary makes release debounce meaningful across independent model
        # observations instead of letting two open scores from one stale chunk
        # release an object during transport.
        if gripper_filter.command >= 0.5:
            transition_indices = np.flatnonzero(
                prefix[:, 6] <= gripper_filter.close_threshold
            )
        else:
            transition_indices = np.flatnonzero(
                prefix[:, 6] >= gripper_filter.open_threshold
            )
        if transition_indices.size:
            return int(transition_indices[0]) + 1
        return int(requested_horizon)


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate all registered MuJoCo tasks with one Evo-1 checkpoint."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", type=int, choices=(1, 3, 4))
    selection.add_argument("--tasks", type=int, nargs="+", choices=(1, 3, 4))
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Control step limit; omitted: use each task definition.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Actions executed before replanning. Omitted: use the checkpoint "
            "contract for Task1 (14 for legacy Stage1, 4 for π-MEM). "
            "Task3/Task4 require horizon=1 for contact feedback; conflicting overrides fail."
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
            "(default: 2; closing remains immediate as in success_random)."
        ),
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
    parser.add_argument("--render", action="store_true", help="Show the front view.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
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
    args.tasks = args.tasks or ([args.task] if args.task is not None else [1, 3, 4])
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Each task may be selected only once")
    if (
        args.horizon is not None
        and args.horizon != 1
        and any(n in (3, 4) for n in args.tasks)
    ):
        parser.error(
            "Task3/Task4 require --horizon 1; use --task 1 for a different Task1 horizon"
        )
    if args.num_episodes < 1:
        parser.error("--num-episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.horizon is not None and args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.horizon is not None and args.horizon > MODEL_ACTION_HORIZON:
        parser.error(f"--horizon cannot exceed model horizon {MODEL_ACTION_HORIZON}")
    if args.memory_frames < 1:
        parser.error("--memory-frames must be at least 1")
    if args.memory_stride_steps < 1:
        parser.error("--memory-stride-steps must be at least 1")
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
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


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

    configure_logging()
    args = parse_args()
    return await evaluate_tasks(args.tasks, args=args)


if __name__ == "__main__":
    asyncio.run(main())
