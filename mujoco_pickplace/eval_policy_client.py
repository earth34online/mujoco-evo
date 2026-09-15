import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
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
MAX_STEPS = 250
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
PRECISION_REPLAN_Z = (
    PickPlaceEnv.CUBE_SUPPORT_Z
    + PickPlaceEnv.EXPERT_GRASP_OFFSET
    + 0.050
)
GRASP_CONFIRM_FINGER_QPOS = 0.010
GRASP_CONFIRM_STEPS = 2

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


class PrecisionExecutionController:
    """Replan grasp-sensitive actions without simulator privileged state.

    The controller only reads the policy-visible 8-D proprioception.  It keeps
    the normal action-chunk horizon for free-space motion and, once contact is
    supported by finger position, for transport/place.  Before that point it
    replans every step near the grasp plane or while a close command is active,
    so π-MEM receives the newest visual/proprioceptive recovery evidence.
    """

    def __init__(
        self,
        precision_z=PRECISION_REPLAN_Z,
        grasp_finger_qpos=GRASP_CONFIRM_FINGER_QPOS,
        confirm_steps=GRASP_CONFIRM_STEPS,
    ):
        if confirm_steps < 1:
            raise ValueError("confirm_steps must be at least 1")
        self.precision_z = float(precision_z)
        self.grasp_finger_qpos = float(grasp_finger_qpos)
        self.confirm_steps = int(confirm_steps)
        self.contact_steps = 0
        self.grasp_confirmed = False

    @staticmethod
    def _validate_robot_state(robot_state):
        robot_state = np.asarray(robot_state, dtype=np.float32)
        if robot_state.shape != (8,):
            raise ValueError(
                f"Expected 8-D robot_state, got {robot_state.shape}"
            )
        if not np.isfinite(robot_state).all():
            raise ValueError("robot_state contains NaN or Inf")
        return robot_state

    def observe(self, robot_state, gripper_command):
        robot_state = self._validate_robot_state(robot_state)
        fingers = float(np.mean(robot_state[6:8]))
        contact_candidate = (
            float(gripper_command) < 0.5
            and fingers >= self.grasp_finger_qpos
        )
        if contact_candidate:
            self.contact_steps += 1
            if self.contact_steps >= self.confirm_steps:
                self.grasp_confirmed = True
        else:
            self.contact_steps = 0

    def execution_horizon(
        self,
        action_chunk,
        requested_horizon,
        robot_state,
        gripper_filter,
        enabled=True,
    ):
        if not enabled or self.grasp_confirmed:
            return int(requested_horizon)
        robot_state = self._validate_robot_state(robot_state)
        prefix = np.asarray(action_chunk, dtype=np.float32)[:requested_horizon]
        if prefix.ndim != 2 or prefix.shape[1] < 7:
            raise ValueError("action_chunk must have shape [horizon, >=7]")

        near_grasp_plane = float(robot_state[2]) <= self.precision_z
        close_imminent = bool(
            np.any(prefix[:, 6] <= gripper_filter.close_threshold)
        )
        close_active = gripper_filter.command < 0.5
        if near_grasp_plane or close_imminent or close_active:
            return 1
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


def parse_args(argv=None):
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
    parser.add_argument(
        "--precision-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replan every control step near or during the first grasp until "
            "finger proprioception confirms contact (default: enabled)."
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
    args = parser.parse_args(argv)
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
    diagnostics_path = (
        Path(args.diagnostics_jsonl) if args.diagnostics_jsonl else None
    )
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text("", encoding="utf-8")

    start_seed = args.start_seed
    if start_seed is None:
        start_seed = int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]
        )
        log.info("Random evaluation start seed: %s", start_seed)

    log.info(f"\n========= Start task{TASK_ID}: {PROMPT} =========")

    async with websockets.connect(
        args.server_url,
        max_size=100_000_000,
        # Real-model inference blocks the server event loop long enough for
        # the default keepalive timeout to close an otherwise healthy request.
        ping_timeout=None,
    ) as ws:
        for ep in range(args.num_episodes):
            print(f"\n===== Task {TASK_ID - 1} | Episode {ep + 1} =====", flush=True)
            print(PROMPT, flush=True)

            episode_seed = start_seed + ep
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
            precision_controller = PrecisionExecutionController()
            executed_steps = 0
            step = 0
            frames = [obs["image_front"].copy()]
            video_path = video_root / TASK_NAME / f"episode_{ep + 1:03d}.mp4"
            render_enabled = maybe_show(frames[0], render_enabled)
            stall_run = 0
            max_stall_run = 0

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
                    inference_started = time.perf_counter()
                    await ws.send(json.dumps(payload))

                    result = await ws.recv()
                    inference_seconds = time.perf_counter() - inference_started
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
                    execution_horizon = precision_controller.execution_horizon(
                        action_chunk,
                        requested_horizon=args.horizon,
                        robot_state=obs["robot_state"],
                        gripper_filter=gripper_filter,
                        enabled=args.precision_replan,
                    )
                    print(
                        f"[Step {step}] execute horizon={execution_horizon} "
                        f"(grasp_confirmed={precision_controller.grasp_confirmed})",
                        flush=True,
                    )
                    append_diagnostic(
                        diagnostics_path,
                        {
                            "kind": "decision",
                            "episode": ep + 1,
                            "seed": episode_seed,
                            "decision_step": step,
                            "executed_steps": executed_steps,
                            "flow_seed": flow_seed,
                            "inference_seconds": inference_seconds,
                            "history_mask": payload["history_mask"],
                            "execution_horizon": execution_horizon,
                            "grasp_confirmed": precision_controller.grasp_confirmed,
                            "robot_state": np.asarray(
                                obs["robot_state"], dtype=float
                            ).tolist(),
                            "action_chunk_first7": action_chunk[:, :7].astype(float).tolist(),
                            "translation_norms": np.linalg.norm(
                                action_chunk[:, :3], axis=1
                            ).astype(float).tolist(),
                            "gripper_scores": action_chunk[:, 6].astype(float).tolist(),
                        },
                    )
                    for action_index in range(execution_horizon):
                        hand_before = env.data.body("hand").xpos.copy()
                        action = np.zeros(7, dtype=np.float32)
                        available = min(7, action_chunk.shape[1])
                        action[:available] = action_chunk[action_index, :available]
                        raw_action = action.copy()
                        print(action[:7])
                        action[6] = gripper_filter.update(action[6])
                        print(f"gripper action", action[6])

                        frames_in, obs, done = env.step_video(
                            action,
                            frames_per_step=FRAMES_PER_STEP,
                        )
                        precision_controller.observe(
                            obs["robot_state"], action[6]
                        )
                        observation_history.append(snapshot_observation(obs))
                        for k in range(len(frames_in["front"])):
                            frames.append(frames_in["front"][k])

                        executed_steps += 1
                        step += 1
                        hand_after = env.data.body("hand").xpos.copy()
                        cube_after = env.data.body("cube").xpos.copy()
                        hand_motion = float(np.linalg.norm(hand_after - hand_before))
                        cube_target_xy = np.array(
                            [cube_after[0] + env.GRASP_X_BIAS, cube_after[1]],
                            dtype=np.float64,
                        )
                        near_failed_grasp = bool(
                            not env.attached
                            and action[6] < 0.5
                            and hand_after[2] <= PRECISION_REPLAN_Z
                            and np.linalg.norm(hand_after[:2] - cube_target_xy) <= 0.040
                        )
                        stalled_now = bool(
                            near_failed_grasp
                            and hand_motion <= 4e-4
                            and np.linalg.norm(raw_action[:3]) <= 1e-3
                        )
                        stall_run = stall_run + 1 if stalled_now else 0
                        max_stall_run = max(max_stall_run, stall_run)
                        if stall_run == 8:
                            print(
                                "[diagnostic] model has remained at an unconfirmed "
                                "grasp for 8 control steps",
                                flush=True,
                            )
                        append_diagnostic(
                            diagnostics_path,
                            {
                                "kind": "action",
                                "episode": ep + 1,
                                "seed": episode_seed,
                                "step": step,
                                "chunk_action_index": action_index,
                                "raw_action": raw_action.astype(float).tolist(),
                                "applied_action": action.astype(float).tolist(),
                                "hand_position": hand_after.astype(float).tolist(),
                                "cube_position": cube_after.astype(float).tolist(),
                                "finger_qpos": np.asarray(
                                    obs["robot_state"][6:8], dtype=float
                                ).tolist(),
                                "hand_motion": hand_motion,
                                "raw_translation_norm": float(
                                    np.linalg.norm(raw_action[:3])
                                ),
                                "actual_two_pad_contact": bool(
                                    env._has_two_sided_grasp_contact()
                                ),
                                "simulator_attached": bool(env.attached),
                                "proprioceptive_grasp_confirmed": bool(
                                    precision_controller.grasp_confirmed
                                ),
                                "near_failed_grasp": near_failed_grasp,
                                "stall_run": stall_run,
                            },
                        )
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

            append_diagnostic(
                diagnostics_path,
                {
                    "kind": "episode_summary",
                    "episode": ep + 1,
                    "seed": episode_seed,
                    "success": bool(done),
                    "executed_steps": executed_steps,
                    "max_unconfirmed_grasp_stall_steps": max_stall_run,
                    "final_simulator_attached": bool(env.attached),
                    "final_proprioceptive_grasp_confirmed": bool(
                        precision_controller.grasp_confirmed
                    ),
                },
            )

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
