from pathlib import Path
import argparse
import shutil

import numpy as np
from tqdm import tqdm

from episode_dataset import CAMERAS, EpisodeDatasetWriter
from pick_place_env import PickPlaceEnv, ScriptedExpertPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Mujoco_training_dataset" / "cache" / "mujoco_pickplace"
NUM_EPISODES = 250
MAX_ATTEMPTS = 20000
MAX_STEPS = 250
MIN_LEN = 20
MAX_ACTION_REVERSALS = 4
MAX_EEF_REVERSALS = 8
MAX_ACTION_JUMP = 0.020
MAX_EEF_STEP = 0.025
RECOVERY_EEF_REVERSAL_ALLOWANCE = 4
RECOVERY_ACTION_JUMP_LIMIT = 0.025
POST_SUCCESS_STEPS = 24


def _reversal_count(vectors, motion_epsilon=1e-5):
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) < 2:
        return 0
    moving = np.linalg.norm(vectors, axis=1) > motion_epsilon
    valid = moving[1:] & moving[:-1]
    dots = np.sum(vectors[1:] * vectors[:-1], axis=1)
    return int(np.sum(valid & (dots < -motion_epsilon ** 2)))


def _phase_entry_count(phases, target):
    return sum(
        left != target and right == target
        for left, right in zip(phases[:-1], phases[1:])
    )


def trajectory_quality(
    states,
    actions,
    phases,
    joint_targets,
    had_two_pad_contact,
):
    phases = list(phases)
    eef_delta = np.diff(states[:, :3], axis=0)
    action_delta = np.diff(actions[:, :3], axis=0)
    recovery_count = _phase_entry_count(phases, "recover")

    lift_indices = np.flatnonzero(np.asarray(phases) == "lift")
    successful_close_index = None
    if len(lift_indices):
        close_before_lift = np.flatnonzero(
            (np.asarray(phases) == "close")
            & (np.arange(len(phases)) < int(lift_indices[0]))
        )
        if len(close_before_lift):
            successful_close_index = int(close_before_lift[-1])
            while (
                successful_close_index > 0
                and phases[successful_close_index - 1] == "close"
            ):
                successful_close_index -= 1

    successful_close_xy_error = np.inf
    successful_close_z_above = np.inf
    if successful_close_index is not None:
        grasp_state = states[successful_close_index]
        hand = grasp_state[:3]
        cube = grasp_state[3:6]
        target_xy = np.array(
            [cube[0] + PickPlaceEnv.GRASP_X_BIAS, cube[1]],
            dtype=np.float64,
        )
        successful_close_xy_error = float(
            np.linalg.norm(hand[:2] - target_xy)
        )
        successful_close_z_above = float(
            hand[2]
            - (
                PickPlaceEnv.CUBE_SUPPORT_Z
                + PickPlaceEnv.EXPERT_GRASP_OFFSET
            )
        )

    allowed_eef_reversals = (
        MAX_EEF_REVERSALS
        + recovery_count * RECOVERY_EEF_REVERSAL_ALLOWANCE
    )
    allowed_action_jump = (
        RECOVERY_ACTION_JUMP_LIMIT
        if recovery_count
        else MAX_ACTION_JUMP
    )
    metrics = {
        "two_pad_contact": bool(had_two_pad_contact),
        "recovery_count": int(recovery_count),
        "successful_close_xy_error": successful_close_xy_error,
        "successful_close_z_above": successful_close_z_above,
        "action_reversals": _reversal_count(actions[:, :3]),
        "eef_reversals": _reversal_count(eef_delta),
        "allowed_eef_reversals": int(allowed_eef_reversals),
        "max_action_jump": float(
            np.max(np.linalg.norm(action_delta, axis=1))
        ) if len(action_delta) else 0.0,
        "allowed_action_jump": float(allowed_action_jump),
        "max_eef_step": float(
            np.max(np.linalg.norm(eef_delta, axis=1))
        ) if len(eef_delta) else 0.0,
        "max_joint_target_delta": float(
            np.max(np.abs(np.diff(joint_targets, axis=0)))
        ) if len(joint_targets) > 1 else 0.0,
        "robot_table_contact": False,
    }
    accepted = (
        len(states) >= MIN_LEN
        and had_two_pad_contact
        and successful_close_xy_error <= PickPlaceEnv.GRASP_CLOSE_XY_TOL + 1e-8
        and successful_close_z_above >= -PickPlaceEnv.GRASP_CLOSE_Z_LOWER_TOL - 1e-8
        and successful_close_z_above <= PickPlaceEnv.GRASP_CLOSE_Z_TOL + 1e-8
        and metrics["action_reversals"] <= MAX_ACTION_REVERSALS
        # A real retry necessarily adds down/up/down direction changes and a
        # phase-boundary command change.  Account only for recorded recoveries;
        # the strict direct-trajectory limits remain unchanged.
        and metrics["eef_reversals"] <= allowed_eef_reversals
        and metrics["max_action_jump"] <= allowed_action_jump
        and metrics["max_eef_step"] <= MAX_EEF_STEP
        and metrics["max_joint_target_delta"] <= PickPlaceEnv.MAX_JOINT_TARGET_DELTA + 1e-8
    )
    return accepted, metrics


def remove_redundant_static_frames(trajectory):
    states = np.asarray(trajectory["states"])
    actions = np.asarray(trajectory["actions"])
    phases = trajectory["phases"]
    dones = trajectory["dones"]
    keep = [0]
    protected_phases = {"close", "release", "settle"}

    for index in range(1, len(states)):
        previous = keep[-1]
        static = (
            index != len(states) - 1
            and phases[index] not in protected_phases
            and phases[index] == phases[previous]
            and np.linalg.norm(states[index] - states[previous]) < 2e-5
            and np.linalg.norm(actions[index, :3]) < 5e-5
            and abs(float(actions[index, -1] - actions[previous, -1])) < 1e-6
            and not dones[index]
        )
        if not static:
            keep.append(index)

    compact = {
        key: [values[index] for index in keep]
        for key, values in trajectory.items()
        if key != "images"
    }
    compact["images"] = {
        camera: [trajectory["images"][camera][index] for index in keep]
        for camera in CAMERAS
    }
    return compact, len(states) - len(keep)


def collect_attempt(env, seed, max_steps):
    obs = env.reset(seed=seed)
    expert = ScriptedExpertPolicy(env)
    trajectory = {
        "states": [],
        "robot_states": [],
        "actions": [],
        "phases": [],
        "dones": [],
        "timestamps": [],
        "joint_targets": [],
        "images": {camera: [] for camera in CAMERAS},
    }
    done = False
    task_succeeded = False
    post_success_remaining = 0
    had_two_pad_contact = False

    control_period = env.model.opt.timestep * env.CONTROL_NSTEP
    for step_index in range(max_steps):
        action = expert(obs)
        phase = expert.phase
        trajectory["states"].append(obs["state"].copy())
        trajectory["robot_states"].append(obs["robot_state"].copy())
        trajectory["actions"].append(action.copy())
        trajectory["phases"].append(phase)
        trajectory["timestamps"].append(step_index * control_period)
        for camera in CAMERAS:
            trajectory["images"][camera].append(obs[f"image_{camera}"].copy())

        obs, done = env.step(action)
        had_two_pad_contact |= env.attached
        trajectory["joint_targets"].append(env.arm_target.copy())
        trajectory["dones"].append(False)
        if done and not task_succeeded:
            task_succeeded = True
            post_success_remaining = POST_SUCCESS_STEPS
        elif task_succeeded:
            post_success_remaining -= 1
            if post_success_remaining <= 0:
                break

    if trajectory["dones"]:
        trajectory["dones"][-1] = task_succeeded

    arrays = {
        "states": np.asarray(trajectory["states"], dtype=np.float32),
        "actions": np.asarray(trajectory["actions"], dtype=np.float32),
        "joint_targets": np.asarray(trajectory["joint_targets"], dtype=np.float64),
    }
    accepted, quality = trajectory_quality(
        arrays["states"],
        arrays["actions"],
        trajectory["phases"],
        arrays["joint_targets"],
        had_two_pad_contact,
    )
    quality["initial_cube_xy"] = env.initial_cube_xy.astype(float).tolist()
    quality["initial_goal_xy"] = env.initial_goal_xy.astype(float).tolist()
    quality["randomize_task"] = bool(env.randomize_task)
    quality["randomization_scale"] = float(env.randomization_scale)
    quality["robot_table_contact"] = bool(env.unsafe_robot_table_contact)
    accepted = bool(accepted and not env.unsafe_robot_table_contact)
    accepted = bool(task_succeeded and accepted)
    quality["success"] = bool(task_succeeded)
    quality["raw_length"] = len(trajectory["states"])
    return trajectory, accepted, quality


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Collect smooth contact-aware demonstrations in the project dataset format."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Randomize cube and goal initial XY positions for each episode "
            "(default: enabled; use --no-randomize-task for fixed-task regression)."
        ),
    )
    parser.add_argument(
        "--randomization-scale",
        type=float,
        default=1.0,
        help="Fraction of the full task randomization range (default: 1.0).",
    )
    parser.add_argument(
        "--compact-static-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove redundant static frames before writing (default: disabled). "
            "π-MEM training should keep the dense 5 Hz timeline."
        ),
    )
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Clear the existing data/videos/meta under dataset-dir before "
             "collecting, then write a fresh dataset starting at episode 0 "
             "(default).",
    )
    write_mode.add_argument(
        "--append",
        dest="overwrite",
        action="store_false",
        help="Preserve the existing dataset and append new episodes. This "
             "non-default behavior must be requested explicitly.",
    )
    return parser


def parse_args(argv=None):
    parser = build_argument_parser()
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.overwrite:
        removed = []
        for sub in ("data", "videos", "meta", "mujoco_pickplace"):
            target = args.dataset_dir / sub
            if target.exists():
                shutil.rmtree(target)
                removed.append(sub)
        print(
            f"Overwriting dataset: cleared {removed or 'nothing'} under "
            f"{args.dataset_dir.resolve()}",
            flush=True,
        )

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(args.num_episodes * 20, args.num_episodes)

    env = PickPlaceEnv(
        image_size=args.image_size,
        randomize_task=args.randomize_task,
        randomization_scale=args.randomization_scale,
    )
    print(
        "Task initialization: "
        f"randomize_task={env.randomize_task}, "
        f"randomization_scale={env.randomization_scale}, "
        f"start_seed={args.start_seed}",
        flush=True,
    )
    action_period = env.model.opt.timestep * env.CONTROL_NSTEP
    writer = EpisodeDatasetWriter(
        args.dataset_dir,
        fps=1.0 / action_period,
        image_size=args.image_size,
        collection_config={
            "randomize_task": bool(args.randomize_task),
            "randomization_scale": float(args.randomization_scale),
            "compact_static_frames": bool(args.compact_static_frames),
        },
    )
    saved = 0
    rejected = 0
    seed = args.start_seed

    try:
        with tqdm(total=args.num_episodes, desc="accepted episodes") as progress:
            for _ in range(max_attempts):
                if saved >= args.num_episodes:
                    break
                trajectory, accepted, quality = collect_attempt(
                    env, seed=seed, max_steps=args.max_steps
                )
                attempt_seed = seed
                seed += 1
                if not accepted:
                    rejected += 1
                    continue

                if args.compact_static_frames:
                    compact, removed = remove_redundant_static_frames(trajectory)
                    # The compacted MP4 is written at a constant FPS, so its
                    # parquet timestamps must describe that compacted timeline.
                    compact["timestamps"] = (
                        np.arange(len(compact["states"]), dtype=np.float64)
                        * action_period
                    ).tolist()
                    quality["timestamps_preserve_control_time"] = False
                else:
                    compact = trajectory
                    removed = 0
                    quality["timestamps_preserve_control_time"] = True
                quality["removed_static_frames"] = removed
                row = writer.write_episode(
                    states=compact["robot_states"],
                    actions=compact["actions"],
                    images=compact["images"],
                    phases=compact["phases"],
                    dones=compact["dones"],
                    seed=attempt_seed,
                    quality=quality,
                    success=True,
                    timestamps=compact["timestamps"],
                )
                writer.validate_episode(row["episode_index"])
                saved += 1
                progress.update(1)
    finally:
        env.renderer.close()

    print(
        f"Collected {saved} episodes; dataset={args.dataset_dir.resolve()}",
        flush=True,
    )
    if saved < args.num_episodes:
        raise RuntimeError(
            f"Only {saved}/{args.num_episodes} episodes passed "
            f"within {max_attempts} attempts"
        )


if __name__ == "__main__":
    main()
