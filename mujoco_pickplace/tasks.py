"""Scene, instruction and action contract registry; one model serves all tasks."""

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import base64
import io
import json
import secrets
import sys
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from .episode_dataset import EpisodeDatasetWriter
from .task_env import trajectory_quality

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    number: int
    key: str
    instruction: str
    action_mask: tuple
    dataset: Path
    scene: Path
    max_steps: int
    gripper_mode: str = "binary"
    execution_horizon: int | None = None

    def prepare_action(self, action, gripper_filter):
        """Apply this task's public action contract, without privileged state."""
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite policy actions")
        action *= np.asarray(self.action_mask, dtype=np.float32)
        if self.gripper_mode == "continuous":
            action[6] = np.clip(action[6], 0.0, 1.0)
        else:
            action[6] = gripper_filter.update(action[6])
        return action

    def make_env(self, seed=None, **kwargs):
        if self.number == 1:
            from .pick_place_env import PickPlaceEnv

            env = PickPlaceEnv(scene_path=self.scene, **kwargs)
            env.reset(seed=seed)
            return env
        from .task_env import PandaTaskEnv

        return PandaTaskEnv(self.number, seed=seed, scene_path=self.scene, **kwargs)


TASKS = {
    1: Task(
        1,
        "mujoco_pickplace",
        "pick up the blue cube and place it on the green target",
        (1, 1, 1, 0, 0, 0, 1),
        ROOT / "Mujoco_training_dataset/cache/mujoco_pickplace",
        ROOT / "mujoco_pickplace/assets/pick_place_scene.xml",
        max_steps=250,
    ),
    3: Task(
        3,
        "mujoco_chopstick",
        "pick up the chopstick and place it inside the storage box",
        (1, 1, 1, 0, 0, 0, 1),
        ROOT / "Mujoco_training_dataset/cache/mujoco_chopstick",
        ROOT / "mujoco_pickplace/assets/task3_scene.xml",
        max_steps=300,
        execution_horizon=1,
    ),
    4: Task(
        4,
        "mujoco_drawer_memory",
        "take the object you just saw out of its drawer and place it on the green receiving area",
        (1, 1, 1, 1, 1, 1, 1),
        ROOT / "Mujoco_training_dataset/cache/mujoco_drawer_memory",
        ROOT / "mujoco_pickplace/assets/task4_scene.xml",
        max_steps=600,
        gripper_mode="continuous",
        execution_horizon=1,
    ),
}


def get_task(number):
    try:
        return TASKS[int(number)]
    except KeyError as error:
        raise ValueError("Supported tasks are 1, 3 and 4") from error


class ObservationHistory:
    def __init__(self, memory_frames=6, stride_seconds=1.0):
        if int(memory_frames) != memory_frames or memory_frames < 1:
            raise ValueError("memory_frames must be a positive integer")
        if not np.isfinite(stride_seconds) or stride_seconds <= 0:
            raise ValueError("stride_seconds must be finite and positive")
        self.memory_frames = int(memory_frames)
        self.stride_seconds = float(stride_seconds)
        self.clear()

    def clear(self):
        self.rows = deque()

    def append(self, observation, simulation_time):
        self.rows.append(
            (
                float(simulation_time),
                observation["robot_state"].copy(),
                observation["image_front"].copy(),
            )
        )
        window = (self.memory_frames - 1) * self.stride_seconds
        while len(self.rows) > 1 and self.rows[1][0] <= simulation_time - window - 1e-6:
            self.rows.popleft()

    def selected(self):
        if not self.rows:
            raise ValueError("History is empty")
        rows = list(self.rows)
        times = np.array([row[0] for row in rows])
        deadlines = (
            times[-1] - np.arange(self.memory_frames - 1, -1, -1) * self.stride_seconds
        )
        tolerance = max(1e-8, 2 * np.finfo(np.float32).eps * max(1.0, abs(times[-1])))
        indices = np.searchsorted(times, deadlines + tolerance, side="right") - 1
        valid = indices >= 0
        indices = np.maximum(indices, 0)
        return [rows[i] for i in indices], valid.tolist()

    def payload(self, task):
        rows, valid = self.selected()
        encoded = []
        for _, _, pixels in rows:
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            encoded.append([base64.b64encode(stream.getvalue()).decode("ascii")])
        return {
            "memory_images": encoded,
            "image_encoding": "png_base64",
            "state": [row[1].tolist() for row in rows],
            "history_mask": valid,
            "image_mask": [1],
            "action_mask": list(task.action_mask) + [0] * 17,
            "prompt": task.instruction,
            "task_key": task.key,
        }


def collection_config(task):
    result = {
        "task_id": task.number,
        "action_convention": "world_delta_pose_m_rad_gripper_0_close_1_open",
        "compact_static_frames": False,
        "control_period_seconds": 0.2,
        "robot_state_convention": "mujoco_hand_quaternion_axis_angle",
        "memory_frames": 6,
        "memory_stride_seconds": 1.0,
    }
    if task.number == 4:
        result.update(
            drawer_pitch_m=0.12,
            drawer_insert_height_m=0.03,
            handle_axis="horizontal",
            presentation="simulated human placement and closure; force disabled during execution",
        )
    return result


def collect_episode(task, env, seed, max_steps, compact_static_frames=False):
    """Run a task's expert and return the common dataset writer fields."""
    if task.number == 1:
        from .collect_data import collect_attempt, remove_redundant_static_frames

        trajectory, accepted, quality = collect_attempt(env, seed, max_steps)
        if compact_static_frames:
            trajectory, removed = remove_redundant_static_frames(trajectory)
            trajectory["timestamps"] = (
                np.arange(len(trajectory["robot_states"]))
                * env.model.opt.timestep
                * env.CONTROL_NSTEP
            ).tolist()
        else:
            removed = 0
        quality.update(
            removed_static_frames=removed,
            timestamps_preserve_control_time=not compact_static_frames,
        )
        episode = dict(
            states=trajectory["robot_states"],
            actions=trajectory["actions"],
            images=trajectory["images"],
            phases=trajectory["phases"],
            timestamps=trajectory["timestamps"],
            dones=trajectory["dones"],
        )
    else:
        if task.number == 4:
            list(env.presentation())
            from .task_env import run_task4_expert as run_expert
        else:
            from .task_env import run_task3_expert as run_expert
        succeeded = run_expert(env)
        quality = trajectory_quality(env, succeeded)
        within_budget = len(env.samples) <= max_steps
        quality["checks"]["within_collection_step_budget"] = within_budget
        if not within_budget:
            quality["failed_checks"].append("within_collection_step_budget")
        accepted = succeeded and not quality["failed_checks"]
        samples = env.samples
        episode = dict(
            states=[r["state"] for r in samples],
            actions=[r["action"] for r in samples],
            images={"front": [r["image"] for r in samples]},
            phases=[r["phase"] for r in samples],
            timestamps=[r["timestamp"] for r in samples],
            dones=[False] * (len(samples) - 1) + [True],
        )
    return episode, bool(accepted), quality


def collection_main(argv=None, args=None):
    """Collect every selected task in one process using one writer contract."""
    from .collect_data import parse_args, build_collection_config

    args = args if args is not None else parse_args(argv)
    seed = (
        args.start_seed
        if args.start_seed is not None
        else secrets.randbelow(2**31 - 100000)
    )
    reports = []
    if (
        len(args.tasks) > 1
        and args.dataset_dir is not None
        and args.dataset_dir.resolve() in {t.dataset.resolve() for t in TASKS.values()}
    ):
        raise ValueError(
            "For multiple tasks, --dataset-dir is the suite root, not one task dataset"
        )
    destinations = {}
    for number in args.tasks:
        task = get_task(number)
        dataset = (
            task.dataset
            if args.dataset_dir is None
            else (
                args.dataset_dir
                if len(args.tasks) == 1
                else args.dataset_dir / task.dataset.name
            )
        ).resolve()
        if dataset in {
            Path(dataset.anchor),
            ROOT,
            ROOT / "Mujoco_training_dataset",
            ROOT / "Mujoco_training_dataset/cache",
        }:
            raise ValueError(
                "Collection must target an individual dataset under the data root"
            )
        for other in TASKS.values():
            if other.number != number and (
                dataset == other.dataset.resolve()
                or other.dataset.resolve() in dataset.parents
            ):
                raise ValueError(
                    f"Task{number} cannot write inside Task{other.number} data"
                )
        if "mujoco_fridge" in str(dataset):
            raise ValueError("Fridge data cannot be used by this task suite")
        destinations[number] = dataset
    for number in args.tasks:
        task = get_task(number)
        dataset = destinations[number]
        env = task.make_env(
            seed=seed,
            **(
                dict(
                    image_size=args.image_size,
                    randomize_task=args.randomize_task,
                    randomization_scale=args.randomization_scale,
                )
                if number == 1
                else dict(capture_frames=False, image_size=args.image_size)
            ),
        )
        try:
            configuration = (
                build_collection_config(
                    env,
                    randomize_task=args.randomize_task,
                    randomization_scale=args.randomization_scale,
                    compact_static_frames=args.compact_static_frames,
                )
                if number == 1
                else collection_config(task)
            )
            if args.overwrite:
                # Clear only this selected dataset's generated contents.
                removed = []
                for name in ("data", "videos", "meta"):
                    target = dataset / name
                    if target.exists():
                        shutil.rmtree(target)
                        removed.append(name)
                print(
                    f"Overwriting dataset: cleared {removed or 'nothing'} under {dataset}",
                    flush=True,
                )
            writer_options = (
                {}
                if number == 1
                else dict(
                    source_policy="native-contact pose expert",
                    source_policy_version=f"task{number}-contact-v1",
                )
            )
            writer = EpisodeDatasetWriter(
                dataset,
                fps=1.0 / (env.model.opt.timestep * env.CONTROL_NSTEP),
                image_size=args.image_size,
                collection_config=configuration,
                task_description=task.instruction,
                **writer_options,
            )
            saved = rejected = 0
            max_attempts = args.max_attempts or args.num_episodes * 20
            with tqdm(total=args.num_episodes, desc="accepted episodes") as progress:
                for attempt in range(max_attempts):
                    if saved >= args.num_episodes:
                        break
                    attempt_seed = seed + attempt
                    if attempt:
                        env.renderer.close()
                        env = task.make_env(
                            seed=attempt_seed,
                            **(
                                dict(
                                    image_size=args.image_size,
                                    randomize_task=args.randomize_task,
                                    randomization_scale=args.randomization_scale,
                                )
                                if number == 1
                                else dict(
                                    capture_frames=False, image_size=args.image_size
                                )
                            ),
                        )
                    episode, accepted, quality = collect_episode(
                        task,
                        env,
                        attempt_seed,
                        args.max_steps or task.max_steps,
                        args.compact_static_frames,
                    )
                    if not accepted:
                        rejected += 1
                        continue
                    row = writer.write_episode(
                        **episode, seed=attempt_seed, quality=quality, success=True
                    )
                    writer.validate_episode(row["episode_index"])
                    saved += 1
                    progress.update(1)
            report = dict(
                task=number, saved=saved, rejected=rejected, dataset=str(dataset)
            )
            reports.append(report)
            print(f"Collected {saved} episodes; dataset={dataset}", flush=True)
            if saved < args.num_episodes:
                raise RuntimeError(
                    f"Only {saved}/{args.num_episodes} episodes passed "
                    f"within {max_attempts} attempts"
                )
        finally:
            env.renderer.close()
    return reports


async def evaluate_tasks(numbers=None, argv=None, args=None):
    """The full task suite, including Task1, shares one model connection."""
    import imageio.v2 as imageio
    import websockets
    from . import eval_policy_client as client

    args = args if args is not None else client.parse_args(argv)
    numbers = list(args.tasks if numbers is None else numbers)
    seed = (
        args.start_seed
        if args.start_seed is not None
        else secrets.randbelow(2**31 - 100000)
    )
    output = Path(args.video_dir).resolve()
    reports = []
    total_steps = 0
    if args.start_seed is None:
        client.log.info("Random evaluation start seed: %s", seed)
    diagnostics = Path(args.diagnostics_jsonl) if args.diagnostics_jsonl else None
    if diagnostics is not None:
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text("")
    async with websockets.connect(
        args.server_url, max_size=100_000_000, ping_timeout=None
    ) as socket:
        metadata = await client.request_model_metadata(socket)
        memory_frames = int(metadata.get("memory_frames", 1))
        stride_seconds = metadata.get("memory_stride_seconds")
        if stride_seconds is None or stride_seconds <= 0:
            stride_seconds = float(metadata.get("memory_stride_steps", 5)) * 0.2
        if args.explicit_memory_frames and args.memory_frames != memory_frames:
            raise ValueError(
                f"Client K={args.memory_frames} differs from checkpoint K={memory_frames}"
            )
        if (
            args.explicit_memory_stride
            and abs(args.memory_stride_steps * 0.2 - stride_seconds) > 1e-6
        ):
            raise ValueError(
                "Explicit client history stride differs from checkpoint settings"
            )
        required = {get_task(number).key for number in numbers}
        if not required.issubset(set(metadata.get("normalization_keys", []))):
            raise ValueError(
                f'Checkpoint lacks task statistics: {required}; available: {metadata.get("normalization_keys")}'
            )
        execution_horizon, precision_replan = client.resolve_execution_policy(
            args, metadata
        )
        for number in numbers:
            task = get_task(number)
            client.log.info(
                f"\n========= Start task{number}: {task.instruction} ========="
            )
            directory = output / f"task{number}"
            directory.mkdir(parents=True, exist_ok=True)
            removed_videos = client.clear_episode_videos(output, f"task{number}")
            if removed_videos:
                client.log.info(
                    "Removed %s stale episode videos from %s/%s",
                    removed_videos,
                    output,
                    f"task{number}",
                )
            # One report per selected task is regenerated, leaving all other tasks intact.
            (directory / "summary.json").write_text("[]\n")
            for episode in range(args.num_episodes):
                episode_seed = seed + episode
                env = task.make_env(
                    seed=episode_seed,
                    **(
                        dict(
                            image_size=448,
                            randomize_task=args.randomize_task,
                            randomization_scale=args.randomization_scale,
                        )
                        if number == 1
                        else dict(capture_frames=False, image_size=448)
                    ),
                )
                history = ObservationHistory(memory_frames, stride_seconds)
                frames = []
                grip = client.GripperCommandFilter(
                    required_steps=args.gripper_debounce_steps
                )
                precision = client.PrecisionExecutionController()
                done = False
                step = 0
                video_frames_per_step = 1
                try:
                    if hasattr(env, "GRASP_X_BIAS"):
                        env.GRASP_X_BIAS = float(
                            metadata.get("recommended_grasp_x_bias", env.GRASP_X_BIAS)
                        )
                    if number == 1 and episode == 0:
                        client.log.info(
                            "Evaluation execution policy: horizon=%s, precision_replan=%s, grasp_x_bias=%.3f",
                            execution_horizon,
                            precision_replan,
                            env.GRASP_X_BIAS,
                        )
                    elif number != 1 and episode == 0:
                        client.log.info(
                            "Evaluation execution policy: horizon=%s, precision_replan=%s",
                            task.execution_horizon,
                            False,
                        )
                    print(
                        f"\n===== Task {number - 1} | Episode {episode + 1} =====",
                        flush=True,
                    )
                    print(task.instruction, flush=True)
                    if number == 1:
                        print(
                            f"cube_xy={env.initial_cube_xy.round(5).tolist()}, "
                            f"goal_xy={env.initial_goal_xy.round(5).tolist()}",
                            flush=True,
                        )
                    observation = env.obs()
                    history.append(observation, env.data.time)
                    frames.append(observation["image_front"])
                    if hasattr(env, "presentation"):
                        for observation in env.presentation():
                            history.append(observation, env.data.time)
                            frames.append(observation["image_front"])
                    max_steps = args.max_steps or task.max_steps
                    while step < max_steps and not done:
                        payload = history.payload(task)
                        print(f"[Step {step}] Send observation", flush=True)
                        await socket.send(json.dumps(payload))
                        chunk = np.asarray(
                            json.loads(await socket.recv()), dtype=np.float32
                        )
                        if (
                            chunk.ndim != 2
                            or not len(chunk)
                            or chunk.shape[1] != 24
                            or not np.isfinite(chunk).all()
                        ):
                            raise ValueError(
                                "Server returned invalid actions; expected [horizon,24]"
                            )
                        print(
                            f"[Step {step}] recivied actions (shape={chunk.shape})",
                            flush=True,
                        )
                        horizon = min(execution_horizon, len(chunk), max_steps - step)
                        if number == 1:
                            horizon = precision.execution_horizon(
                                chunk,
                                horizon,
                                observation["robot_state"],
                                grip,
                                enabled=precision_replan,
                            )
                        else:
                            # Explicit task contract; conflicting CLI overrides are rejected.
                            horizon = min(horizon, task.execution_horizon)
                        print(f"[Step {step}] execute horizon={horizon}", flush=True)
                        client.append_diagnostic(
                            diagnostics,
                            dict(
                                kind="decision",
                                task=number,
                                episode=episode,
                                simulation_time=float(env.data.time),
                                history_times=[r[0] for r in history.selected()[0]],
                                execution_horizon=horizon,
                            ),
                        )
                        for action in chunk[:horizon]:
                            before_state = observation["robot_state"].copy()
                            raw_action = action[:7].copy()
                            print(action[:7])
                            action = task.prepare_action(action[:7], grip)
                            print("gripper action", action[6])
                            if hasattr(env, "step_video"):
                                captured, observation, done = env.step_video(
                                    action, frames_per_step=client.FRAMES_PER_STEP
                                )
                                video_frames_per_step = len(captured["front"])
                                frames.extend(captured["front"])
                            else:
                                env.phase = "policy_execution"
                                observation, done = env.step(action)
                                frames.append(observation["image_front"])
                            step += 1
                            if diagnostics and number != 1:
                                from .task_env import rotation_matrix, rotation_vector

                                after_state = observation["robot_state"]
                                executed = env.samples[-1]["action"]
                                actual_rotation = (
                                    rotation_matrix(after_state[3:6])
                                    @ rotation_matrix(before_state[3:6]).T
                                )
                                row = env.rows[-1]
                                client.append_diagnostic(
                                    diagnostics,
                                    dict(
                                        kind="action",
                                        task=number,
                                        episode=episode,
                                        step=step,
                                        simulation_time=float(env.data.time),
                                        raw_action=raw_action.tolist(),
                                        executed_action=executed.tolist(),
                                        state_before=before_state.tolist(),
                                        state_after=after_state.tolist(),
                                        translation_tracking_error_mm=float(
                                            np.linalg.norm(
                                                after_state[:3]
                                                - before_state[:3]
                                                - executed[:3]
                                            )
                                            * 1000
                                        ),
                                        rotation_tracking_error_degrees=float(
                                            np.rad2deg(
                                                np.linalg.norm(
                                                    rotation_vector(
                                                        rotation_matrix(executed[3:6])
                                                        @ actual_rotation.T
                                                    )
                                                )
                                            )
                                        ),
                                        physics_contacts={
                                            key: row.get(key)
                                            for key in (
                                                "two_finger_contact",
                                                "two_finger_handle_contact",
                                                "handle_two_contact_physics_steps",
                                                "longest_handle_contact_gap_seconds",
                                                "finger_table_contacts",
                                                "palm_table_contacts",
                                                "table_penetration_m",
                                                "unexpected_robot_contacts_peak",
                                            )
                                        },
                                    ),
                                )
                            history.append(observation, env.data.time)
                            reward = 1.0 if done else 0.0
                            print(
                                f"[Step {step}] reward={reward:.2f}, done={done}",
                                flush=True,
                            )
                            if done:
                                print("Task completed", flush=True)
                                break
                    report = dict(
                        task=number,
                        episode=episode,
                        seed=episode_seed,
                        success=bool(done),
                        steps=step,
                        checkpoint_metadata=metadata,
                    )
                    if number != 1:
                        report.update(
                            first_grasp_success=(
                                bool(env.first_grasp_lifted)
                                if number == 3 and env.grasp_attempts
                                else None
                            ),
                            grasp_attempts=env.grasp_attempts if number == 3 else None,
                            object_grasp_confirmed=bool(env.had_object_contact),
                            object_lifted=bool(env.was_lifted),
                            finger_table_contact=bool(env.finger_table_contact_seen),
                            table_penetration_m=min(
                                [0.0] + [row["table_penetration_m"] for row in env.rows]
                            ),
                            unsafe_execution_contact=bool(env.unsafe_execution_contact),
                            transport_drop=bool(env.transport_dropped),
                            correct_first_drawer=(
                                (env.first_selected_drawer == env.target)
                                if number == 4
                                else None
                            ),
                            natural_corrections=None,
                        )
                    fps = video_frames_per_step / (
                        env.model.opt.timestep * env.CONTROL_NSTEP
                    )
                    imageio.mimwrite(
                        directory / f"episode_{episode:04d}.mp4",
                        frames,
                        fps=fps,
                        macro_block_size=1,
                    )
                    reports.append(report)
                    (directory / "summary.json").write_text(
                        json.dumps(
                            [r for r in reports if r["task"] == number], indent=2
                        )
                        + "\n"
                    )
                    if args.render:
                        client.maybe_show(frames[-1], True)
                    client.append_diagnostic(
                        diagnostics, dict(kind="episode_summary", **report)
                    )
                    total_steps += step
                    result_text = "✅ Success" if done else "❌ Fail"
                    client.log.info(
                        f"Task {number - 1} | Episode {episode + 1}: {result_text}"
                    )
                finally:
                    history.clear()
                    env.renderer.close()
            task_success = sum(r["success"] for r in reports if r["task"] == number)
            client.log.info(
                f"========= Task {number} Summary: {task_success}/{args.num_episodes} Successful ========="
            )
    summary = dict(
        tasks=numbers,
        successful_episodes=sum(r["success"] for r in reports),
        total_episodes=len(reports),
        episodes=reports,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    client.log.info("\n========= Overall Task Summary =========")
    client.log.info(
        f"✅ Total Successful Episodes: {summary['successful_episodes']}/{summary['total_episodes']}"
    )
    client.log.info(
        f"📊 Average Steps: {total_steps / max(summary['total_episodes'], 1):.2f}"
    )
    client.log.info(
        f"success_rate={summary['successful_episodes'] / max(summary['total_episodes'], 1):.3f}"
    )
    return reports
