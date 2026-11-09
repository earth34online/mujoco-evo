from dataclasses import dataclass
from pathlib import Path
from collections import deque
import base64
import io
import json
import secrets
import sys
import shutil
import hashlib
import inspect
import os
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from tqdm import tqdm

from .episode_dataset import (
    EpisodeDatasetWriter, contact_source_policy_version, compare_environment_contract,
    CONTACT_QUALITY_SCHEMA, CONTACT_QUALITY_VERSION, SOURCE_POLICY_VERSION, _atomic_json,
)
from .task_env import (
    trajectory_quality, DRAWER_START_JITTER, DRAWER_APPROACH_JITTER,
    DRAWER_OBSERVATION_HAND_POSITION, DRAWER_EXECUTION_HAND_POSITION,
    DRAWER_CONTROL_PERIOD_SECONDS,
    OBJECT_GRASP_HAND_Z_OFFSETS_M,
    CABINET_POSITION, CABINET_YAW,
)

ROOT = Path(__file__).resolve().parents[1]


def evaluation_failure_stop(env, elapsed_steps, recovery_steps):
    """Audit failures don't terminate an otherwise executable policy episode."""
    reasons = {event["reason"] for event in env.failure_events}
    if (not np.isfinite(getattr(env.data, "qpos", [])).all()
            or not np.isfinite(getattr(env.data, "qvel", [])).all()):
        return "invalid_physics_state"
    # A soft contact event is not an emergency. Stop on actual deep scene
    # penetration / excessive table force, rather than a latched audit flag.
    row = env.rows[-1] if env.rows else {}
    if (row.get("table_penetration_m", 0.0) < -0.003
            or row.get("peak_table_contact_force_n", 0.0) > 30.0
            or any(pair[2] < -0.003 for pair in row.get("unexpected_robot_contacts_peak", []))):
        return "unsafe_physics_stop"
    outcome = env.outcome()
    if outcome["physical_complete"] and not outcome["strict_quality_passed"]:
        return "physical_complete_quality_failed"
    if not reasons:
        return None
    if recovery_steps and elapsed_steps >= recovery_steps:
        return "recovery_step_limit"
    return None


def task1_action_evidence(env, raw_action, prepared_action, before_state, after_state):
    """Read-only execution evidence; never advance grasp or success state."""
    cube = env.data.body("cube").xpos.copy()
    goal = env.model.body("goal").pos.copy()
    cube_velocity = env.data.qvel[env.cube_dof_id:env.cube_dof_id + 3].copy()
    executed = np.asarray(prepared_action, dtype=np.float64).copy()
    # target_hand is Task1's workspace-clipped target submitted to IK.
    executed[:3] = np.asarray(env.target_hand) - np.asarray(before_state[:3])
    executed[3:6] = 0.0
    executed[6] = env.gripper
    in_goal = bool(np.linalg.norm(cube[:2] - goal[:2]) <= env.GOAL_RADIUS)
    at_support = bool(abs(cube[2] - env.CUBE_SUPPORT_Z) < 0.012)
    released = bool(env.gripper > 0.5 and not env.attached)
    cube_speed = float(np.linalg.norm(cube_velocity))
    return dict(
        raw_action=np.asarray(raw_action).tolist(),
        prepared_action=np.asarray(prepared_action).tolist(),
        executed_action=executed.tolist(),
        state_before=np.asarray(before_state).tolist(),
        state_after=np.asarray(after_state).tolist(),
        translation_tracking_error_mm=float(np.linalg.norm(
            np.asarray(after_state[:3]) - np.asarray(before_state[:3]) - executed[:3]) * 1000),
        object_position=cube.tolist(), goal_position=goal.tolist(),
        object_linear_velocity=cube_velocity.tolist(),
        grasp=dict(attached=bool(env.attached), gripper=float(env.gripper),
                   two_sided_contact=bool(env._has_two_sided_grasp_contact()),
                   release_counter=int(env.release_counter)),
        outcome=dict(
            object_in_goal_xy=in_goal, object_at_support_height=at_support,
            object_speed=cube_speed, released=released,
            physical_complete=bool(in_goal and at_support and released
                                   and cube_speed < env.SUCCESS_MAX_CUBE_SPEED),
            unsafe_robot_table_contact=bool(env.unsafe_robot_table_contact),
            robot_table_contact_now=bool(env.has_robot_table_contact()),
            success_candidate=bool(env._success_candidate()),
            success_counter=int(env.success_counter),
            success_dwell_steps=int(env.SUCCESS_DWELL_STEPS),
            success=bool(env.success()),
        ),
    )


def target_history_evidence(env, history):
    """Simulator-only audit annotation; never added to policy observations."""
    rows, valid = history.selected()
    visible_times = getattr(env, "_target_visible_times", [])
    slots = [slot for slot, (row, ok) in enumerate(zip(rows, valid))
             if ok and any(abs(row[0] - t) < 1e-6 for t in visible_times)]
    return dict(history_times=[row[0] for row in rows], target_visible_slots=slots,
                target_available=bool(slots))


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
    gripper_open_required_steps: int | None = None
    evaluation_max_steps: int | None = None

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
        evaluation_max_steps=200,
    ),
    3: Task(
        3,
        "mujoco_chopstick",
        "pick up the chopstick and place it inside the storage box",
        (1, 1, 1, 0, 0, 0, 1),
        ROOT / "Mujoco_training_dataset/cache/mujoco_chopstick",
        ROOT / "mujoco_pickplace/assets/task3_scene.xml",
        max_steps=300,
        evaluation_max_steps=250,
        execution_horizon=1,
        gripper_open_required_steps=1,
    ),
    4: Task(
        4,
        "mujoco_drawer_memory",
        "take the object you just saw out of its drawer and place it on the green receiving area",
        (1, 1, 1, 1, 1, 1, 1),
        ROOT / "Mujoco_training_dataset/cache/mujoco_drawer_memory",
        ROOT / "mujoco_pickplace/assets/task4_scene.xml",
        max_steps=600,
        evaluation_max_steps=600,
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
        "expert_close_rule": "continuous_two_finger_contact_40ms",
    }
    if task.number in OBJECT_GRASP_HAND_Z_OFFSETS_M:
        result["expert_object_grasp_hand_z_offset_m"] = OBJECT_GRASP_HAND_Z_OFFSETS_M[task.number]
    if task.gripper_open_required_steps is not None:
        result["gripper_open_required_steps"] = task.gripper_open_required_steps
    if task.number == 4:
        result.update(
            control_period_seconds=DRAWER_CONTROL_PERIOD_SECONDS,
            drawer_pitch_m=0.12,
            drawer_insert_height_m=0.03,
            drawer_inner_width_m=0.248,
            drawer_object_xy_jitter_m=[0.006, 0.010],
            scene_light_mode="fixed",
            handle_axis="horizontal",
            cabinet_position_m=CABINET_POSITION.tolist(),
            cabinet_yaw_radians=float(CABINET_YAW),
            expert_transfer_path="local_turn_symmetric_grasp_clear_workspace_v2",
            expert_handle_approach_offset_frame="cabinet",
            common_observation_hand_position_m=DRAWER_OBSERVATION_HAND_POSITION.tolist(),
            observation_hand_xyz_jitter_m=DRAWER_START_JITTER.tolist(),
            common_execution_hand_position_m=DRAWER_EXECUTION_HAND_POSITION.tolist(),
            execution_hand_xyz_jitter_m=DRAWER_START_JITTER.tolist(),
            expert_handle_approach_xyz_jitter_m=DRAWER_APPROACH_JITTER.tolist(),
            presentation_close_duration_seconds=1.2,
            presentation_closed_at_joint_stop=True,
            presentation="simulated human placement and closure; force disabled during execution",
        )
    elif task.number == 3:
        result.update(chopstick_length_m=0.24, chopstick_diameter_min_m=0.008,
                      chopstick_diameter_max_m=0.010, chopstick_rest_height_m=0.012)
    return result


def task_environment_contract(task, configuration=None, *, randomize_task=True, randomization_scale=1.0):
    """Capture actual assets/control code at collection or before evaluation.

    Training only copies the recorded dataset contract; it must not stamp the
    current workstation's assets onto demonstrations collected in the past.
    """
    import mujoco

    engine = dict(name="mujoco", python_package_version=mujoco.__version__,
                  runtime_version=mujoco.mj_versionString())
    if engine["python_package_version"] != engine["runtime_version"]:
        raise RuntimeError(f"MuJoCo Python package/loaded engine mismatch: {engine}")
    if configuration is None:
        if task.number == 1:
            from .collect_data import build_collection_config
            from .pick_place_env import PickPlaceEnv
            configuration = build_collection_config(
                PickPlaceEnv, randomize_task=randomize_task,
                randomization_scale=randomization_scale, compact_static_frames=False)
        else:
            configuration = collection_config(task)
    panda = ROOT / "mujoco_menagerie/franka_emika_panda"
    paths = [task.scene, panda / "panda.xml"]
    # Include visual/collision assets as well as the two XML files. Paths in
    # the fingerprint are repository-relative, so cloud/local roots can differ.
    for xml_path in list(paths):
        for element in ET.parse(xml_path).getroot().iter():
            if element.tag in {"mesh", "texture", "hfield", "skin"} and element.get("file"):
                paths.append((panda / "assets" / element.get("file")).resolve())
    assets = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(set(paths))}
    controller = ROOT / "mujoco_pickplace" / ("pick_place_env.py" if task.number == 1 else "task_env.py")
    control_sources = {controller.name: hashlib.sha256(controller.read_bytes()).hexdigest(),
                       "Task.prepare_action": hashlib.sha256(inspect.getsource(Task.prepare_action).encode()).hexdigest()}
    if task.gripper_mode == "binary":
        from .eval_policy_client import GripperCommandFilter, PrecisionExecutionController
        control_sources["GripperCommandFilter"] = hashlib.sha256(inspect.getsource(GripperCommandFilter).encode()).hexdigest()
        if task.number == 1:
            control_sources["PrecisionExecutionController"] = hashlib.sha256(inspect.getsource(PrecisionExecutionController).encode()).hexdigest()
    option = ET.parse(task.scene).getroot().find("option")
    timestep = float(option.get("timestep", "0.002")) if option is not None else 0.002
    if task.number == 1:
        from .pick_place_env import PickPlaceEnv
        control_steps = PickPlaceEnv.CONTROL_NSTEP
    else:
        from .task_env import PandaTaskEnv
        control_steps = 50 if task.number == 4 else PandaTaskEnv.CONTROL_NSTEP
    return dict(schema="mujoco-task-environment", version=2, task_id=task.number,
                physics_engine=engine,
                control_period_seconds=timestep * control_steps,
                action_convention="world_delta_pose_m_rad_gripper_0_close_1_open",
                state_convention="mujoco_hand_quaternion_axis_angle",
                action_mask=list(task.action_mask), gripper_mode=task.gripper_mode,
                instruction=task.instruction, collection_config=dict(configuration),
                assets_sha256=assets, controller_sources_sha256=control_sources,
                controller_sha256=hashlib.sha256(json.dumps(control_sources, sort_keys=True).encode()).hexdigest(),
                quality_schema=(dict(schema=SOURCE_POLICY_VERSION, version=3) if task.number == 1
                                else dict(schema=CONTACT_QUALITY_SCHEMA, version=CONTACT_QUALITY_VERSION)))


def _persist_evaluation_summary(state, status, error=None):
    reports = state["episodes"]
    summary = dict(run_id=state["run_id"], status=status,
                   started_utc=state["started_utc"], updated_utc=datetime.now(timezone.utc).isoformat(),
                   tasks=state["tasks"], requested_episodes_per_task=state["requested_episodes_per_task"],
                   successful_episodes=sum(r["success"] for r in reports),
                   total_episodes=len(reports), episodes=reports,
                   active_episode=state.get("active_episode"),
                   environment_compatibility=state.get("environment_compatibility"),
                   environment_compatibility_all_verified=bool(state.get("environment_compatibility")) and all(
                       item["status"] == "verified" for item in state["environment_compatibility"].values()))
    if error is not None:
        summary["error"] = dict(type=type(error).__name__, message=str(error))
    _atomic_json(state["output"] / "summary.json", summary)
    return summary


class RecoveryContextRejected(ValueError):
    """A valid recorded episode cannot supply a safe correction prefix here."""


def _recorded_empty_grasp_sources(path):
    """Select real empty-closure episodes, without synthesizing a failure quota."""
    return _recorded_recovery_sources(path, 3)


def _recorded_recovery_sources(path, task_number, source_kind=None):
    """Select certified real prefixes; expert diagnostics never enter policy input."""
    if task_number not in (3, 4):
        raise ValueError("Recorded correction sources require Task3 or Task4")
    path = Path(path).resolve()
    summary = json.loads((path.parent / "summary.json").read_text())
    by_episode = {row["episode"]: row for row in summary if row["task"] == task_number}
    groups = {}
    action_bytes = path.read_bytes()
    for line in action_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "action" and row.get("task") == task_number:
            groups.setdefault(row["episode"], []).append(row)
    result = []
    digest = hashlib.sha256(action_bytes).hexdigest()
    for episode, rows in sorted(groups.items()):
        record = by_episode.get(episode)
        if not record or any(r.get("run_id") != record.get("run_id") for r in rows):
            raise ValueError("Recovery actions and summary must belong to the same evaluation run")
        if [r["step"] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Recovery action steps must be complete and contiguous")
        if task_number == 4:
            if (source_kind == "handle_policy_state"
                    or source_kind is None and record.get("drawer_selection_status") == "wrong"):
                # This is a safe policy-state continuation, not a fabricated
                # failure. Intervene before a wrong-shelf approach reaches it.
                prefix = rows[:5]
                safe = (len(prefix) == 5 and not record["success"]
                        and record.get("drawer_selection_status") == "wrong"
                        and all(not row["outcome"].get("failure_events")
                            and row["outcome"].get("first_selected_drawer") is None
                            and not row["physics_contacts"]["two_finger_handle_contact"]
                            and max(row["physics_contacts"]["drawer_positions"]) <= .005
                            for row in prefix))
                if safe:
                    result.append(dict(seed=record["seed"], episode=episode, rows=rows,
                        path=str(path), sha256=digest, run_id=record["run_id"],
                        kind="handle_policy_state", intervention_step=len(prefix)))
                continue
            from .task_env import HANDLE_ROTATION, OBJECT_ROTATION, rotation_matrix, rotation_vector
            # A real empty closure after opening a correct drawer needs
            # object-stage labels, not another handle demonstration.
            object_failure = next((row for row in rows
                if not record["success"]
                and row["outcome"].get("drawer_selection_status") == "correct"
                and row["outcome"].get("opened_drawer") == row["outcome"].get("first_selected_drawer")
                and row["outcome"].get("opened_drawer") is not None
                and not row["outcome"].get("failure_events")
                and not row["physics_contacts"]["two_finger_contact"]
                and np.linalg.norm(rotation_vector(
                    OBJECT_ROTATION @ rotation_matrix(row["state_after"][3:6]).T)) < .12
                and row["executed_action"][6] <= .2
                and sum(row["state_after"][6:8]) < .006
                and max(row["physics_contacts"]["drawer_positions"]) >= .165
                and not row["outcome"].get("object_retention_state") in {"held", "dropped"}
                and row["step"] > 2), None)
            if object_failure is not None:
                result.append(dict(seed=record["seed"], episode=episode, rows=rows,
                                   path=str(path), sha256=digest, run_id=record["run_id"],
                                   kind="object_empty_grasp", intervention_step=object_failure["step"]))
                continue
            # Correct only a measured early reversal, while target history is
            # still available. Later stalled episodes cannot supply that cue.
            closest = np.full(3, np.inf)
            had_handle_contact = False
            observed_opening = 0.0
            # Four seconds at Task4's 10 Hz leaves only a short chance to
            # intervene. Actual replay and final selection must still pass
            # the existing five-second visible-history qualification.
            for row in rows[:40]:
                if row["outcome"].get("failure_events"):
                    break
                contacts = row["physics_contacts"]
                bilateral = bool(contacts.get("two_finger_handle_contact", False))
                had_handle_contact |= bilateral
                positions = contacts.get("drawer_positions", [])
                if len(positions) == 4:
                    observed_opening = max(observed_opening, max(abs(value) for value in positions))
                if row["outcome"].get("first_selected_drawer") is not None:
                    if row["outcome"].get("quality_checks", {}).get("correct_first_drawer") is not True:
                        break
                    if (had_handle_contact and not bilateral
                            and row["executed_action"][6] <= .1
                            and len(positions) == 4 and observed_opening <= .005):
                        result.append(dict(seed=record["seed"], episode=episode, rows=rows,
                                           path=str(path), sha256=digest, run_id=record["run_id"],
                                           kind="handle_contact_loss", intervention_step=row["step"]))
                        break
                    continue
                errors = np.asarray([
                    row["physics_contacts"][name] for name in (
                        "handle_axial_tracking_error_m", "handle_vertical_tracking_error_m",
                        "handle_lateral_tracking_error_m")
                ], dtype=float)
                closest = np.minimum(closest, errors)
                orientation_error = float(np.linalg.norm(rotation_vector(
                    HANDLE_ROTATION @ rotation_matrix(row["state_after"][3:6]).T)))
                if np.max(errors - closest) >= 0.005 or orientation_error >= np.deg2rad(2.0):
                    result.append(dict(seed=record["seed"], episode=episode, rows=rows,
                                       path=str(path), sha256=digest, run_id=record["run_id"],
                                       kind="handle_approach_deviation", intervention_step=row["step"]))
                    break
            continue
        previous = 1.0
        failed_close = False
        for row in rows:
            command = row["executed_action"][6]
            if (previous >= .5 and command < .5
                    and not row["physics_contacts"]["two_finger_contact"]
                    and sum(row["state_after"][6:8]) < .004
                    and not row["outcome"].get("quality_checks", {}).get("object_lifted", False)):
                failed_close = True
                break
            previous = command
        if failed_close:
            result.append(dict(seed=record["seed"], episode=episode, rows=rows,
                               path=str(path), sha256=digest, run_id=record["run_id"]))
    return result


def _replay_handle_deviation_context(task, env, source):
    """Expert intervention at an actual, safe, early off-path handle approach."""
    from .task_env import HANDLE_ROTATION, rotation_vector
    if source.get("kind") == "handle_policy_state":
        return _replay_handle_policy_state_context(task, env, source)
    tracking_errors = []
    max_state_error = 0.0
    contact_loss = source.get("kind") == "handle_contact_loss"
    if source.get("kind") not in {"handle_approach_deviation", "handle_contact_loss"}:
        raise RecoveryContextRejected("Unsupported handle intervention kind")
    handle = env.model.geom(f"handle{env.target}").id
    had_bilateral = False
    maximum_opening = 0.0
    for row in source["rows"][:source["intervention_step"]]:
        max_state_error = max(max_state_error, float(np.max(
            np.abs(env.robot_state() - np.asarray(row["state_before"])))))
        if max_state_error > 1e-4:
            raise RecoveryContextRejected("Recorded handle prefix does not match the current physical state")
        env.phase = "policy_context"
        env.step(task.prepare_action(row["raw_action"], None))
        max_state_error = max(max_state_error, float(np.max(
            np.abs(env.robot_state() - np.asarray(row["state_after"])))))
        if max_state_error > 1e-4:
            raise RecoveryContextRejected("Recorded handle prefix end state does not physically reproduce")
        had_bilateral |= bool(env.contacts(handle)[0])
        if contact_loss:
            maximum_opening = max(maximum_opening, max(
                abs(float(env.data.qpos[env.model.joint(f"drawer_slide{k}").qposadr[0]]))
                for k in range(4)))
        if (env.failure_events or (env.first_selected_drawer is not None
                and (not contact_loss or env.first_selected_drawer != env.target))):
            raise RecoveryContextRejected("Handle correction requires a safe prefix with no wrong selection")
        # Match the selector's actual finger/handle tracking diagnostics.
        # Total distance can improve while vertical or lateral alignment gets worse.
        tracking_errors.append(np.asarray([env.rows[-1][name] for name in (
            "handle_axial_tracking_error_m", "handle_vertical_tracking_error_m",
            "handle_lateral_tracking_error_m")], dtype=float))
    deviation = float(np.max(tracking_errors[-1] - np.min(tracking_errors, axis=0)))
    orientation_error = float(np.linalg.norm(rotation_vector(
        HANDLE_ROTATION @ env.data.xmat[env.hand].reshape(3, 3).T)))
    if contact_loss:
        if (not had_bilateral or env.first_selected_drawer != env.target
                or env.contacts(handle)[0] or env.gripper > .1 or maximum_opening > .005):
            raise RecoveryContextRejected("Correct-handle loss did not reproduce before opening the drawer")
        env._recorded_handle_loss_evidence = dict(
            recorded_first_selected_drawer=int(env.first_selected_drawer),
            recorded_bilateral_before_loss=True, recorded_bilateral_at_intervention=False,
            recorded_closed_at_intervention=True, recorded_max_drawer_open_m=maximum_opening)
    elif deviation < 0.005 and orientation_error < np.deg2rad(2.0):
        raise RecoveryContextRejected("Recorded handle deviation did not physically reproduce")
    env._recorded_handle_deviation_m = deviation
    env._recorded_handle_orientation_error_rad = orientation_error
    env._recorded_replay_state_error = max_state_error
    env._recorded_handle_deviation_time = float(env.data.time)
    if env.last_visible_time is None or env.data.time - env.last_visible_time > 5.0 + 1e-6:
        raise RecoveryContextRejected("Target history already expired before handle intervention")
    count = len(env.samples)
    # Withdraw before changing height: a policy may approach the wrong shelf.
    # A diagonal move from that near-cabinet pose can hit the closed front.
    if not contact_loss and not env.move(
        env.data.xpos[env.hand] - .04 * HANDLE_ROTATION[:, 2], .25, "recover"
    ):
        raise RecoveryContextRejected("Safe handle withdrawal failed")
    if not env.orient(HANDLE_ROTATION, 0.25, "recover"):
        raise RecoveryContextRejected("Safe handle orientation recovery failed")
    return count


def _replay_handle_policy_state_context(task, env, source):
    """Verify a safe policy-visited state; this source makes no failure claim."""
    from .task_env import HANDLE_ROTATION
    steps = source.get("intervention_step")
    if type(steps) is not int or steps < 2 or len(source["rows"]) < steps:
        raise RecoveryContextRejected("Invalid policy-state intervention prefix")
    handles = [env.model.geom(f"handle{k}").id for k in range(4)]
    slides = [env.model.joint(f"drawer_slide{k}").qposadr[0] for k in range(4)]
    maximum_opening = 0.0
    substeps = 0
    max_state_error = 0.0

    def check_prefix_state():
        nonlocal maximum_opening
        maximum_opening = max(maximum_opening, max(abs(float(env.data.qpos[j])) for j in slides))
        if (maximum_opening > .005 or any(env.contacts(h)[0] for h in handles)
                or env.first_selected_drawer is not None or env.failure_events
                or env.unsafe_execution_contact or env.transport_dropped or env.motion_failures):
            raise RecoveryContextRejected("Policy-state prefix must be safe, unopened and never selected or bilateral")

    previous_override = env.__dict__.get("_update_object_retention")
    had_override = "_update_object_retention" in env.__dict__
    original = env._update_object_retention

    def observe_substep():
        nonlocal substeps
        original()
        substeps += 1
        check_prefix_state()

    check_prefix_state()
    env._update_object_retention = observe_substep
    try:
        for row in source["rows"][:steps]:
            error = float(np.max(np.abs(env.robot_state() - np.asarray(row["state_before"]))))
            if not np.isfinite(error) or error > 1e-4:
                raise RecoveryContextRejected("Policy-state prefix start does not reproduce")
            max_state_error = max(max_state_error, error)
            env.phase = "policy_context"
            env.step(task.prepare_action(row["raw_action"], None))
            error = float(np.max(np.abs(env.robot_state() - np.asarray(row["state_after"]))))
            if not np.isfinite(error) or error > 1e-4:
                raise RecoveryContextRejected("Policy-state prefix end does not reproduce")
            max_state_error = max(max_state_error, error)
            check_prefix_state()
    finally:
        if had_override:
            env._update_object_retention = previous_override
        else:
            del env._update_object_retention
    if substeps != steps * env.CONTROL_NSTEP:
        raise RecoveryContextRejected("Policy-state prefix physics monitoring was incomplete")
    if env.last_visible_time is None or env.data.time - env.last_visible_time > 5.0 + 1e-6:
        raise RecoveryContextRejected("Target history already expired before policy-state intervention")
    env._recorded_policy_state_evidence = dict(
        policy_state_prefix_verified=True, confirmed_failure_claim=False,
        recorded_intervention_sim_time=float(env.data.time),
        recorded_replay_max_state_error=max_state_error,
        recorded_first_selected_drawer=None, recorded_bilateral_ever=False,
        recorded_max_drawer_open_m=maximum_opening,
        recorded_prefix_physics_substeps=int(substeps),
        recorded_physics_substeps_per_control=int(env.CONTROL_NSTEP))
    count = len(env.samples)
    if not env.orient(HANDLE_ROTATION, 0.25, "recover"):
        raise RecoveryContextRejected("Safe policy-state orientation recovery failed")
    return count


def _replay_object_empty_grasp_context(task, env, source):
    """Replay a certified drawer prefix and expose only corrective expert labels."""
    from .eval_policy_client import GripperCommandFilter
    grip = GripperCommandFilter(required_steps=1)
    replay_error = 0.0
    for row in source["rows"][:source["intervention_step"]]:
        replay_error = max(replay_error, float(np.max(
            np.abs(env.robot_state() - np.asarray(row["state_before"])))))
        if replay_error > 1e-4:
            raise RecoveryContextRejected("Recorded object prefix start does not reproduce")
        env.phase = "policy_context"
        env.step(task.prepare_action(row["raw_action"], grip))
        replay_error = max(replay_error, float(np.max(
            np.abs(env.robot_state() - np.asarray(row["state_after"])))))
        if replay_error > 1e-4 or env.failure_events or env.was_lifted:
            raise RecoveryContextRejected("Object correction requires a safe, unlifted physical prefix")
    opening = float(env.data.qpos[env.model.joint(
        f"drawer_slide{env.target}").qposadr[0]])
    aperture = float(sum(env.robot_state()[6:8]))
    if (env.first_selected_drawer != env.target or opening < .165
            or env.contacts()[0] or aperture >= .006 or env.gripper > .2):
        raise RecoveryContextRejected("Recorded object empty grasp did not physically reproduce")
    env._recorded_object_failure = dict(
        safe_object_empty_grasp_reproduced=True,
        recorded_intervention_sim_time=float(env.data.time),
        recorded_replay_max_state_error=replay_error,
        recorded_selected_drawer=int(env.first_selected_drawer),
        recorded_drawer_open_m=opening,
        recorded_aperture_m=aperture,
        recorded_gripper_command=float(env.gripper),
        recorded_object_contact=False,
    )
    context_length = len(env.samples)
    env.hold(1, 1.0, "recover")
    return context_length


def _continue_drawer_object_expert(env):
    """Collect the ordinary object/placement stages from an actual empty grasp."""
    from .task_env import OBJECT_ROTATION, OBJECT_GRASP_HAND_Z_OFFSETS_M
    receiving_xy = env.data.geom_xpos[env.model.geom("receiving_area").id, :2].copy()
    if not env.orient(OBJECT_ROTATION, 1.0, "recover"):
        return False
    obj = env.data.xpos[env.obj].copy()
    for pose, phase in [
        ([*env.data.xpos[env.hand, :2], obj[2] + .19], "object_clearance_lift"),
        (obj + [0, 0, .19], "object_approach"),
        (obj + [0, 0, OBJECT_GRASP_HAND_Z_OFFSETS_M[4]], "object_descend"),
    ]:
        if not env.move(pose, 1.0, phase):
            return False
    if not env.close_gripper("object_close"):
        env.motion_failures.append(dict(phase="object_close", reason="missing two-finger contact"))
        return False
    for pose, phase in [
        (obj + [-.03, 0, .205], "object_lift"),
        ([receiving_xy[0] - .02, receiving_xy[1], obj[2] + .205], "clear_drawer"),
        ([*receiving_xy, env.height + .26], "transport"),
        ([*receiving_xy, env.height + .15], "lower"),
    ]:
        if not env.move(pose, 0.0, phase):
            return False
    env.hold(round(1.0 / (env.model.opt.timestep * env.CONTROL_NSTEP)), 1.0, "release")
    if not env.move([*receiving_xy, env.height + .26], 1.0, "retreat"):
        return False
    return env.settle()


def _replay_empty_grasp_context(task, env, source):
    """Re-simulate recorded requests; verify an empty close before intervention."""
    from .eval_policy_client import GripperCommandFilter
    grip = GripperCommandFilter(required_steps=task.gripper_open_required_steps or 2)
    previous = 1.0
    failed_at = None
    for row in source["rows"]:
        env.phase = "policy_context"
        action = task.prepare_action(row["raw_action"], grip)
        env.step(action)
        if env.failure_events or env.was_lifted:
            break  # A strict expert episode cannot disguise an earlier safety/drop failure.
        aperture = sum(env.data.qpos[env.model.joint(n).qposadr[0]]
                       for n in ("finger_joint1", "finger_joint2"))
        if (failed_at is None and previous >= .5 and action[6] < .5
                and not env.contacts()[0] and aperture < .004):
            failed_at = float(env.data.time)
            env._recorded_failed_grasp_time = failed_at
        # Samples are pre-action observations. Retain a real, one-second
        # recorded response to the failure, not just the action that failed.
        # Never synthesize a pause or force the policy to fail again.
        if failed_at is not None and env.data.time - failed_at >= 1.0 - 1e-8:
            if env.contacts()[0]:
                raise RecoveryContextRejected("Recorded policy recovered before the requested expert intervention")
            count = len(env.samples)
            env.hold(1, 1.0, "recover")
            return count
        previous = action[6]
    raise RecoveryContextRejected(
        "Recorded requests did not reproduce a safe, empty grasp in this environment; "
        f"steps={len(env.samples)}, failure_events={env.failure_events}, "
        f"object_lifted={env.was_lifted}"
    )


def collect_episode(task, env, seed, max_steps, compact_static_frames=False, recovery_context=None):
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
        from .task_env import CollectionStepLimit
        env._collection_max_steps = max_steps
        env._collection_execution_steps = 0
        context_length = 0
        try:
            if recovery_context is not None:
                if task.number == 3:
                    replay = _replay_empty_grasp_context
                elif recovery_context.get("kind") == "object_empty_grasp":
                    replay = _replay_object_empty_grasp_context
                else:
                    replay = _replay_handle_deviation_context
                context_length = replay(task, env, recovery_context)
            succeeded = (_continue_drawer_object_expert(env)
                         if recovery_context is not None
                         and recovery_context.get("kind") == "object_empty_grasp"
                         else run_expert(env))
            if context_length:
                if recovery_context.get("kind") == "handle_policy_state":
                    env.policy_state_continuations = getattr(env, "policy_state_continuations", 0) + 1
                else:
                    env.natural_corrections = getattr(env, "natural_corrections", 0) + 1
        except CollectionStepLimit:
            succeeded = False
            env.motion_failures.append(dict(phase=env.phase, reason="collection_step_budget"))
        finally:
            del env._collection_max_steps
            del env._collection_execution_steps
        quality = trajectory_quality(env, succeeded)
        if recovery_context is not None:
            source_record = dict(
                path=recovery_context["path"], sha256=recovery_context["sha256"],
                run_id=recovery_context["run_id"], episode=recovery_context["episode"],
                history_only_frames=context_length,
            )
            if task.number == 3:
                source_record.update(
                    safe_empty_grasp_reproduced=bool(context_length),
                    recorded_empty_grasp_sim_time=getattr(env, "_recorded_failed_grasp_time", None),
                    recorded_response_seconds=float(
                        env.samples[context_length]["timestamp"] - env._recorded_failed_grasp_time)
                        if context_length else 0.0,
                )
            elif recovery_context.get("kind") == "object_empty_grasp":
                source_record.update(kind="object_empty_grasp",
                    intervention_step=recovery_context["intervention_step"],
                    **env._recorded_object_failure)
            elif recovery_context.get("kind") == "handle_policy_state":
                source_record.update(kind="handle_policy_state",
                    safe_handle_deviation_reproduced=False,
                    intervention_step=recovery_context["intervention_step"],
                    **env._recorded_policy_state_evidence)
            else:
                source_record.update(
                    kind=recovery_context.get("kind", "handle_approach_deviation"),
                    safe_handle_deviation_reproduced=bool(context_length),
                    deviation_measure="handle_tracking_component_or_orientation",
                    recorded_handle_deviation_m=getattr(env, "_recorded_handle_deviation_m", None),
                    recorded_handle_orientation_error_rad=getattr(env, "_recorded_handle_orientation_error_rad", None),
                    recorded_deviation_sim_time=getattr(env, "_recorded_handle_deviation_time", None),
                    recorded_replay_max_state_error=getattr(env, "_recorded_replay_state_error", None),
                    intervention_step=recovery_context["intervention_step"],
                )
                if recovery_context.get("kind") == "handle_contact_loss":
                    source_record.update(safe_handle_contact_loss_reproduced=bool(context_length),
                                         **env._recorded_handle_loss_evidence)
            quality["correction_source"] = source_record
            if recovery_context.get("kind") == "object_empty_grasp":
                # The policy opened the drawer in the excluded history. The
                # expert suffix contains object stages, not handle stages.
                quality["policy_context_checks"] = {
                    name: quality["checks"].pop(name)
                    for name in ("horizontal_handle_approach", "handle_contact_through_pull")
                }
                quality["failed_checks"] = [
                    name for name, passed in quality["checks"].items() if not passed
                ]
        # Human presentation contributes history, not robot execution steps.
        execution_steps = sum(r["phase"] not in ("observe_target", "presentation_close")
                              for r in env.samples)
        within_budget = (execution_steps <= max_steps and not any(
            row.get("reason") == "collection_step_budget" for row in env.motion_failures))
        quality["execution_length"] = execution_steps
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
    recovery_sources = (_recorded_recovery_sources(
                            args.recovery_actions, args.tasks[0],
                            source_kind=getattr(args, "recovery_kind", None))
                        if getattr(args, "recovery_actions", None) is not None else None)
    if recovery_sources is not None and getattr(args, "recovery_kind", None) is not None:
        recovery_sources = [source for source in recovery_sources if source.get("kind") == args.recovery_kind]
    if recovery_sources is not None and not args.overwrite:
        destination = args.dataset_dir or get_task(args.tasks[0]).dataset
        manifest = Path(destination) / "meta/episodes.jsonl"
        existing = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
        known = {
            (source.get("sha256"), source.get("episode"), source.get("kind"), source.get("intervention_step"))
            for row in existing
            for source in [row.get("quality", {}).get("correction_source") or {}]
        }
        recovery_sources = [source for source in recovery_sources if
            (source.get("sha256"), source.get("episode"), source.get("kind"), source.get("intervention_step")) not in known]
    if recovery_sources is not None and args.num_episodes is None:
        args.num_episodes = len(recovery_sources)
        if not recovery_sources:
            raise ValueError("No eligible recorded correction episodes are available")
    if recovery_sources is not None and args.num_episodes > len(recovery_sources):
        raise ValueError(f"Only {len(recovery_sources)} eligible recorded correction episodes are available")
    seed = (
        args.start_seed
        if args.start_seed is not None
        else secrets.randbelow(2**31 - 100000)
    )
    if recovery_sources is not None:
        seed = recovery_sources[0]["seed"]
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
        paired = (
            number == 4
            and recovery_sources is None
            and getattr(args, "paired_drawer_histories", None) is True
        )
        env = task.make_env(
            seed=seed,
            **({"target": 0} if paired else {}),
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
                    source_policy_version=contact_source_policy_version(number),
                )
            )
            writer = EpisodeDatasetWriter(
                dataset,
                fps=1.0 / (env.model.opt.timestep * env.CONTROL_NSTEP),
                image_size=args.image_size,
                collection_config=configuration,
                environment_contract=task_environment_contract(task, configuration),
                task_description=task.instruction,
                **writer_options,
            )
            saved = rejected = 0
            rejected_episodes = []
            attempt_results = []
            collection_run_id = uuid.uuid4().hex
            collection_started_at = datetime.now(timezone.utc).isoformat()
            def persist_collection(status):
                writer.record_collection_report(dict(
                    run_id=collection_run_id, process_id=os.getpid(),
                    started_at=collection_started_at,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    task=number, saved=saved, rejected=rejected, dataset=str(dataset),
                    attempts=len(attempt_results), status=status,
                    rejected_episodes=rejected_episodes, attempt_results=attempt_results,
                ))
            persist_collection("running")
            max_attempts = args.max_attempts or args.num_episodes * 20
            if recovery_sources is not None:
                max_attempts = min(max_attempts, len(recovery_sources))
            with tqdm(total=args.num_episodes, desc="accepted episodes") as progress:
                for attempt in range(max_attempts):
                    if saved >= args.num_episodes:
                        break
                    source = recovery_sources[attempt] if recovery_sources is not None else None
                    group, target = divmod(attempt, 4) if paired else (attempt, None)
                    attempt_seed = source["seed"] if source is not None else seed + group
                    attempt_result = dict(seed=attempt_seed, status="running", stage="initializing_environment")
                    if paired:
                        attempt_result["presented_drawer"] = target
                    attempt_results.append(attempt_result)
                    persist_collection("running")
                    try:
                        if attempt:
                            env.renderer.close()
                            env = task.make_env(
                                seed=attempt_seed,
                                **({"target": target} if paired else {}),
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
                        attempt_result["stage"] = "expert_execution"
                        episode, accepted, quality = collect_episode(
                            task,
                            env,
                            attempt_seed,
                            args.max_steps or task.max_steps,
                            args.compact_static_frames,
                            recovery_context=source,
                        )
                        if not accepted:
                            rejected += 1
                            rejection = dict(seed=attempt_seed,
                                failed_checks=quality.get("failed_checks", []),
                                motion_failures=quality.get("motion_failures", []))
                            rejection["category"] = (
                                "step_budget" if "within_collection_step_budget" in rejection["failed_checks"]
                                else "physical_failure" if not quality.get("success", True) or rejection["motion_failures"]
                                else "quality_rejection")
                            rejected_episodes.append(rejection)
                            attempt_result.update(status="rejected", **rejection)
                            persist_collection("running")
                            continue
                        attempt_result["stage"] = "encoding_and_writing"
                        row = writer.write_episode(
                            **episode, seed=attempt_seed, quality=quality, success=True
                        )
                        attempt_result.update(stage="validation", episode_index=row["episode_index"])
                        writer.validate_episode(row["episode_index"])
                        saved += 1
                        attempt_result["status"] = "accepted"
                        persist_collection("running")
                        progress.update(1)
                    except RecoveryContextRejected as error:
                        # A real policy prefix may be unsafe or no longer
                        # reproduce an empty grasp. Reject that attempt, just
                        # like an expert quality failure, and try the next
                        # recorded episode. Corrupt logs/programming errors
                        # still propagate through the BaseException path.
                        rejected += 1
                        rejection = dict(
                            seed=attempt_seed,
                            category="physical_failure",
                            failed_checks=["safe_empty_grasp_context"],
                            motion_failures=[],
                            error=str(error),
                        )
                        rejected_episodes.append(rejection)
                        attempt_result.update(status="rejected", **rejection)
                        persist_collection("running")
                    except BaseException as error:
                        attempt_result.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error",
                                              exception_type=type(error).__name__, error=str(error))
                        persist_collection(attempt_result["status"])
                        raise
            report = dict(
                task=number, saved=saved, rejected=rejected, dataset=str(dataset)
            )
            reports.append(report)
            persist_collection("completed" if saved >= args.num_episodes else "attempt_limit")
            print(f"Collected {saved} episodes; dataset={dataset}", flush=True)
            if saved < args.num_episodes:
                raise RuntimeError(
                    f"Only {saved}/{args.num_episodes} episodes passed "
                    f"within {max_attempts} attempts"
                )
        finally:
            env.renderer.close()
    return reports


async def _evaluate_tasks(numbers=None, argv=None, args=None, *, run_state):
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
    reports = run_state["episodes"]
    total_steps = 0
    if args.start_seed is None:
        client.log.info("Random evaluation start seed: %s", seed)
    diagnostics = Path(args.diagnostics_jsonl) if args.diagnostics_jsonl else None
    if diagnostics is not None:
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text("")
    async with websockets.connect(
        args.server_url, max_size=100_000_000, ping_timeout=None, proxy=None
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
        compatibility = {}
        for number in numbers:
            task = get_task(number)
            current = task_environment_contract(task, randomize_task=args.randomize_task,
                                                 randomization_scale=args.randomization_scale)
            checked = compare_environment_contract(
                (metadata.get("task_environment_contracts") or {}).get(task.key), current)
            compatibility[task.key] = checked
            if checked["status"] == "mismatch":
                raise ValueError(f"Task{number} checkpoint/environment mismatch: {checked['differences']}")
            if checked["status"] == "unverified":
                client.log.warning("Task%s checkpoint/environment compatibility unverified: %s", number, checked["reason"])
        run_state["environment_compatibility"] = compatibility
        execution_horizon, precision_replan = client.resolve_execution_policy(
            args, metadata
        )
        output.mkdir(parents=True, exist_ok=True)
        run_state["started"] = True
        _persist_evaluation_summary(run_state, "running")
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
            _atomic_json(directory / "summary.json", [])
            action_trace = directory / "actions.jsonl"
            action_trace.write_text("")
            for episode in range(args.num_episodes):
                episode_seed = seed + episode
                run_state["active_episode"] = dict(task=number, episode=episode,
                                                   seed=episode_seed, steps=0, stage="initialization")
                _persist_evaluation_summary(run_state, "running")
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
                    required_steps=task.gripper_open_required_steps or args.gripper_debounce_steps
                )
                precision = client.PrecisionExecutionController()
                done = False
                step = 0
                first_failure_step = None
                first_failure_history = None
                termination_reason = None
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
                        f"\n===== Task {number} | Episode {episode + 1} =====",
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
                    task_budget = task.evaluation_max_steps or task.max_steps
                    max_steps = min(args.max_steps or task_budget, task_budget)
                    while step < max_steps and not done:
                        run_state["active_episode"].update(steps=step, stage="inference")
                        payload = history.payload(task)
                        if metadata.get("native_feedback_episode_protocol") == 1 and number in (3, 4):
                            payload["episode_id"] = f"{run_state['run_id']}:{number}:{episode}"
                            payload["episode_step"] = step
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
                                run_id=run_state["run_id"],
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
                            run_state["active_episode"].update(steps=step, stage="execution")
                            if number == 1:
                                action_record = dict(
                                    kind="action", run_id=run_state["run_id"], task=number,
                                    episode=episode, step=step,
                                    simulation_time=float(env.data.time),
                                    history_times=[r[0] for r in history.selected()[0]],
                                    **task1_action_evidence(env, raw_action, action, before_state,
                                                            observation["robot_state"]),
                                )
                                client.append_diagnostic(action_trace, action_record)
                                client.append_diagnostic(diagnostics, action_record)
                            else:
                                from .task_env import rotation_matrix, rotation_vector

                                after_state = observation["robot_state"]
                                executed = env.samples[-1]["action"]
                                actual_rotation = (
                                    rotation_matrix(after_state[3:6])
                                    @ rotation_matrix(before_state[3:6]).T
                                )
                                row = env.rows[-1]
                                state_bounds = metadata.get("state_normalization_contracts", {}).get(task.key)
                                state_excess = None
                                if state_bounds:
                                    state_excess = np.maximum(
                                        np.maximum(np.asarray(state_bounds["min"][:len(before_state)]) - before_state,
                                                   before_state - np.asarray(state_bounds["max"][:len(before_state)])), 0.0
                                    ).tolist()
                                client.append_diagnostic(
                                    action_trace,
                                    action_record := dict(
                                        kind="action",
                                        run_id=run_state["run_id"],
                                        task=number,
                                        episode=episode,
                                        step=step,
                                        simulation_time=float(env.data.time),
                                        history_times=[r[0] for r in history.selected()[0]],
                                        raw_action=raw_action.tolist(),
                                        executed_action=executed.tolist(),
                                        state_before=before_state.tolist(),
                                        state_after=after_state.tolist(),
                                        state_normalization_excess=state_excess,
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
                                                "drawer_positions",
                                                "handle_axial_tracking_error_m",
                                                "handle_vertical_tracking_error_m",
                                                "handle_lateral_tracking_error_m",
                                                "handle_two_contact_physics_steps",
                                                "longest_handle_contact_gap_seconds",
                                                "finger_table_contacts",
                                                "palm_table_contacts",
                                                "table_penetration_m",
                                                "unexpected_robot_contacts_peak",
                                                "unpermitted_table_contact",
                                                "peak_table_contact_force_n",
                                                "first_unpermitted_table_evidence",
                                            )
                                        },
                                        ik_status=row.get("ik_status"),
                                        pose_tracking=row.get("pose_tracking"),
                                        outcome=env.outcome(),
                                    ),
                                )
                                client.append_diagnostic(diagnostics, action_record)
                            history.append(observation, env.data.time)
                            reward = 1.0 if done else 0.0
                            print(
                                f"[Step {step}] reward={reward:.2f}, done={done}",
                                flush=True,
                            )
                            failed_quality = number != 1 and bool(env.failure_events)
                            if failed_quality and first_failure_step is None:
                                first_failure_step = step
                                first_failure_history = target_history_evidence(env, history) if number == 4 else None
                            termination_reason = (
                                evaluation_failure_stop(
                                    env, step - first_failure_step if first_failure_step is not None else 0,
                                    getattr(args, "recovery_observation_steps", 0),
                                ) if number != 1 else None
                            )
                            if done or termination_reason:
                                if done:
                                    print("Task completed", flush=True)
                                break
                        if termination_reason:
                            break
                    report = dict(
                        run_id=run_state["run_id"],
                        task=number,
                        episode=episode,
                        seed=episode_seed,
                        success=bool(done),
                        steps=step,
                        checkpoint_metadata=metadata,
                        checkpoint_identity_verified=bool(metadata.get("checkpoint_identity", {}).get("weights_sha256")
                                                          if metadata.get("checkpoint_identity") else False),
                        environment_compatibility=compatibility[task.key],
                    )
                    if number != 1:
                        import mujoco

                        report.update(
                            **env.outcome(),
                            mujoco_version=mujoco.__version__,
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
                                if number == 4 and env.first_selected_drawer is not None
                                else None
                            ),
                            natural_corrections=None,
                            recovery_observation_steps=(
                                step - first_failure_step if first_failure_step is not None else 0
                            ),
                            physical_complete_after_failure=bool(env.failure_events and env.outcome()["physical_complete"]),
                            recovery_elapsed_sim_seconds=(
                                float(env.data.time - env.failure_events[0]["simulation_time"])
                                if env.failure_events and "simulation_time" in env.failure_events[0] else None),
                            target_history_at_first_failure=first_failure_history,
                            target_history_at_end=target_history_evidence(env, history) if number == 4 else None,
                        )
                        if termination_reason:
                            report["termination_status"] = termination_reason
                        if report["termination_status"] in ("running", "quality_failure"):
                            report["termination_status"] = "step_limit"
                    fps = video_frames_per_step / (
                        env.model.opt.timestep * env.CONTROL_NSTEP
                    )
                    run_state["active_episode"]["stage"] = "video_encoding"
                    imageio.mimwrite(
                        directory / f"episode_{episode:04d}.mp4",
                        frames,
                        fps=fps,
                        macro_block_size=1,
                    )
                    reports.append(report)
                    _atomic_json(directory / "summary.json", [r for r in reports if r["task"] == number])
                    run_state["active_episode"] = None
                    _persist_evaluation_summary(run_state, "running")
                    if args.render:
                        client.maybe_show(frames[-1], True)
                    client.append_diagnostic(
                        diagnostics, dict(kind="episode_summary", **report)
                    )
                    total_steps += step
                    result_text = "✅ Success" if done else "❌ Fail"
                    client.log.info(
                        f"Task {number} | Episode {episode + 1}: {result_text}"
                    )
                finally:
                    history.clear()
                    env.renderer.close()
            task_success = sum(r["success"] for r in reports if r["task"] == number)
            client.log.info(
                f"========= Task {number} Summary: {task_success}/{args.num_episodes} Successful ========="
            )
    summary = _persist_evaluation_summary(run_state, "completed")
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


async def evaluate_tasks(numbers=None, argv=None, args=None):
    """Keep durable progress on ordinary errors, cancellation and interruption."""
    from .eval_policy_client import parse_args
    args = args if args is not None else parse_args(argv)
    selected = list(args.tasks if numbers is None else numbers)
    state = dict(run_id=uuid.uuid4().hex, started_utc=datetime.now(timezone.utc).isoformat(),
                 tasks=selected, requested_episodes_per_task=args.num_episodes,
                 output=Path(args.video_dir).resolve(), episodes=[], started=False)
    try:
        return await _evaluate_tasks(selected, args=args, run_state=state)
    except BaseException as error:
        if state["started"]:
            _persist_evaluation_summary(state, "interrupted", error)
        raise
