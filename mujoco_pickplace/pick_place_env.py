import numpy as np
import mujoco
from pathlib import Path
import tempfile

SCENE_XML = Path(__file__).with_name("pick_place_scene.xml")
PANDA_DIR = Path(__file__).resolve().parents[1] / "mujoco_menagerie" / "franka_emika_panda"
PANDA_INCLUDE = "../mujoco_menagerie/franka_emika_panda/panda.xml"


def _load_model():
    scene_xml = SCENE_XML.read_text(encoding="utf-8")
    scene_xml = scene_xml.replace(PANDA_INCLUDE, "panda.xml")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".xml",
            prefix=".evo1_pick_place_",
            dir=PANDA_DIR,
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(scene_xml)
        return mujoco.MjModel.from_xml_path(str(temporary_path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class PickPlaceEnv:
    ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
    ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
    HOME_QPOS = np.array(
        [0.0, 0.0, 0.0, -np.pi / 2.0,
        0.0, np.pi / 2.0, -np.pi / 4.0],
        dtype=np.float64,
    )
    CONTROL_NSTEP = 100
    MAX_JOINT_TARGET_DELTA = 0.04

    MAX_DPOS = 0.012
    FINGER_TRAVEL = 0.04
    RED_NAIL_Z = 0.125
    GRASP_OFFSET = RED_NAIL_Z
    GRASP_X_BIAS = 0.006
    PLACE_Z = 0.195
    SAFE_Z = 0.280
    GRASP_CLOSE_TOL = 0.014
    GRASP_GEOM_XY_TOL = 0.006
    GRASP_GEOM_Z_TOL = 0.004
    GRASP_CLOSE_XY_TOL = GRASP_CLOSE_TOL
    GRASP_CLOSE_Z_TOL = GRASP_GEOM_Z_TOL
    GRASP_GEOM_FINGER_OPEN_MAX = 0.039
    GRASP_HOLD_FORCE_KP = 55.0
    GRASP_HOLD_FORCE_KD = 1.8
    GRASP_HOLD_MAX_FORCE = 4.0
    TABLE_TOP = 0.03
    CUBE_HALF = 0.030
    CUBE_SUPPORT_Z = TABLE_TOP + CUBE_HALF
    GOAL_RADIUS = 0.075
    MIN_hand_Z = TABLE_TOP + RED_NAIL_Z + 0.010
    HAND_LOW = np.array([0.20, -0.30, MIN_hand_Z], dtype=np.float64)
    HAND_HIGH = np.array([0.66, 0.30, 0.42], dtype=np.float64)
    CUBE_XY_LOW = np.array([0.30, -0.14], dtype=np.float64)
    CUBE_XY_HIGH = np.array([0.40, -0.06], dtype=np.float64)
    GOAL_XY_LOW = np.array([0.38, 0.08], dtype=np.float64)
    GOAL_XY_HIGH = np.array([0.52, 0.17], dtype=np.float64)
    FIXED_CUBE_XY = np.array([0.35, -0.10], dtype=np.float64)
    FIXED_GOAL_XY = np.array([0.45, 0.11], dtype=np.float64)
    FINGER_JOINTS = ("finger_joint1", "finger_joint2")
    FINGER_BODIES = ("left_finger", "right_finger")
    SUCCESS_DWELL_STEPS = 8
    SUCCESS_MAX_CUBE_SPEED = 0.020
    EXPERT_TRANSFER_X_ALIGN_TOL = 0.020
    EXPERT_LOWER_XY_TOL = 0.006
    EXPERT_LOWER_Z_TOL = 0.010

    def __init__(self, image_size=448, randomize_task=False):
        self.model = _load_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
        self.randomize_task = bool(randomize_task)
        self.arm_joints = self.ARM_JOINTS
        self.arm_actuators = self.ARM_ACTUATORS
        self.arm_qpos_ids = np.array([
            self.model.joint(name).qposadr[0] for name in self.arm_joints
        ])
        self.arm_dof_ids = np.array([
            self.model.joint(name).dofadr[0] for name in self.arm_joints
        ])
        self.arm_ranges = np.array([
            self.model.joint(name).range for name in self.arm_joints
        ])
        self.hand_body_id = self.model.body("hand").id
        self.cube_body_id = self.model.body("cube").id
        self.cube_qpos_id = self.model.joint("cube_free").qposadr[0]
        self.cube_dof_id = self.model.joint("cube_free").dofadr[0]
        self.finger_joints = self.FINGER_JOINTS
        self.finger_body_ids = tuple(
            self.model.body(name).id for name in self.FINGER_BODIES
        )
        self.finger_actuator = "actuator8"
        self.gripper = 1.0
        self.attached = False
        self.release_counter = 0
        self.success_counter = 0
        self.unsafe_robot_table_contact = False
        self.tool_z_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.target_hand = np.array([0.105, -0.18, 0.225], dtype=np.float64)
        self.arm_target = self.HOME_QPOS.copy()
        self._configure_task_physics()

    @property
    def grasp_hold_offset(self):
        return np.array([
            self.GRASP_X_BIAS,
            0.0,
            self.GRASP_OFFSET,
        ], dtype=np.float64)

    def _configure_task_physics(self):
        for body_id in self.finger_body_ids:
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            for geom_id in geom_ids:
                if self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]:
                    self.model.geom_friction[geom_id] = [8.0, 0.5, 0.01]
                    self.model.geom_solref[geom_id] = [0.006, 0.6]
                    self.model.geom_solimp[geom_id] = [0.95, 0.99, 0.001, 0.5, 2.0]
        actuator_id = self.model.actuator(self.finger_actuator).id
        self.model.actuator_forcerange[actuator_id] = [-250.0, 250.0]

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize_task:
            cube_xy = rng.uniform(self.CUBE_XY_LOW, self.CUBE_XY_HIGH)
            goal_xy = rng.uniform(self.GOAL_XY_LOW, self.GOAL_XY_HIGH)
        else:
            cube_xy = self.FIXED_CUBE_XY.copy()
            goal_xy = self.FIXED_GOAL_XY.copy()
        self.model.body("goal").pos[:] = [goal_xy[0], goal_xy[1], 0.035]

        self.data.qpos[self.cube_qpos_id:self.cube_qpos_id + 7] = [
            cube_xy[0], cube_xy[1], self.CUBE_SUPPORT_Z,
            1.0, 0.0, 0.0, 0.0,
        ]

        self.data.qpos[self.arm_qpos_ids] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self.tool_z_ref[:] = self.data.xmat[self.hand_body_id].reshape(3, 3)[:, 2]
        self.target_hand[:] = self.data.body("hand").xpos
        arm_targets = self._solve_ik(self.target_hand, self.HOME_QPOS)
        self.data.qpos[self.arm_qpos_ids] = arm_targets
        for joint_name in self.finger_joints:
            self.data.qpos[self.model.joint(joint_name).qposadr[0]] = self.FINGER_TRAVEL

        self.gripper = 1.0
        self.attached = False
        self.release_counter = 0
        self.success_counter = 0
        self.unsafe_robot_table_contact = False
        self.arm_target = arm_targets.copy()
        self._set_actuator_targets(arm_targets)
        self._apply_grasp_force()
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        hand = self.data.body("hand").xpos.copy()
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()
        state = np.concatenate([
            hand, cube, goal,
            np.array([self.gripper], dtype=np.float32),
        ]).astype(np.float32)

        front = self.render("front")
        return {
            "state": state,
            "robot_state": self.robot_state(),
            "image": front,
            "image_front": front,
        }

    def robot_state(self):
        left_qpos = self.data.qpos[self.model.joint("finger_joint1").qposadr[0]]
        right_qpos = self.data.qpos[self.model.joint("finger_joint2").qposadr[0]]
        return np.concatenate([
            self.data.body("hand").xpos.copy(),
            self._hand_axis_angle(),
            np.array([left_qpos, right_qpos], dtype=np.float64),
        ]).astype(np.float32)

    def render(self, camera="front"):
        if camera == "wrist":
            camera = "workspace"
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _hand_axis_angle(self):
        quat = self.data.xquat[self.hand_body_id].copy()
        quat = quat / (np.linalg.norm(quat) + 1e-12)
        w = float(np.clip(quat[0], -1.0, 1.0))
        den = np.sqrt(max(1.0 - w * w, 0.0))
        if den < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return quat[1:4] * (2.0 * np.arccos(w) / den)

    @staticmethod
    def _action_gripper(action):
        if len(action) >= 7:
            return action[6]
        if len(action) >= 4:
            return action[3]
        raise ValueError("PickPlaceEnv action must have 4 or 7 dimensions")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -self.MAX_DPOS, self.MAX_DPOS)
        self.gripper = float(np.clip(self._action_gripper(action), 0.0, 1.0))

        current_hand = self.data.body("hand").xpos.copy()
        self.target_hand[:] = np.clip(
            current_hand + dpos,
            self.HAND_LOW,
            self.HAND_HIGH,
        )
        arm_targets = self._solve_ik(
            self.target_hand,
            self.arm_target.copy(),
        )
        self._set_actuator_targets(arm_targets)
        self._apply_grasp_force()
        mujoco.mj_step(self.model, self.data, nstep=self.CONTROL_NSTEP)
        self.unsafe_robot_table_contact |= self.has_robot_table_contact()
        # Refresh the grasp state after the physics sweep. The cube position is
        # governed by MuJoCo contacts; this call no longer kinematically moves it.
        self._apply_grasp_force()
        self._update_grasp_logic()
        mujoco.mj_forward(self.model, self.data)

        obs = self.obs()
        done = self._update_success_counter()
        return obs, done

    def step_video(self, action, frames_per_step=4, cameras=("front",)):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -self.MAX_DPOS, self.MAX_DPOS)
        self.gripper = float(np.clip(self._action_gripper(action), 0.0, 1.0))
        current_hand = self.data.body("hand").xpos.copy()
        self.target_hand[:] = np.clip(
            current_hand + dpos,
            self.HAND_LOW,
            self.HAND_HIGH,
        )
        arm_targets = self._solve_ik(
            self.target_hand,
            self.arm_target.copy(),
        )
        self._set_actuator_targets(arm_targets)
        self._apply_grasp_force()

        # NOTE: _update_grasp_logic() is intentionally only applied after the
        # full sweep (exactly like step()) so the physics/obs are identical to
        # what the policy was trained on. The intermediate rendered frames show
        # the cube in its physical position.
        render_count = min(max(1, frames_per_step), self.CONTROL_NSTEP)
        render_steps = set(np.linspace(
            1,
            self.CONTROL_NSTEP,
            num=render_count,
            dtype=int,
        ))
        frames = {cam: [] for cam in cameras}
        for step_in_sweep in range(1, self.CONTROL_NSTEP + 1):
            self._apply_grasp_force()
            mujoco.mj_step(self.model, self.data)
            self.unsafe_robot_table_contact |= self.has_robot_table_contact()
            if step_in_sweep in render_steps:
                mujoco.mj_forward(self.model, self.data)
                for cam in cameras:
                    frames[cam].append(self.render(cam).copy())

        mujoco.mj_forward(self.model, self.data)
        self._update_grasp_logic()
        obs = self.obs()
        done = self._update_success_counter()
        return frames, obs, done

    def _set_actuator_targets(self, arm_targets):
        arm_targets = np.asarray(arm_targets, dtype=np.float64)
        arm_targets = self.arm_target + np.clip(
            arm_targets - self.arm_target,
            -self.MAX_JOINT_TARGET_DELTA,
            self.MAX_JOINT_TARGET_DELTA,
        )
        arm_targets = np.clip(
            arm_targets,
            self.arm_ranges[:, 0],
            self.arm_ranges[:, 1],
        )
        self.arm_target = arm_targets.copy()
        for actuator_name, target in zip(self.arm_actuators, arm_targets):
            self.data.ctrl[self.model.actuator(actuator_name).id] = target

        finger_target = 255.0 * self.gripper
        self.data.ctrl[self.model.actuator(self.finger_actuator).id] = finger_target

    def _solve_ik(self, target, seed_q, target_z=None):
        original_q = self.data.qpos[self.arm_qpos_ids].copy()
        q = np.clip(np.asarray(seed_q, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        target = np.asarray(target, dtype=np.float64)
        if target_z is None:
            target_z = self.tool_z_ref
        target_z = np.asarray(target_z, dtype=np.float64)
        target_z = target_z / (np.linalg.norm(target_z) + 1e-12)
        identity = np.eye(len(self.arm_joints))
        best_q = q.copy()
        best_cost = np.inf
        ori_weight = 0.35

        for _ in range(80):
            self.data.qpos[self.arm_qpos_ids] = q
            mujoco.mj_forward(self.model, self.data)

            pos_err = target - self.data.body("hand").xpos

            R = self.data.xmat[self.hand_body_id].reshape(3, 3)
            current_z = R[:, 2]
            axis = np.cross(current_z, target_z)
            s = np.linalg.norm(axis)
            c = np.clip(np.dot(current_z, target_z), -1.0, 1.0)
            if s < 1e-8:
                ori_err = np.zeros(3)
            else:
                axis = axis / s
                ori_err = axis * np.arctan2(s, c)

            if np.linalg.norm(pos_err) < 5e-4 and np.linalg.norm(ori_err) < 1e-3:
                best_q = q.copy()
                break

            cost = np.linalg.norm(pos_err) + 0.05 * np.linalg.norm(ori_err)
            if cost < best_cost:
                best_cost = cost
                best_q = q.copy()

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
            jac = np.vstack([
                jacp[:, self.arm_dof_ids],
                ori_weight * jacr[:, self.arm_dof_ids],
            ])
            error = np.concatenate([pos_err, ori_weight * ori_err])
            damping = 2e-3
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(6))
            delta = jac_pinv @ error
            nullspace = identity - jac_pinv @ jac
            delta += nullspace @ (0.01 * (self.HOME_QPOS - q))
            q += np.clip(delta, -0.08, 0.08)
            q = np.clip(q, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

        self.data.qpos[self.arm_qpos_ids] = original_q
        mujoco.mj_forward(self.model, self.data)
        return best_q

    def _update_grasp_logic(self):
        if self.gripper < 0.5:
            if (
                self.attached
                or self._has_two_sided_grasp_contact()
                or self._has_geometric_grasp()
            ):
                self.attached = True
                self.release_counter = 0
        else:
            self.release_counter += 1
            if self.release_counter >= 2:
                self.attached = False
                self.release_counter = 0
        self.data.xfrc_applied[self.cube_body_id] = 0.0

    def _apply_grasp_force(self):
        self.data.xfrc_applied[self.cube_body_id] = 0.0
        if not self.attached or self.gripper >= 0.5:
            return
        desired_cube = self.data.body("hand").xpos - self.grasp_hold_offset
        desired_cube[2] = max(desired_cube[2], self.CUBE_SUPPORT_Z + 0.002)
        cube = self.data.body("cube").xpos.copy()
        linear_velocity = self.data.qvel[self.cube_dof_id:self.cube_dof_id + 3]
        force = (
            self.GRASP_HOLD_FORCE_KP * (desired_cube - cube)
            - self.GRASP_HOLD_FORCE_KD * linear_velocity
        )
        force[2] += self.model.body_mass[self.cube_body_id] * abs(self.model.opt.gravity[2])
        force_norm = np.linalg.norm(force)
        if force_norm > self.GRASP_HOLD_MAX_FORCE:
            force *= self.GRASP_HOLD_MAX_FORCE / force_norm
        self.data.xfrc_applied[self.cube_body_id, :3] = force

    def _finger_qpos(self):
        return np.array([
            self.data.qpos[self.model.joint(name).qposadr[0]]
            for name in self.finger_joints
        ], dtype=np.float64)

    def _has_geometric_grasp(self):
        if self.gripper >= 0.5:
            return False
        if float(np.mean(self._finger_qpos())) > self.GRASP_GEOM_FINGER_OPEN_MAX:
            return False
        cube = self.data.body("cube").xpos.copy()
        desired_cube = self.data.body("hand").xpos - self.grasp_hold_offset
        xy_ok = np.linalg.norm(cube[:2] - desired_cube[:2]) < self.GRASP_GEOM_XY_TOL
        z_ok = abs(cube[2] - desired_cube[2]) < self.GRASP_GEOM_Z_TOL
        return bool(xy_ok and z_ok)

    def _has_two_sided_grasp_contact(self):
        cube_id = self.model.geom("cube_geom").id
        contacted = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist > 1e-3:
                continue
            if cube_id in (contact.geom1, contact.geom2):
                other_geom = contact.geom2 if contact.geom1 == cube_id else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                if other_body in self.finger_body_ids:
                    contacted.add(self.model.body(other_body).name)
        return contacted == set(self.FINGER_BODIES)

    def _tips_enclose_cube(self, tol=None, z_tol=None):
        if tol is None:
            tol = self.CUBE_HALF + 0.010
        if z_tol is None:
            z_tol = 0.034
        cube = self.data.body("cube").xpos
        tips = []
        for body_id in self.finger_body_ids:
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            collision_geoms = [
                geom_id for geom_id in geom_ids
                if self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]
            ]
            if not collision_geoms:
                tips.append(self.data.xpos[body_id].copy())
            else:
                z_sorted = sorted(
                    collision_geoms,
                    key=lambda geom_id: self.data.geom_xpos[geom_id, 2],
                    reverse=True,
                )
                tips.append(self.data.geom_xpos[z_sorted[0]].copy())
        for t in tips:
            if abs(t[2] - cube[2]) > z_tol:
                return False
            if np.linalg.norm(t[:2] - cube[:2]) > tol:
                return False
        sep = tips[0][:2] - tips[1][:2]
        mid = 0.5 * (tips[0][:2] + tips[1][:2])
        return float(np.dot(sep, cube[:2] - mid)) < 0.0

    def _success_candidate(self):
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()
        xy_ok = np.linalg.norm(cube[:2] - goal[:2]) <= self.GOAL_RADIUS
        z_ok = abs(cube[2] - self.CUBE_SUPPORT_Z) < 0.012
        cube_speed = np.linalg.norm(self.data.qvel[self.cube_dof_id:self.cube_dof_id + 3])
        released = self.gripper > 0.5 and not self.attached
        return bool(
            xy_ok
            and z_ok
            and released
            and cube_speed < self.SUCCESS_MAX_CUBE_SPEED
            and not self.unsafe_robot_table_contact
        )

    def has_robot_table_contact(self):
        table_id = self.model.geom("table").id
        robot_body_names = {"hand", "left_finger", "right_finger"}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if table_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == table_id else contact.geom1
            other_body = self.model.geom_bodyid[other_geom]
            if self.model.body(other_body).name in robot_body_names:
                return True
        return False

    def _update_success_counter(self):
        if self._success_candidate():
            self.success_counter += 1
        else:
            self.success_counter = 0
        return self.success()

    def success(self):
        return bool(self.success_counter >= self.SUCCESS_DWELL_STEPS)


def scripted_expert(obs):
    state = obs["state"]
    hand = state[0:3]
    cube = state[3:6]
    goal = state[6:9]
    gripper = state[9]

    safe_z = PickPlaceEnv.SAFE_Z
    grasp_z = cube[2] + PickPlaceEnv.GRASP_OFFSET
    place_z = PickPlaceEnv.PLACE_Z
    xy_tol = 0.025
    grasp_pose = np.array([cube[0] + PickPlaceEnv.GRASP_X_BIAS, cube[1], grasp_z], dtype=np.float32)
    place_pose = np.array([goal[0] + PickPlaceEnv.GRASP_X_BIAS, goal[1], place_z], dtype=np.float32)

    cube_xy = cube[:2]
    goal_xy = goal[:2]
    hand_xy = hand[:2]
    grasp_xy = grasp_pose[:2]
    place_xy = place_pose[:2]

    close_tol = 0.012

    if gripper > 0.5:
        if np.linalg.norm(hand - grasp_pose) > close_tol:
            if np.linalg.norm(hand_xy - grasp_xy) > xy_tol:
                target = np.array([cube[0] + PickPlaceEnv.GRASP_X_BIAS, cube[1], safe_z], dtype=np.float32)
            else:
                target = grasp_pose
            grip_cmd = 1.0
        else:
            target = hand.copy()
            grip_cmd = 0.0
    else:
        if hand[2] < safe_z - 0.010:
            target = np.array([hand[0], hand[1], safe_z], dtype=np.float32)
            grip_cmd = 0.0
        elif np.linalg.norm(hand_xy - place_xy) > xy_tol:
            target = np.array([goal[0] + PickPlaceEnv.GRASP_X_BIAS, goal[1], safe_z], dtype=np.float32)
            grip_cmd = 0.0
        elif hand[2] > place_z + 0.010:
            target = place_pose
            grip_cmd = 0.0
        else:
            target = hand.copy()
            grip_cmd = 1.0

    dpos = np.clip(target - hand, -0.012, 0.012)
    return np.r_[dpos, 0.0, 0.0, 0.0, grip_cmd].astype(np.float32)

class ScriptedExpertPolicy:

    def __init__(self, env):
        self.env = env
        self.phase = "approach"
        self.phase_steps = 0

    def _set_phase(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.phase_steps = 0

    def _move(self, target, gripper):
        hand = self.env.data.body("hand").xpos.copy()
        error = np.asarray(target, dtype=np.float64) - hand
        distance = np.linalg.norm(error)
        if distance < 1e-8:
            dpos = np.zeros(3, dtype=np.float64, )
        else:
            step_size = min(self.env.MAX_DPOS, 0.65 * distance, )
            dpos = error / distance * step_size

        return np.r_[dpos.astype(np.float32), 0.0, 0.0, 0.0, float(gripper), ].astype(np.float32)

    def _move_before_attachment(self, target, gripper):
        action = self._move(target, gripper)
        action[2] = min(float(action[2]), 0.0)
        return action

    def __call__(self, obs):
        self.phase_steps += 1
        state = obs["state"]
        hand, cube, goal = state[:3], state[3:6], state[6:9]
        safe_z = self.env.SAFE_Z
        grasp_z = cube[2] + self.env.GRASP_OFFSET
        place_z = self.env.PLACE_Z
        hand_cube_xy = np.array([cube[0] + self.env.GRASP_X_BIAS, cube[1]])
        hand_goal_xy = np.array([goal[0] + self.env.GRASP_X_BIAS, goal[1]])

        if self.phase == "approach":
            target = np.array([hand_cube_xy[0], hand_cube_xy[1], safe_z])
            if np.linalg.norm(hand - target) < 0.014:
                self._set_phase("descend")
                target = np.array([hand_cube_xy[0], hand_cube_xy[1], grasp_z])
            return self._move(target, 1.0)

        if self.phase == "descend":
            target = np.array([hand_cube_xy[0], hand_cube_xy[1], grasp_z], dtype=np.float64)
            xy_error = np.linalg.norm(hand[:2] - target[:2])
            z_error = abs(float(hand[2] - target[2]))
            grasp_pose_ready = (
                xy_error <= self.env.GRASP_CLOSE_XY_TOL
                and
                z_error <= self.env.GRASP_CLOSE_Z_TOL)

            action = self._move_before_attachment(
                target, 0.0 if grasp_pose_ready else 1.0)
            if 0.0 < hand[2] - target[2] < 0.020:
                action[2] = max(float(action[2]), -0.003)

            if grasp_pose_ready:
                self._set_phase("close")
                return action

            return action

        if self.phase == "close":
            grasp_target = np.array([hand_cube_xy[0], hand_cube_xy[1], grasp_z], dtype=np.float64)

            if self.env.attached:
                self._set_phase("lift")
                return self._move(np.array([hand[0], hand[1], safe_z], dtype=np.float64), 0.0)

            if self.phase_steps > 14:
                self._set_phase("approach")

                return self._move(np.array([hand[0], hand[1], safe_z], dtype=np.float64), 1.0)

            action = self._move_before_attachment(grasp_target, 0.0)
            if 0.0 < hand[2] - grasp_target[2] < 0.020:
                action[2] = max(float(action[2]), -0.003)
            return action

        if self.phase == "lift":
            target = np.array([hand[0], hand[1], safe_z])
            if hand[2] > safe_z - 0.012:
                self._set_phase("transfer")
                target = np.array([hand_goal_xy[0], hand_goal_xy[1], safe_z])
            return self._move(target, 0.0)

        if self.phase == "transfer":
            safe_target = np.array(
                [hand_goal_xy[0], hand_goal_xy[1], safe_z],
                dtype=np.float64,
            )

            # First finish the X alignment while staying at safe height.
            if (
                abs(hand[0] - hand_goal_xy[0])
                > self.env.EXPERT_TRANSFER_X_ALIGN_TOL
            ):
                target = np.array(
                    [hand_goal_xy[0], hand[1], safe_z],
                    dtype=np.float64,
                )
                return self._move(target, 0.0)

            # Then converge to the full safe pose above the goal.
            xy_error = np.linalg.norm(hand[:2] - safe_target[:2])
            z_error = abs(float(hand[2] - safe_z))
            ready_to_lower = (
                xy_error <= self.env.EXPERT_LOWER_XY_TOL
                and z_error <= self.env.EXPERT_LOWER_Z_TOL
            )

            if ready_to_lower:
                self._set_phase("lower")
                lower_target = np.array(
                    [hand_goal_xy[0], hand_goal_xy[1], place_z],
                    dtype=np.float64,
                )
                return self._move(lower_target, 0.0)

            return self._move(safe_target, 0.0)

        if self.phase == "lower":
            target = np.array([hand_goal_xy[0], hand_goal_xy[1], place_z])
            if np.linalg.norm(hand - target) < 0.010:
                self._set_phase("release")
                return self._move(hand, 1.0)
            return self._move(target, 0.0)

        if self.phase == "release":
            if self.phase_steps >= self.env.SUCCESS_DWELL_STEPS:
                self._set_phase("settle")
            return self._move(
                np.array([hand_goal_xy[0], hand_goal_xy[1], place_z]),
                1.0,
            )

        return self._move(
            np.array([hand_goal_xy[0], hand_goal_xy[1], safe_z]),
            1.0,
        )
