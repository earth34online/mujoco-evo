"""Native Panda pose control and contact inspection for the new tabletop tasks."""

from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

HOME = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, -np.pi / 4])
DRAWER_PARK = HOME.copy()
DRAWER_PARK[0] = 0.70
DRAWER_PITCH = 0.12
HANDLE_HEIGHT = 0.070
CABINET_RISER = 0.04
CABINET_Y = -0.10
CABINET_FRONT = 0.60
HANDLE_ROTATION = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OBJECT_ROTATION = np.diag([-1.0, 1.0, -1.0])


def scene_xml(task, height=0.03, target=0, scene_path=None):
    path = (
        Path(scene_path)
        if scene_path is not None
        else Path(__file__).parent / "assets" / f"task{task}_scene.xml"
    )
    scene = ET.parse(path).getroot()
    panda_path = (
        Path(__file__).resolve().parents[1]
        / "mujoco_menagerie/franka_emika_panda/panda.xml"
    )
    root = ET.parse(panda_path).getroot()
    root.remove(root.find("keyframe"))
    for body in root.find("worldbody").iter("body"):
        body.set("gravcomp", "1")
    for geom in root.find("worldbody").iter("geom"):
        name = geom.get("class", "")
        if name == "collision" or name.startswith("fingertip_pad_collision_"):
            geom.set("class", "task_contact_" + name)
    gripper = root.find("actuator/general[@name='actuator8']")
    gripper.set("class", f"task{task}_gripper")
    for attribute in ("gainprm", "biasprm", "forcerange", "ctrlrange"):
        gripper.attrib.pop(attribute, None)
    for section in scene:
        if section.tag == "include":
            continue
        current = root.find(section.tag)
        if current is None:
            root.append(section)
        elif section.tag in ("compiler", "option"):
            current.attrib.update(section.attrib)
        else:
            for child in section:
                current.append(child)
    panda = Path(__file__).resolve().parents[1] / "mujoco_menagerie/franka_emika_panda"
    root.find("compiler").set("meshdir", str(panda / "assets"))
    if task == 3:
        shift = height - 0.03
        for name in ("table",):
            geom = root.find(f"worldbody/geom[@name='{name}']")
            p = np.fromstring(geom.get("pos"), sep=" ")
            p[2] += shift
            geom.set("pos", " ".join(map(str, p)))
        for name in ("object", "bin"):
            body = root.find(f"worldbody/body[@name='{name}']")
            p = np.fromstring(body.get("pos"), sep=" ")
            p[2] += shift
            body.set("pos", " ".join(map(str, p)))
        for geom in root.findall("worldbody/geom"):
            if geom.get("name", "").startswith("leg_"):
                p = np.fromstring(geom.get("pos"), sep=" ")
                z = np.fromstring(geom.get("size"), sep=" ")
                p[2] += shift / 2
                z[2] += shift / 2
                geom.set("pos", " ".join(map(str, p)))
                geom.set("size", " ".join(map(str, z)))
    elif abs(height - 0.03) > 1e-9:
        raise ValueError("Task4 uses the validated fixed tabletop height")
    if task == 4:
        body = root.find("worldbody/body[@name='object']")
        p = np.fromstring(body.get("pos"), sep=" ")
        p[2] += DRAWER_PITCH * target
        body.set("pos", " ".join(map(str, p)))
    return ET.tostring(root, encoding="unicode")


class NativePanda:
    def __init__(
        self,
        task=3,
        height=0.03,
        target=0,
        object_xy=None,
        yaw=0.0,
        scene_path=None,
        image_size=448,
    ):
        self.task, self.height, self.target = task, height, target
        self.model = mujoco.MjModel.from_xml_string(
            scene_xml(task, height, target, scene_path)
        )
        self.data = mujoco.MjData(self.model)
        self.arm_q = np.array(
            [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
        )
        self.arm_v = np.array(
            [self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
        )
        self.ranges = np.array(
            [self.model.joint(f"joint{i}").range for i in range(1, 8)]
        )
        self.hand = self.model.body("hand").id
        self.obj = self.model.body("object").id
        self.object_geom = self.model.geom("object_geom").id
        self.object_q = self.model.joint("object_free").qposadr[0]
        self.fingers = [self.model.body(n).id for n in ("left_finger", "right_finger")]
        self.robot_bodies = {
            self.model.body(n).id
            for n in (
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "hand",
                "left_finger",
                "right_finger",
            )
        }
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
        self.data.qpos[self.arm_q] = HOME
        if object_xy is not None:
            self.data.qpos[self.object_q : self.object_q + 2] = object_xy
        self.data.qpos[self.object_q + 3 : self.object_q + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        for n in ("finger_joint1", "finger_joint2"):
            self.data.qpos[self.model.joint(n).qposadr[0]] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.rotation = self.data.xmat[self.hand].reshape(3, 3).copy()
        self.arm_target = HOME.copy()
        self.rows = []
        self.frames = []
        self.capture_frames = True
        self.phase = "initial"
        self.gripper = 1.0
        self.motion_failures = []
        if task == 4:
            # One common, target-independent observation pose. Establish the
            # reachable elbow branch during reset, before an episode starts.
            self.rotation = HANDLE_ROTATION.copy()
            # Fixed neutral branch, identical for every drawer target.
            self.arm_target = np.array(
                [
                    0.489108288,
                    -0.680687301,
                    -0.727604305,
                    -2.602739972,
                    2.8973,
                    2.690697232,
                    2.142461256,
                ]
            )
            self.data.qpos[self.arm_q] = self.arm_target
            mujoco.mj_forward(self.model, self.data)
        self.set_controls(self.arm_target, 1.0)
        mujoco.mj_step(self.model, self.data, nstep=100)
        mujoco.mj_forward(self.model, self.data)
        self.record()

    def set_controls(self, q, grip):
        for i, v in enumerate(q, 1):
            self.data.ctrl[self.model.actuator(f"actuator{i}").id] = v
        self.data.ctrl[self.model.actuator("actuator8").id] = 255 * grip

    def ik(self, xyz, seed=None, allow_fallback=True):
        if self.task == 4:
            return self.drawer_ik(xyz, seed)
        original = self.data.qpos[self.arm_q].copy()
        q = self.arm_target.copy() if seed is None else np.asarray(seed).copy()
        best = q.copy()
        best_cost = float("inf")
        for _ in range(100):
            self.data.qpos[self.arm_q] = q
            mujoco.mj_forward(self.model, self.data)
            pos = xyz - self.data.xpos[self.hand]
            R = self.data.xmat[self.hand].reshape(3, 3)
            # Quaternion error remains valid for large orientation differences.
            qr = np.empty(4)
            qt = np.empty(4)
            err = np.empty(3)
            mujoco.mju_mat2Quat(qr, R.ravel())
            mujoco.mju_mat2Quat(qt, self.rotation.ravel())
            mujoco.mju_subQuat(err, qt, qr)
            err = R @ err  # Quaternion difference is local; mj_jacBody is world-frame.
            cost = np.linalg.norm(pos) + 0.06 * np.linalg.norm(err)
            if cost < best_cost:
                best_cost = cost
                best = q.copy()
            if np.linalg.norm(pos) < 0.0003 and np.linalg.norm(err) < 0.001:
                break
            jp = np.zeros((3, self.model.nv))
            jr = jp.copy()
            mujoco.mj_jacBody(self.model, self.data, jp, jr, self.hand)
            J = np.vstack([jp[:, self.arm_v], 0.35 * jr[:, self.arm_v]])
            pinv = J.T @ np.linalg.inv(J @ J.T + 0.0001 * np.eye(6))
            dq = pinv @ np.r_[pos, 0.35 * err]
            q = np.clip(
                q + np.clip(dq, -0.08, 0.08), self.ranges[:, 0], self.ranges[:, 1]
            )
        self.data.qpos[self.arm_q] = original
        mujoco.mj_forward(self.model, self.data)
        if allow_fallback and best_cost > 0.001:
            # A local solve at Panda HOME can hit a joint limit even when
            # another elbow/wrist branch reaches the requested pose.
            rng = np.random.default_rng(8)
            for trial in rng.uniform(
                self.ranges[:, 0], self.ranges[:, 1], size=(12, 7)
            ):
                candidate = self.ik(xyz, seed=trial, allow_fallback=False)
                self.data.qpos[self.arm_q] = candidate
                mujoco.mj_forward(self.model, self.data)
                pos = xyz - self.data.xpos[self.hand]
                qr = np.empty(4)
                qt = np.empty(4)
                err = np.empty(3)
                mujoco.mju_mat2Quat(qr, self.data.xmat[self.hand])
                mujoco.mju_mat2Quat(qt, self.rotation.ravel())
                mujoco.mju_subQuat(err, qt, qr)
                cost = np.linalg.norm(pos) + 0.06 * np.linalg.norm(err)
                if cost < best_cost:
                    best_cost = cost
                    best = candidate
                self.data.qpos[self.arm_q] = original
                mujoco.mj_forward(self.model, self.data)
                if best_cost < 0.0003:
                    break
        return best

    def drawer_ik(self, xyz, seed=None):
        original = self.data.qpos[self.arm_q].copy()
        q = self.arm_target.copy() if seed is None else np.asarray(seed).copy()
        damping = 0.0001

        def residual(q):
            self.data.qpos[self.arm_q] = q
            mujoco.mj_forward(self.model, self.data)
            return np.r_[
                self.data.xpos[self.hand] - xyz,
                0.1 * (self.data.xmat[self.hand].reshape(3, 3) - self.rotation).ravel(),
            ]

        r = residual(q)
        for _ in range(250):
            if np.linalg.norm(r[:3]) < 0.0002 and np.linalg.norm(r[3:]) < 0.0001:
                break
            jp = np.zeros((3, self.model.nv))
            jr = jp.copy()
            mujoco.mj_jacBody(self.model, self.data, jp, jr, self.hand)
            R = self.data.xmat[self.hand].reshape(3, 3).copy()
            angular = []
            for x, y, z in jr[:, self.arm_v].T:
                angular.append(
                    0.1 * (np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ R).ravel()
                )
            J = np.vstack([jp[:, self.arm_v], np.array(angular).T])
            gradient = J.T @ r
            free = ~(
                ((q <= self.ranges[:, 0] + 1e-7) & (gradient > 0))
                | ((q >= self.ranges[:, 1] - 1e-7) & (gradient < 0))
            )
            A = J[:, free]
            dq = np.zeros(7)
            dq[free] = -np.linalg.solve(
                A.T @ A + damping * np.eye(int(free.sum())), A.T @ r
            )
            dq *= min(1.0, 0.10 / max(np.max(np.abs(dq)), 1e-10))
            candidate = np.clip(q + dq, self.ranges[:, 0], self.ranges[:, 1])
            trial = residual(candidate)
            if trial @ trial < r @ r:
                q = candidate
                r = trial
                damping = max(1e-8, damping / 2)
            else:
                residual(q)
                damping = min(10.0, damping * 5)
            if np.max(np.abs(dq)) < 1e-8:
                break
        self.data.qpos[self.arm_q] = original
        mujoco.mj_forward(self.model, self.data)
        return q

    def render(self, camera="front"):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def contacts(self, geom=None):
        geom = self.object_geom if geom is None else geom
        sides = set()
        table = []
        deepest = 0.0
        static = [self.model.geom("table").id]
        for c in self.data.contact:
            if c.dist > 0:
                continue
            if geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == geom else c.geom1
                if self.model.geom_bodyid[other] in self.fingers:
                    sides.add(int(self.model.geom_bodyid[other]))
            for g, h in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in static and self.model.geom_bodyid[h] in self.fingers + [
                    self.hand
                ]:
                    table.append(int(h))
                    deepest = min(deepest, float(c.dist))
        return len(sides) == 2, len(table), deepest

    def table_contact_counts(self):
        """Separate fingertip/table contact from palm/table contact for evaluation."""
        table = self.model.geom("table").id
        fingers, palm = 0, 0
        for contact in self.data.contact:
            if contact.dist > 0 or table not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == table else contact.geom1
            body = self.model.geom_bodyid[other]
            fingers += int(body in self.fingers)
            palm += int(body == self.hand)
        return fingers, palm

    def is_finger_table_contact(self, geom1, geom2):
        table = self.model.geom("table").id
        return (geom1 == table and self.model.geom_bodyid[geom2] in self.fingers) or (
            geom2 == table and self.model.geom_bodyid[geom1] in self.fingers
        )

    def object_bounds(self):
        g = self.object_geom
        if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = self.model.geom_dataid[g]
            a = self.model.mesh_vertadr[mid]
            n = self.model.mesh_vertnum[mid]
            vertices = self.model.mesh_vert[a : a + n]
        else:
            vertices = (
                np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                * self.model.geom_size[g]
            )
        world = (
            vertices @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
        )
        return world.min(0).tolist(), world.max(0).tolist()

    def unexpected_contacts(self):
        allowed = {self.object_geom}
        if self.task == 4:
            allowed.update(self.model.geom(f"handle{k}").id for k in range(4))
        result = []
        for ct in self.data.contact:
            if ct.dist >= -0.00005:
                continue
            b1, b2 = self.model.geom_bodyid[ct.geom1], self.model.geom_bodyid[ct.geom2]
            if b1 not in self.robot_bodies and b2 not in self.robot_bodies:
                continue
            if (b1 in self.fingers and ct.geom2 in allowed) or (
                b2 in self.fingers and ct.geom1 in allowed
            ):
                continue
            result.append([int(ct.geom1), int(ct.geom2), float(ct.dist)])
        return result

    def record(
        self,
        peak_table=0,
        peak_depth=0,
        peak_unexpected=None,
        peak_inclination=None,
        peak_finger_table=0,
        peak_palm_table=0,
    ):
        two, table, depth = self.contacts()
        finger_table, palm_table = self.table_contact_counts()
        row = {
            "t": float(self.data.time),
            "phase": self.phase,
            "hand": self.data.xpos[self.hand].tolist(),
            "hand_rotation": self.data.xmat[self.hand].tolist(),
            "object": self.data.xpos[self.obj].tolist(),
            "two_finger_contact": two,
            "table_contacts": max(table, peak_table),
            "finger_table_contacts": max(finger_table, peak_finger_table),
            "palm_table_contacts": max(palm_table, peak_palm_table),
            "table_penetration_m": min(depth, peak_depth),
            "gripper": self.gripper,
            "fingers": [
                float(self.data.qpos[self.model.joint(n).qposadr[0]])
                for n in ("finger_joint1", "finger_joint2")
            ],
            "external_object_force": self.data.xfrc_applied[self.obj].tolist(),
        }
        if self.task == 4:
            row["drawer_positions"] = [
                float(self.data.qpos[self.model.joint(f"drawer_slide{k}").qposadr[0]])
                for k in range(4)
            ]
            row["two_finger_handle_contact"] = self.contacts(
                self.model.geom(f"handle{self.target}").id
            )[0]
            error = (
                self.data.geom_xpos[self.model.geom(f"handle{self.target}").id]
                - self.data.xpos[self.hand]
                - 0.1029 * self.data.xmat[self.hand].reshape(3, 3)[:, 2]
            )
            row["handle_grasp_tracking_error_m"] = float(np.linalg.norm(error))
            row["handle_normal_tracking_error_m"] = float(np.linalg.norm(error[[0, 2]]))
            row["handle_axial_tracking_error_m"] = float(abs(error[0]))
            row["handle_vertical_tracking_error_m"] = float(abs(error[2]))
            row["handle_lateral_tracking_error_m"] = float(abs(error[1]))
            row["tool_axis_inclination_deg"] = float(
                np.rad2deg(
                    np.arcsin(
                        np.clip(self.data.xmat[self.hand].reshape(3, 3)[2, 2], -1, 1)
                    )
                )
            )
        row["arm_joint_positions"] = self.data.qpos[self.arm_q].tolist()
        row["unexpected_robot_contacts"] = self.unexpected_contacts()
        row["unexpected_robot_contacts_peak"] = (
            peak_unexpected
            if peak_unexpected is not None
            else row["unexpected_robot_contacts"]
        )
        if self.task == 4:
            row["peak_tool_axis_inclination_deg"] = (
                abs(row["tool_axis_inclination_deg"])
                if peak_inclination is None
                else peak_inclination
            )
        self.rows.append(row)
        row["object_bounds_low"], row["object_bounds_high"] = self.object_bounds()
        row["object_rotation"] = self.data.xquat[self.obj].tolist()
        row["external_joint_force_max"] = float(np.abs(self.data.qfrc_applied).max())
        row["object_collision_penetration_m"] = min(
            [0.0]
            + [
                float(c.dist)
                for c in self.data.contact
                if self.object_geom in (c.geom1, c.geom2)
            ]
        )
        if self.capture_frames:
            self.frames.append(self.render())

    def _apply_pose(self, xyz, grip):
        self.gripper = grip
        q = self.ik(np.asarray(xyz), allow_fallback=self.task != 4)
        joint_increment = 0.08 if self.task == 4 else 0.04
        q = self.arm_target + np.clip(
            q - self.arm_target, -joint_increment, joint_increment
        )
        self.integrate(q, grip)

    def integrate(self, q, grip):
        self.gripper = grip
        self.arm_target = q
        self.set_controls(q, grip)
        peak = 0
        peak_finger_table = 0
        peak_palm_table = 0
        depth = 0.0
        unexpected = {}
        inclination = 0.0
        handle_contact_steps = 0
        handle_gap = 0
        longest_handle_gap = 0
        for _ in range(100):
            if hasattr(self, "_presentation_substep"):
                self._presentation_substep()
            mujoco.mj_step(self.model, self.data)
            _, n, d = self.contacts()
            peak = max(peak, n)
            finger_table, palm_table = self.table_contact_counts()
            peak_finger_table = max(peak_finger_table, finger_table)
            peak_palm_table = max(peak_palm_table, palm_table)
            depth = min(depth, d)
            if self.task == 4:
                contact = self.contacts(self.model.geom(f"handle{self.target}").id)[0]
                handle_contact_steps += int(contact)
                handle_gap = 0 if contact else handle_gap + 1
                longest_handle_gap = max(longest_handle_gap, handle_gap)
            for g, h, dist in self.unexpected_contacts():
                key = (g, h)
                if key not in unexpected or dist < unexpected[key][2]:
                    unexpected[key] = [g, h, dist]
            if self.task == 4:
                inclination = max(
                    inclination,
                    float(
                        abs(
                            np.rad2deg(
                                np.arcsin(
                                    np.clip(
                                        self.data.xmat[self.hand].reshape(3, 3)[2, 2],
                                        -1,
                                        1,
                                    )
                                )
                            )
                        )
                    ),
                )
        mujoco.mj_forward(self.model, self.data)
        self.record(
            peak,
            depth,
            list(unexpected.values()),
            inclination,
            peak_finger_table,
            peak_palm_table,
        )
        if self.task == 4:
            self.rows[-1]["handle_two_contact_physics_steps"] = handle_contact_steps
            self.rows[-1]["longest_handle_contact_gap_seconds"] = (
                longest_handle_gap * self.model.opt.timestep
            )

    def move(self, xyz, grip, phase, limit=65, tolerance=0.002):
        self.phase = phase
        xyz = np.array(xyz, float)
        for _ in range(limit):
            delta = xyz - self.data.xpos[self.hand]
            if np.linalg.norm(delta) < tolerance:
                return True
            # Same metres-per-control-step convention as Task1.
            norm = np.linalg.norm(delta)
            d = (
                np.clip(delta, -0.012, 0.012)
                if self.task == 4
                else delta * min(1, 0.010 / norm)
            )
            self._step_pose(self.data.xpos[self.hand] + d, grip)
        self.motion_failures.append(
            {
                "phase": phase,
                "target": xyz.tolist(),
                "actual": self.data.xpos[self.hand].tolist(),
            }
        )
        return False

    def hold(self, n, grip, phase):
        self.phase = phase
        for _ in range(n):
            self._step_pose(self.data.xpos[self.hand].copy(), grip)

    def orient(self, rotation, grip, phase):
        self.rotation = np.asarray(rotation, float)
        self.phase = phase
        pos = self.data.xpos[self.hand].copy()
        for _ in range(100):
            self._step_pose(pos, grip)
            if (
                np.linalg.norm(self.data.xmat[self.hand].reshape(3, 3) - self.rotation)
                < 0.01
            ):
                return True
        return False


def rotation_vector(rotation):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation).ravel())
    if quat[0] < 0:
        quat = -quat
    length = np.linalg.norm(quat[1:])
    return (
        np.zeros(3)
        if length < 1e-10
        else quat[1:] * (2 * np.arctan2(length, quat[0]) / length)
    )


def rotation_matrix(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-10:
        return np.eye(3)
    axis = np.asarray(vector) / angle
    quat = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


class PandaTaskEnv(NativePanda):
    CONTROL_NSTEP = 100
    MAX_DPOS = 0.012
    MAX_DROT = 0.10

    def __init__(
        self, task, seed=None, capture_frames=True, scene_path=None, image_size=448
    ):
        self.seed = seed
        rng = np.random.default_rng(seed)
        height = float(rng.uniform(0.03, 0.07)) if task == 3 else 0.03
        target = int(rng.integers(4)) if task == 4 else 0
        xy = rng.uniform([0.345, -0.115], [0.375, -0.085]) if task == 3 else None
        yaw = float(rng.uniform(-np.pi / 36, np.pi / 36)) if task == 3 else 0.0
        super().__init__(task, height, target, xy, yaw, scene_path, image_size)
        self.capture_frames = capture_frames
        self.samples = []
        self.execution_start = float(self.data.time)
        self.presentation_complete = task == 3
        self.last_visible_time = None
        self._target_visible_times = []
        self.first_selected_time = None
        self.first_selected_drawer = None
        self.was_lifted = False
        self.had_object_contact = False
        self.grasp_attempts = 0
        self.first_grasp_lifted = False
        self.transport_dropped = False
        self._lift_start = self.data.xpos[self.obj, 2]
        self._success_dwell = 0
        self.wrong_drawer_opened = False
        self.unsafe_execution_contact = False
        self.finger_table_contact_seen = False
        self._human_close_start = None
        if task == 4:
            # Common robot pose for all four histories, clear of the presented drawer.
            self.arm_target = self.drawer_ik(np.array([0.30, -0.10, 0.334]))
            self.data.qpos[self.arm_q] = self.arm_target
            self.data.qvel[:] = 0.0
            self.set_controls(self.arm_target, 1.0)
            joint = self.model.joint(f"drawer_slide{target}")
            self.data.qpos[joint.qposadr[0]] = 0.12
            self.data.qpos[self.object_q] -= 0.12
            mujoco.mj_forward(self.model, self.data)
            self.rows.clear()
            self.frames.clear()

    def _presentation_substep(self):
        """Declared simulated human closure, disabled before policy execution."""
        if self._human_close_start is None:
            return
        elapsed = self.data.time - self._human_close_start
        progress = np.clip(elapsed / 0.8, 0.0, 1.0)
        desired = 0.06 * (1.0 + np.cos(np.pi * progress))
        velocity = (
            -0.06 * np.pi / 0.8 * np.sin(np.pi * progress) if progress < 1 else 0.0
        )
        joint = self.model.joint(f"drawer_slide{self.target}")
        q, v = joint.qposadr[0], joint.dofadr[0]
        self.data.qfrc_applied[v] = np.clip(
            2000 * (desired - self.data.qpos[q]) + 30 * (velocity - self.data.qvel[v]),
            -20.0,
            20.0,
        )

    def presentation(self):
        """Yield observations of the human presentation. No target is sent to the model."""
        if self.task != 4 or self.presentation_complete:
            return
        self._record_target_visibility()
        self.phase = "observe_target"
        for _ in range(15):
            self._step_pose(self.data.xpos[self.hand].copy(), 1.0)
            self._record_target_visibility()
            yield self.obs()
        self._human_close_start = float(self.data.time)
        self.phase = "presentation_close"
        for _ in range(7):
            self.rotation = HANDLE_ROTATION.copy()
            self._step_pose(np.array([0.38, -0.10, 0.334]), 1.0)
            self._record_target_visibility()
            yield self.obs()
        self._human_close_start = None
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.presentation_complete = True
        self.execution_start = float(self.data.time)
        self.first_selected_time = None
        self.first_selected_drawer = None
        self._lift_start = float(self.data.xpos[self.obj, 2])

    def _record_target_visibility(self):
        # Simulator-only acceptance evidence; never part of robot_state or payload.
        self.renderer.enable_segmentation_rendering()
        segmentation = self.render("front")
        self.renderer.disable_segmentation_rendering()
        if np.any(segmentation[:, :, 0] == self.object_geom):
            self.last_visible_time = float(self.data.time)
            self._target_visible_times.append(float(self.data.time))

    def robot_state(self):
        return np.r_[
            self.data.xpos[self.hand],
            self._hand_axis_angle(),
            [
                self.data.qpos[self.model.joint(n).qposadr[0]]
                for n in ("finger_joint1", "finger_joint2")
            ],
        ].astype(np.float32)

    def _hand_axis_angle(self):
        # Same quaternion convention as the original Task1 robot_state.
        quat = self.data.xquat[self.hand].copy()
        quat /= np.linalg.norm(quat) + 1e-12
        w = float(np.clip(quat[0], -1.0, 1.0))
        den = np.sqrt(max(1.0 - w * w, 0.0))
        return np.zeros(3) if den < 1e-8 else quat[1:4] * (2 * np.arccos(w) / den)

    def obs(self):
        front = self.render("front")
        return {"robot_state": self.robot_state(), "image_front": front, "image": front}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(
                "Expected seven finite actions: metres, world rotation radians, gripper"
            )
        action = action.copy()
        action[:3] = np.clip(action[:3], -self.MAX_DPOS, self.MAX_DPOS)
        if self.task == 3:
            action[3:6] = 0.0
        else:
            length = np.linalg.norm(action[3:6])
            action[3:6] *= min(1.0, self.MAX_DROT / max(length, 1e-12))
            self.rotation = rotation_matrix(action[3:6]) @ self.data.xmat[
                self.hand
            ].reshape(3, 3)
        action[6] = np.clip(action[6], 0.0, 1.0)
        if self.task == 3 and self.gripper >= 0.5 and action[6] < 0.5:
            self.grasp_attempts += 1
        self.samples.append(
            {
                "state": self.robot_state(),
                "image": self.render("front"),
                "action": action,
                "phase": self.phase,
                "timestamp": float(self.data.time),
            }
        )
        self._apply_pose(self.data.xpos[self.hand] + action[:3], action[6])
        self._update_events()
        return self.obs(), self.success()

    def _step_pose(self, xyz, grip):
        current = self.data.xmat[self.hand].reshape(3, 3)
        goal_rotation = self.rotation.copy()
        action = np.r_[
            np.asarray(xyz) - self.data.xpos[self.hand],
            rotation_vector(goal_rotation @ current.T),
            grip,
        ]
        result = self.step(action)
        self.rotation = goal_rotation
        return result

    def safe_wrist_arc(self, grip):
        """Expert FK waypoints executed through exactly the policy's 7D pose interface."""
        scratch = mujoco.MjData(self.model)
        start = self.arm_target.copy()
        count = int(np.ceil(np.max(np.abs(DRAWER_PARK - start)) / 0.035))
        self.phase = "safe_wrist_arc"
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            scratch.qpos[self.arm_q] = start + fraction * (DRAWER_PARK - start)
            mujoco.mj_forward(self.model, scratch)
            goal = scratch.xpos[self.hand].copy()
            rotation = scratch.xmat[self.hand].reshape(3, 3).copy()
            for _ in range(25):
                self.rotation = rotation
                self._step_pose(goal, grip)
                if (
                    np.linalg.norm(goal - self.data.xpos[self.hand]) < 0.002
                    and np.linalg.norm(
                        rotation_vector(
                            rotation @ self.data.xmat[self.hand].reshape(3, 3).T
                        )
                    )
                    < 0.012
                ):
                    break
            else:
                self.motion_failures.append(
                    {"phase": self.phase, "reason": "pose waypoint unreachable"}
                )
                return False
        return True

    def _update_events(self):
        two = self.contacts()[0]
        self.had_object_contact |= two
        self.was_lifted |= two and self.data.xpos[self.obj, 2] > self._lift_start + 0.05
        if self.task == 3 and self.grasp_attempts == 1:
            self.first_grasp_lifted |= (
                two and self.data.xpos[self.obj, 2] > self._lift_start + 0.05
            )
        if self.was_lifted and self.gripper < 0.5 and not two:
            self.transport_dropped = True
        if self.task == 4 and self.presentation_complete:
            for k in range(4):
                selected = self.contacts(self.model.geom(f"handle{k}").id)[0]
                opened = (
                    self.data.qpos[self.model.joint(f"drawer_slide{k}").qposadr[0]]
                    > 0.015
                )
                handle_grasp = (
                    self.data.geom_xpos[self.model.geom(f"handle{k}").id]
                    - 0.1029 * HANDLE_ROTATION[:, 2]
                )
                aligned = (
                    np.linalg.norm(self.data.xpos[self.hand] - handle_grasp) < 0.020
                )
                if len(self.rows) > 1:
                    aligned &= (
                        abs(self.rows[-1]["hand"][2] - self.rows[-2]["hand"][2]) < 0.004
                    )
                self.wrong_drawer_opened |= opened and k != self.target
                if self.first_selected_time is None and (selected or opened or aligned):
                    self.first_selected_time = float(self.data.time)
                    self.first_selected_drawer = k
        if self.presentation_complete:
            row = self.rows[-1]
            self.finger_table_contact_seen |= bool(row["finger_table_contacts"])
            blocking_contacts = row["unexpected_robot_contacts_peak"]
            table_contact = row["table_contacts"]
            if self.task == 3:
                # Evaluation accepts a physically lifted and stably placed rod
                # despite fingertip/table contact. Expert quality remains strict.
                blocking_contacts = [
                    contact
                    for contact in blocking_contacts
                    if not self.is_finger_table_contact(contact[0], contact[1])
                ]
                table_contact = row["palm_table_contacts"]
            self.unsafe_execution_contact |= bool(table_contact or blocking_contacts)

    def success(self):
        low, high = map(np.asarray, self.object_bounds())
        if self.task == 3:
            placed = (
                np.all(low[:2] >= [0.506, -0.079])
                and np.all(high[:2] <= [0.594, 0.199])
                and self.height + 0.003 < low[2] < self.height + 0.012
            )
        else:
            placed = (
                np.all(low[:2] >= [0.29, -0.32])
                and np.all(high[:2] <= [0.43, -0.20])
                and self.height < low[2] < self.height + 0.010
            )
            placed &= self.first_selected_drawer == self.target
            placed &= all(
                self.data.qpos[self.model.joint(f"drawer_slide{k}").qposadr[0]] < 0.015
                for k in range(4)
                if k != self.target
            )
        stable = (
            np.linalg.norm(
                self.data.qvel[self.model.joint("object_free").dofadr[0] :][:3]
            )
            < 0.025
        )
        valid = bool(
            placed
            and stable
            and self.was_lifted
            and self.had_object_contact
            and not self.transport_dropped
            and not self.wrong_drawer_opened
            and not self.unsafe_execution_contact
            and self.gripper > 0.5
        )
        self._success_dwell = self._success_dwell + 1 if valid else 0
        return self._success_dwell >= 5

    def close(self):
        self.renderer.close()


def run_task3_expert(env):
    height = env.height
    env.natural_corrections = 0
    for attempt in range(2):
        obj = env.data.xpos[env.obj].copy()
        for pose, phase in [
            (np.r_[obj[:2], height + 0.22], "approach"),
            (np.r_[obj[:2], height + 0.1132], "descend"),
        ]:
            if not env.move(pose, 1.0, phase, tolerance=0.0015):
                return False
        env.hold(5, 0.0, "close")
        if env.contacts()[0]:
            break
        if attempt == 1:
            env.motion_failures.append(
                {"phase": "close", "reason": "missing two-finger contact"}
            )
            return False
        # Retry only after an actual failed close. Never manufacture a failure or quota.
        env.natural_corrections += 1
        env.hold(3, 1.0, "recover")
        if not env.move(
            np.r_[env.data.xpos[env.obj, :2], height + 0.22], 1.0, "recover"
        ):
            return False
    for pose, phase in [
        (np.r_[obj[:2], height + 0.23], "lift"),
        ([0.55, 0.06, height + 0.23], "transport"),
        ([0.55, 0.06, height + 0.18], "lower"),
    ]:
        if not env.move(pose, 0.0, phase):
            return False
    env.hold(5, 1.0, "release")
    if not env.move([0.55, 0.06, height + 0.24], 1.0, "retreat"):
        return False
    env.hold(8, 1.0, "settle")
    return env._success_dwell >= 5


def run_task4_expert(env):
    handle = env.model.geom(f"handle{env.target}").id
    grasp = env.data.geom_xpos[handle].copy() - 0.1029 * HANDLE_ROTATION[:, 2]
    for pose, phase in [(grasp - [0.02, 0, 0], "approach"), (grasp, "handle_approach")]:
        if not env.move(pose, 0.25, phase, tolerance=0.0003):
            return False
    env.phase = "handle_close"
    for _ in range(5):
        env.rotation = HANDLE_ROTATION.copy()
        env._step_pose(grasp, 0.0)
    if not env.contacts(handle)[0]:
        env.motion_failures.append(
            {"phase": "handle_close", "reason": "missing two-finger contact"}
        )
        return False
    if not env.move(grasp - [0.178, 0, 0], 0.0, "pull"):
        return False
    if env.data.qpos[env.model.joint(f"drawer_slide{env.target}").qposadr[0]] < 0.165:
        return False
    env.hold(5, 1.0, "handle_release")
    for pose, phase in [
        (env.data.xpos[env.hand] + [-0.01, -0.08, 0], "handle_withdraw")
    ]:
        if not env.move(pose, 1.0, phase):
            return False
    if not env.move(
        env.data.xpos[env.hand] + [0, 0, 0.09], 1.0, "handle_clearance_lift"
    ):
        return False
    if not env.move([0.37, -0.30, max(0.43, grasp[2] + 0.12)], 1.0, "handle_retreat"):
        return False
    if not env.safe_wrist_arc(1.0) or not env.orient(
        OBJECT_ROTATION, 1.0, "topdown_orientation"
    ):
        return False
    obj = env.data.xpos[env.obj].copy()
    for pose, phase in [
        ([obj[0], 0.15, obj[2] + 0.19], "object_side_approach"),
        (obj + [0, 0, 0.19], "object_approach"),
        (obj + [0, 0, 0.114], "object_descend"),
    ]:
        if not env.move(pose, 1.0, phase):
            return False
    env.hold(5, 0.0, "object_close")
    if not env.contacts()[0]:
        env.motion_failures.append(
            {"phase": "object_close", "reason": "missing two-finger contact"}
        )
        return False
    for pose, phase in [
        (obj + [-0.03, 0, 0.205], "object_lift"),
        ([0.34, -0.26, obj[2] + 0.205], "clear_drawer"),
        ([0.36, -0.26, env.height + 0.26], "transport"),
        ([0.36, -0.26, env.height + 0.15], "lower"),
    ]:
        if not env.move(pose, 0.0, phase):
            return False
    env.hold(5, 1.0, "release")
    if not env.move([0.36, -0.26, env.height + 0.26], 1.0, "retreat"):
        return False
    env.hold(8, 1.0, "settle")
    return env._success_dwell >= 5


SCHEMA = "mujoco-panda-contact"
VERSION = 1


def trajectory_quality(env, succeeded):
    selected_times = []
    visible_slots = []
    rows = [r for r in env.rows if r["t"] > env.execution_start + 1e-8]
    held = [
        r
        for r in rows
        if r["phase"] in ("lift", "transport", "object_lift", "clear_drawer", "lower")
    ]
    checks = {
        "complete_motion": not env.motion_failures,
        "two_finger_grasp": bool(env.had_object_contact),
        "object_lifted": bool(env.was_lifted),
        "transport_retained": not env.transport_dropped
        and bool(held)
        and all(r["two_finger_contact"] for r in held),
        "no_robot_table_contact": all(r["table_contacts"] == 0 for r in rows),
        "no_unexpected_robot_contact": all(
            not r["unexpected_robot_contacts_peak"] for r in rows
        ),
        "no_execution_external_force": all(
            not any(r["external_object_force"]) and r["external_joint_force_max"] == 0
            for r in rows
        ),
        "no_object_gravity_assistance": env.model.body_gravcomp[env.obj] == 0,
        "robot_actuators_only": env.model.nu == 8,
        "stable_placement": bool(succeeded),
    }
    if env.task == 4:
        handle = [
            r for r in rows if r["phase"] in ("handle_approach", "handle_close", "pull")
        ]
        pull = [r for r in rows if r["phase"] == "pull"]
        sample_times = np.array([r["timestamp"] for r in env.samples])
        if env.first_selected_time is not None:
            decision_time = (
                env.first_selected_time - env.CONTROL_NSTEP * env.model.opt.timestep
            )
            indices = (
                np.searchsorted(
                    sample_times,
                    decision_time - np.arange(5, -1, -1) + 1e-6,
                    side="right",
                )
                - 1
            )
            selected_times = sample_times[np.maximum(indices, 0)].tolist()
            visible_slots = [
                k
                for k, time in enumerate(selected_times)
                if indices[k] >= 0
                and any(
                    abs(time - visible) < 1e-6 for visible in env._target_visible_times
                )
            ]
        checks.update(
            {
                "horizontal_handle_approach": bool(handle)
                and max(r["peak_tool_axis_inclination_deg"] for r in handle) < 2.0,
                "handle_contact_through_pull": bool(pull)
                and all(
                    r.get("handle_two_contact_physics_steps", 0) > 0
                    and r["longest_handle_contact_gap_seconds"] <= 0.04
                    and r["handle_axial_tracking_error_m"] < 0.0085
                    and r["handle_vertical_tracking_error_m"] < 0.003
                    and r["handle_lateral_tracking_error_m"] < 0.015
                    for r in pull
                ),
                "no_wrong_drawer_opened": all(
                    all(
                        pos < 0.015
                        for k, pos in enumerate(r["drawer_positions"])
                        if k != env.target
                    )
                    for r in rows
                ),
                "correct_first_drawer": env.first_selected_drawer == env.target,
                "presentation_complete": env.presentation_complete,
                "selection_within_history": bool(visible_slots)
                and env.last_visible_time is not None
                and env.first_selected_time - env.last_visible_time <= 5.0 + 1e-6,
            }
        )
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "success": bool(succeeded),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "task_id": env.task,
        "raw_length": len(env.samples),
        "table_height_m": env.height,
        "first_grasp_success": bool(
            env.had_object_contact and getattr(env, "natural_corrections", 0) == 0
        ),
        "natural_corrections": getattr(env, "natural_corrections", 0),
        "transport_drop": bool(env.transport_dropped),
        "last_target_visible_sim_time": env.last_visible_time,
        "drawer_selected_sim_time": env.first_selected_time,
        "sampled_history_sim_times_at_selection": selected_times,
        "target_visible_history_slots_at_selection": visible_slots,
        "target_drawer_diagnostic_only": env.target if env.task == 4 else None,
        "motion_failures": env.motion_failures,
        "timestamps_preserve_control_time": True,
    }
