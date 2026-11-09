from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .episode_dataset import CONTACT_QUALITY_SCHEMA, CONTACT_QUALITY_VERSION

HOME = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, -np.pi / 4])
DRAWER_PITCH = 0.12
HANDLE_HEIGHT = 0.070
CABINET_RISER = 0.07
CABINET_Y = -0.03
CABINET_FRONT = 0.60
CABINET_YAW = np.pi / 6
CABINET_POSITION = np.array([0.6580384757729336, CABINET_Y, 0.03])
CABINET_ROTATION = np.array([
    [np.cos(CABINET_YAW), -np.sin(CABINET_YAW), 0.0],
    [np.sin(CABINET_YAW), np.cos(CABINET_YAW), 0.0],
    [0.0, 0.0, 1.0],
])
HANDLE_ROTATION = CABINET_ROTATION @ np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# The parallel gripper can grasp opposite box faces with either downward yaw.
# This symmetric choice avoids saturating joint 7 during the local wrist turn.
OBJECT_ROTATION = CABINET_ROTATION @ np.diag([1.0, -1.0, -1.0])
DRAWER_START_JITTER = np.array([0.008, 0.012, 0.010])
DRAWER_APPROACH_JITTER = np.array([0.003, 0.004, 0.004])
DRAWER_OBSERVATION_HAND_POSITION = np.array([0.29, -0.13, 0.42])
# The simulated human closure also provides time to reach a common, nearer
# start pose, independent of the presented drawer. This preserves K=6 history
# through real bilateral handle selection without speeding the contact phase.
DRAWER_EXECUTION_HAND_POSITION = np.array([0.34, -0.13, 0.34])
DRAWER_CONTROL_PERIOD_SECONDS = 0.1
# Hand-body origin above the object-body origin at expert grasp, in metres.
# These target the finger pads farther onto the object, retaining floor clearance.
OBJECT_GRASP_HAND_Z_OFFSETS_M = {3: 0.1047, 4: 0.104}
# One micrometre absorbs contact-solver/rotation rounding at a box wall.
# This is a numeric boundary allowance, not a grasp or placement margin.
PLACEMENT_BOUNDARY_EPS_M = 1e-6


class CollectionStepLimit(RuntimeError):
    """An expert exhausted the collector's robot execution budget."""


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
        # A 240 mm cooking chopstick, tapered from 8 to 10 mm, supported
        # on visible 12 mm rests. These are ordinary collision bodies.
        shift = height - 0.03
        for name in ("table",):
            geom = root.find(f"worldbody/geom[@name='{name}']")
            p = np.fromstring(geom.get("pos"), sep=" ")
            p[2] += shift
            geom.set("pos", " ".join(map(str, p)))
        for name in ("object", "bin", "chopstick_rest0", "chopstick_rest1"):
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
        # A room light must not follow the COM of a hidden drawer object.
        # Keep ordinary shading/shadows, but use fixed lights for this scene.
        for light in root.find("worldbody").iter("light"):
            light.set("mode", "fixed")
        body = root.find("worldbody/body[@name='object']")
        p = np.fromstring(body.get("pos"), sep=" ")
        p[2] += DRAWER_PITCH * target
        body.set("pos", " ".join(map(str, p)))
    return ET.tostring(root, encoding="unicode")


class NativePanda:
    CONTROL_NSTEP = 100
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
            if task == 3:
                for k, offset in enumerate((-0.085, 0.085)):
                    body = self.model.body(f"chopstick_rest{k}").id
                    self.model.body_pos[body, :2] = np.asarray(object_xy) + np.array(
                        [-np.sin(yaw) * offset, np.cos(yaw) * offset]
                    )
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
        self.ik_status = None
        self._consecutive_ik_rejections = 0
        self._handle_gap_seconds = 0.0
        self._handle_contact_seconds = 0.0
        self._object_contact_seconds = 0.0
        self._finger_touch_seconds = 0.0
        if task == 4:
            # Establish a common neutral elbow branch during reset. The
            # randomized observation pose is solved before the episode starts.
            self.rotation = HANDLE_ROTATION.copy()
            # Fixed neutral branch, identical for every drawer target.
            self.arm_target = np.array(
                [
                    0.9427260189686637,
                    -1.4704158712747544,
                    -1.1346626160111408,
                    -2.962699836648303,
                    0.3558672243265793,
                    3.3997699955080676,
                    -2.2627666630392693,
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
        # A backtracked command can be smaller than the normal solve tolerance.
        # Returning the unchanged seed immediately then labels a nonzero action
        # as executable although its FK never moves. Resolve small increments
        # relative to their size; keep the existing acceptance/safety limits.
        position_distance = float(np.linalg.norm(r[:3]))
        rotation_distance = float(np.linalg.norm(r[3:]))
        position_tolerance, rotation_tolerance = 0.0002, 0.0001
        if position_distance < position_tolerance and rotation_distance < rotation_tolerance:
            # Only refine targets the ordinary solver would accept without an
            # iteration. A tiny rotation residual must not tighten an ordinary
            # translation solve and lengthen the memory-critical approach.
            if position_distance > 1e-10:
                position_tolerance = min(position_tolerance, max(1e-6, position_distance * 0.1))
            if rotation_distance > 1e-10:
                rotation_tolerance = min(rotation_tolerance, max(1e-8, rotation_distance * 0.1))
        for _ in range(250):
            if np.linalg.norm(r[:3]) < position_tolerance and np.linalg.norm(r[3:]) < rotation_tolerance:
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
        # Report unweighted task-space errors, including stalled/limited solves.
        residual(q)
        position_error = float(np.linalg.norm(self.data.xpos[self.hand] - xyz))
        rotation_error = float(np.linalg.norm(rotation_vector(
            self.rotation @ self.data.xmat[self.hand].reshape(3, 3).T
        )))
        self.ik_status = dict(
            converged=position_error <= 0.001 and rotation_error <= 0.01,
            position_residual_m=position_error,
            rotation_residual_rad=rotation_error,
        )
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

    def controlled_fingertip_touch(self):
        """Allow only a brief, shallow, low-force distal pad touch during pickup.

        The pickup region is inferred from physical geometry, never an expert
        phase label. The same rule is recorded for expert and policy validation.
        """
        table = self.model.geom("table").id
        touches = []
        evidence = []
        max_force = 0.0
        eligible = self.task == 3 and not getattr(self, "was_lifted", False)
        eligible &= np.linalg.norm(
            self.data.xpos[self.hand, :2] - self.data.xpos[self.obj, :2]
        ) <= 0.025
        eligible &= self.data.xmat[self.hand].reshape(3, 3)[2, 2] < -0.98
        for i, ct in enumerate(self.data.contact):
            if ct.dist > 0 or table not in (ct.geom1, ct.geom2):
                continue
            other = ct.geom2 if ct.geom1 == table else ct.geom1
            body = int(self.model.geom_bodyid[other])
            if body not in self.robot_bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
            local = self.data.xmat[body].reshape(3, 3).T @ (
                ct.pos - self.data.xpos[body]
            )
            # Panda's distal pad ends at local z=0.053 m. Mesh/palm and
            # proximal finger contacts cannot acquire this exemption.
            distal_pad = self.model.geom_type[other] == mujoco.mjtGeom.mjGEOM_BOX
            touches.append(bool(
                body in self.fingers and distal_pad and local[2] >= 0.048
                and ct.dist >= -0.0005 and np.linalg.norm(force[:3]) <= 5.0
            ))
            evidence.append(dict(geom_id=int(other), body_id=body,
                local_contact_position_m=local.tolist(), penetration_m=float(ct.dist),
                contact_force_n=float(np.linalg.norm(force[:3])), distal_pad=bool(distal_pad)))
        self._finger_touch_seconds = (
            self._finger_touch_seconds + self.model.opt.timestep if touches else 0.0
        )
        self._table_touch_evidence = dict(contacts=evidence,
            duration_seconds=self._finger_touch_seconds, pickup_region=bool(eligible))
        return bool(touches and eligible and all(touches)
                    and self._finger_touch_seconds <= 0.10 + 1e-9), max_force

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

    def unexpected_contacts(self, data=None, minimum_distance=-0.00005):
        data = self.data if data is None else data
        allowed = {self.object_geom}
        if self.task == 4:
            allowed.update(self.model.geom(f"handle{k}").id for k in range(4))
        result = []
        for ct in data.contact:
            if ct.dist >= minimum_distance:
                continue
            b1, b2 = self.model.geom_bodyid[ct.geom1], self.model.geom_bodyid[ct.geom2]
            if b1 not in self.robot_bodies and b2 not in self.robot_bodies:
                continue
            # Empty closure brings the opposing finger pads into contact at
            # the joint stop. MuJoCo's soft constraint admits a small overlap;
            # this is ordinary gripper closure, not a collision with the scene.
            # Keep deep overlap, finger meshes and all other self-collisions.
            if (b1 in self.fingers and b2 in self.fingers and b1 != b2
                    and self.model.geom_type[ct.geom1] == mujoco.mjtGeom.mjGEOM_BOX
                    and self.model.geom_type[ct.geom2] == mujoco.mjtGeom.mjGEOM_BOX
                    and ct.dist >= -0.0005):
                continue
            if (b1 in self.fingers and ct.geom2 in allowed) or (
                b2 in self.fingers and ct.geom1 in allowed
            ):
                continue
            result.append([int(ct.geom1), int(ct.geom2), float(ct.dist)])
        return result

    def unsafe_pose_contacts(self, q, grip):
        """Check the robot's joint path against physical scene collisions.

        This does not select a drawer or supply a task goal. Expected finger
        contacts remain allowed; scene and robot self-collisions are rejected
        before sending the joint target to the actuators.
        """
        if not hasattr(self, "_pose_check_data"):
            self._pose_check_data = mujoco.MjData(self.model)
        scratch = self._pose_check_data
        initial = self.data.qpos[self.arm_q].copy()
        original_margin = self.model.geom_margin.copy()
        # Reserve clearance for the accepted 1 mm IK error and 0.01 rad
        # rotation error at a roughly 0.1 m tool. A collision-free exact IK
        # pose alone cannot protect against this real actuator tracking error.
        clearance = 0.002
        def closest(contacts):
            result = {}
            for g, h, distance in contacts:
                key = tuple(sorted((g, h)))
                result[key] = min(result.get(key, float("inf")), distance)
            return result
        try:
            self.model.geom_margin[:] = np.maximum(original_margin, clearance)
            scratch.qpos[:] = self.data.qpos
            mujoco.mj_forward(self.model, scratch)
            previous = closest(self.unexpected_contacts(scratch, minimum_distance=clearance))
            for fraction in np.linspace(0.0, 1.0, 9)[1:]:
                scratch.qpos[:] = self.data.qpos
                scratch.qpos[self.arm_q] = initial + fraction * (q - initial)
                for name in ("finger_joint1", "finger_joint2"):
                    adr = self.model.joint(name).qposadr[0]
                    scratch.qpos[adr] += fraction * (0.04 * grip - scratch.qpos[adr])
                mujoco.mj_forward(self.model, scratch)
                contacts = self.unexpected_contacts(scratch, minimum_distance=clearance)
                # Existing clearance violations must not trap the robot: a
                # path that monotonically increases separation is an escape.
                # New pairs or worsening clearance remain blocked.
                unsafe = [pair for pair in contacts
                          if tuple(sorted(pair[:2])) not in previous
                          or pair[2] < previous[tuple(sorted(pair[:2]))] - 1e-7
                          or (previous[tuple(sorted(pair[:2]))] <= 0.0
                              and pair[2] <= previous[tuple(sorted(pair[:2]))] + 1e-7)]
                if unsafe:
                    return unsafe
                previous = closest(contacts)
        finally:
            # Only the scratch forward uses this margin. Actual scene/contact
            # physics and Panda assets retain their configured values.
            self.model.geom_margin[:] = original_margin
        return []

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
            axial = float(error @ HANDLE_ROTATION[:, 2])
            lateral = float(error @ HANDLE_ROTATION[:, 0])
            row["handle_normal_tracking_error_m"] = float(np.linalg.norm([axial, error[2]]))
            row["handle_axial_tracking_error_m"] = abs(axial)
            row["handle_vertical_tracking_error_m"] = float(abs(error[2]))
            row["handle_lateral_tracking_error_m"] = abs(lateral)
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
        start_position = self.data.xpos[self.hand].copy()
        start_rotation = self.data.xmat[self.hand].reshape(3, 3).copy()
        requested_grip = float(grip)
        self.gripper = grip
        self.executed_pose_fraction = 1.0
        q = self.ik(np.asarray(xyz),
                    seed=self.data.qpos[self.arm_q].copy() if self.task == 4 else None,
                    allow_fallback=self.task != 4)
        joint_increment = 0.08 if self.task == 4 else 0.04
        if self.task == 4:
            current_q = self.data.qpos[self.arm_q].copy()
            current_xyz = self.data.xpos[self.hand].copy()
            current_rotation = self.data.xmat[self.hand].reshape(3, 3).copy()
            goal_rotation = self.rotation.copy()
            angular_delta = rotation_vector(goal_rotation @ current_rotation.T)
            fraction = 1.0
            original_status = self.ik_status.copy()
            joint_limit_exceeded = bool(np.max(np.abs(q - current_q)) > joint_increment + 1e-9)
            # Clipping each joint independently can produce an unrelated tool
            # rotation even though the full IK solution is accurate. Reduce the
            # Cartesian increment and re-solve on the current physical branch.
            while (not self.ik_status["converged"] or
                   np.max(np.abs(q - current_q)) > joint_increment + 1e-9) and fraction > 1 / 128:
                fraction *= 0.5
                self.rotation = rotation_matrix(fraction * angular_delta) @ current_rotation
                q = self.drawer_ik(current_xyz + fraction * (xyz - current_xyz), current_q)
            self.rotation = goal_rotation
            if not self.ik_status["converged"] or np.max(np.abs(q-current_q)) > joint_increment + 1e-9:
                q = current_q.copy()
                fraction = 0.0
            self.executed_pose_fraction = fraction
            self.ik_status.update(
                requested_position_residual_m=original_status["position_residual_m"],
                requested_rotation_residual_rad=original_status["rotation_residual_rad"],
                requested_joint_limit_exceeded=joint_limit_exceeded,
                applied_pose_fraction=fraction,
                execution=("execute" if fraction == 1 else "execute_reduced_pose" if fraction
                           else "hold_unreachable_pose" if not self.ik_status["converged"]
                           else "hold_joint_limit"),
            )
        elif self.task != 4:
            q = self.arm_target + np.clip(q-self.arm_target,-joint_increment,joint_increment)
        if self.task == 4:
            # Aperture changes can collide even when all arm motion was rejected.
            unsafe = self.unsafe_pose_contacts(q, grip)
            limiting_contacts = list(unsafe)
            if unsafe:
                # Try smaller increments of the same requested pose/aperture.
                # Collision checks need the same backtracking as IK checks;
                # rejecting a full move does not prove every fraction is unsafe.
                status = self.ik_status.copy()
                pose_fraction = self.executed_pose_fraction
                actual_grip = float(np.clip(np.mean([
                    self.data.qpos[self.model.joint(name).qposadr[0]] / 0.04
                    for name in ("finger_joint1", "finger_joint2")
                ]), 0.0, 1.0))
                for factor in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
                    reduced = pose_fraction * factor
                    candidate_grip = actual_grip + factor * (grip - actual_grip)
                    if reduced:
                        self.rotation = rotation_matrix(reduced * angular_delta) @ current_rotation
                        candidate = self.drawer_ik(current_xyz + reduced * (xyz-current_xyz), current_q)
                        if (not self.ik_status["converged"]
                                or np.max(np.abs(candidate-current_q)) > joint_increment + 1e-9):
                            continue
                    else:
                        candidate = current_q.copy()
                    if not self.unsafe_pose_contacts(candidate, candidate_grip):
                        q, grip = candidate, candidate_grip
                        self.executed_pose_fraction = reduced
                        if reduced:
                            status.update(self.ik_status)
                        status.update(execution="execute_collision_reduced_pose" if reduced else status["execution"],
                                      applied_pose_fraction=reduced)
                        unsafe = []
                        break
                self.rotation = goal_rotation
                self.ik_status = status
            if unsafe:
                # Hold the physical pose and aperture; opening/closing can
                # itself introduce a scene collision. The next policy request
                # sees the actual held state and can propose another action.
                q = self.data.qpos[self.arm_q].copy()
                grip = self.gripper = float(np.clip(np.mean([
                    self.data.qpos[self.model.joint(name).qposadr[0]] / 0.04
                    for name in ("finger_joint1", "finger_joint2")
                ]), 0.0, 1.0))
                self.executed_pose_fraction = 0.0
                self.ik_status.update(execution="hold_unsafe_pose", applied_pose_fraction=0.0,
                                      unsafe_pose_contacts=unsafe)
            self._consecutive_ik_rejections = (
                self._consecutive_ik_rejections + 1 if not self.executed_pose_fraction else 0
            )
            self.ik_status["consecutive_rejected_steps"] = self._consecutive_ik_rejections
            self.ik_status["limiting_pose_contacts"] = limiting_contacts
            self.ik_status["limiting_pose_contact_names"] = [dict(
                geom_ids=[int(g), int(h)],
                geoms=[self.model.geom(g).name, self.model.geom(h).name],
                bodies=[self.model.body(self.model.geom_bodyid[g]).name,
                        self.model.body(self.model.geom_bodyid[h]).name],
                distance_m=float(distance),
            ) for g, h, distance in limiting_contacts if isinstance(g, (int, np.integer))
              and isinstance(h, (int, np.integer))]
        # The target's FK and the actual actuator motion are different evidence.
        # Capture both without consulting object positions or task answers.
        if not hasattr(self, "_tracking_data"):
            self._tracking_data = mujoco.MjData(self.model)
        self._tracking_data.qpos[:] = self.data.qpos
        self._tracking_data.qpos[self.arm_q] = q
        mujoco.mj_forward(self.model, self._tracking_data)
        allowed_position = self._tracking_data.xpos[self.hand].copy()
        allowed_rotation = self._tracking_data.xmat[self.hand].reshape(3, 3).copy()
        self.integrate(q, grip)
        actual = self.data.xmat[self.hand].reshape(3, 3)
        progress = pose_progress(
            np.asarray(xyz) - start_position,
            self.data.xpos[self.hand] - start_position,
            rotation_vector(self.rotation @ start_rotation.T),
            rotation_vector(actual @ start_rotation.T),
            self.executed_pose_fraction,
        )
        self.rows[-1]["pose_tracking"] = dict(
                requested_position=np.asarray(xyz).tolist(),
                requested_rotation=self.rotation.ravel().tolist(),
                allowed_position=allowed_position.tolist(),
                allowed_rotation=allowed_rotation.ravel().tolist(),
                requested_gripper_target=requested_grip,
                allowed_gripper_target=float(grip),
                actual_finger_positions_m=[float(self.data.qpos[self.model.joint(name).qposadr[0]])
                                           for name in ("finger_joint1", "finger_joint2")],
                allowed_translation_m=(allowed_position - start_position).tolist(),
                actual_translation_m=(self.data.xpos[self.hand] - start_position).tolist(),
                allowed_position_tracking_error_m=float(np.linalg.norm(self.data.xpos[self.hand] - allowed_position)),
                position_error_m=float(np.linalg.norm(self.data.xpos[self.hand] - xyz)),
                rotation_error_rad=float(np.linalg.norm(rotation_vector(self.rotation @ actual.T))),
                joint_limited=(self.ik_status or {}).get("requested_joint_limit_exceeded", False),
                pose_reduced=self.executed_pose_fraction < 1.0,
                **progress,
            )
        for name, flag in (("severely_reduced", progress["severely_reduced"]),
                           ("no_progress", progress["no_progress"])):
            attribute = "_consecutive_" + name + "_steps"
            count = getattr(self, attribute, 0) + 1 if flag else 0
            setattr(self, attribute, count)
            self.rows[-1]["pose_tracking"]["consecutive_" + name + "_steps"] = count

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
        longest_handle_gap = 0.0
        unpermitted_table = False
        max_table_force = 0.0
        first_unpermitted_table = None
        for _ in range(self.CONTROL_NSTEP):
            if hasattr(self, "_presentation_substep"):
                self._presentation_substep()
            mujoco.mj_step(self.model, self.data)
            object_contact, n, d = self.contacts()
            self._object_contact_seconds = (
                self._object_contact_seconds + self.model.opt.timestep if object_contact else 0.0
            )
            peak = max(peak, n)
            finger_table, palm_table = self.table_contact_counts()
            peak_finger_table = max(peak_finger_table, finger_table)
            peak_palm_table = max(peak_palm_table, palm_table)
            depth = min(depth, d)
            allowed_table, table_force = self.controlled_fingertip_touch()
            max_table_force = max(max_table_force, table_force)
            unpermitted_table |= bool(n and not allowed_table)
            if n and not allowed_table and first_unpermitted_table is None:
                first_unpermitted_table = dict(simulation_time=float(self.data.time),
                                              **self._table_touch_evidence)
            if self.task == 4:
                contact = self.contacts(self.model.geom(f"handle{self.target}").id)[0]
                handle_contact_steps += int(contact)
                self._handle_contact_seconds = (
                    self._handle_contact_seconds + self.model.opt.timestep if contact else 0.0
                )
                self._handle_gap_seconds = (
                    0.0 if contact else self._handle_gap_seconds + self.model.opt.timestep
                )
                longest_handle_gap = max(longest_handle_gap, self._handle_gap_seconds)
            for g, h, dist in self.unexpected_contacts():
                if allowed_table and self.is_finger_table_contact(g, h):
                    continue
                key = (g, h)
                if key not in unexpected or dist < unexpected[key][2]:
                    unexpected[key] = [g, h, dist]
            if hasattr(self, "failure_events") and self.presentation_complete:
                if (n and not allowed_table) or unexpected:
                    self.unsafe_execution_contact = True
                    self._failure("unsafe_execution_contact")
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
            if hasattr(self, "failure_events") and self.presentation_complete:
                self._update_object_retention()
                if self.task == 4:
                    self._update_drawer_selection()
                self._update_placement()
        mujoco.mj_forward(self.model, self.data)
        # mj_step integrates qpos after computing derived contacts/positions.
        # Match the existing final forward before exposing cached placement;
        # the same timestamp cannot add any stable duration.
        if hasattr(self, "failure_events") and self.presentation_complete:
            self._update_placement()
        self.record(
            peak,
            depth,
            list(unexpected.values()),
            inclination,
            peak_finger_table,
            peak_palm_table,
        )
        self.rows[-1]["unpermitted_table_contact"] = unpermitted_table
        self.rows[-1]["peak_table_contact_force_n"] = max_table_force
        self.rows[-1]["first_unpermitted_table_evidence"] = first_unpermitted_table
        self.rows[-1]["ik_status"] = self.ik_status
        if self.task == 4:
            self.rows[-1]["handle_two_contact_physics_steps"] = handle_contact_steps
            self.rows[-1]["longest_handle_contact_gap_seconds"] = (
                longest_handle_gap
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
                delta * min(1.0, 0.012 / max(np.max(np.abs(delta)), 1e-12))
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

    def settle(self):
        """Expert-only wait for actual release/support/dwell, bounded in seconds."""
        period = self.model.opt.timestep * self.CONTROL_NSTEP
        for _ in range(int(np.ceil(5.0 / period))):
            self.hold(1, 1.0, "settle")
            if self.success():
                return True
        self.motion_failures.append({"phase": "settle", "reason": "placement did not settle within 5 seconds"})
        return False

    def close_gripper(self, phase, geom=None, xyz=None):
        """Expert-only closure: continue when real bilateral contact is stable.

        A fixed five-step pause teaches near-identical closed-hand observations
        both to wait and to start pulling/lifting. Use contact evidence rather
        than an unobserved expert timer; policy execution never calls this.
        """
        self.phase = phase
        position = self.data.xpos[self.hand].copy() if xyz is None else np.asarray(xyz)
        for _ in range(5):
            self._step_pose(position, 0.0)
            duration = self._object_contact_seconds if geom is None else self._handle_contact_seconds
            if self.contacts(geom)[0] and duration >= 0.04 - 1e-9:
                return True
        return False

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
        self.motion_failures.append(
            dict(phase=phase, reason="orientation target not reached",
                 position_error_m=float(np.linalg.norm(pos - self.data.xpos[self.hand])),
                 rotation_error_rad=float(np.linalg.norm(rotation_vector(
                     self.rotation @ self.data.xmat[self.hand].reshape(3, 3).T))),
                 ik_status=self.ik_status)
        )
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


def pose_progress(requested_translation, actual_translation, requested_rotation,
                  actual_rotation, fraction):
    """Diagnostic progress along the requested motion; never a policy correction.

    Sub-0.5 mm translations and sub-0.005 rad rotations are treated as deliberate
    holds. Meaningful motion with less than 5% directional progress (or below
    0.1 mm / 0.001 rad resolution) is reported separately from an IK rejection.
    """
    distance = float(np.linalg.norm(requested_translation))
    angle = float(np.linalg.norm(requested_rotation))
    linear = float(np.dot(actual_translation, requested_translation) / distance) if distance else 0.0
    angular = float(np.dot(actual_rotation, requested_rotation) / angle) if angle else 0.0
    translation_stalled = distance >= 0.0005 and linear < max(0.0001, 0.05 * distance)
    rotation_stalled = angle >= 0.005 and angular < max(0.001, 0.05 * angle)
    return dict(requested_translation_m=np.asarray(requested_translation).tolist(),
                requested_translation_length_m=distance, requested_rotation_length_rad=angle,
                translation_progress_m=linear, rotation_progress_rad=angular,
                translation_progress_ratio=linear / distance if distance else None,
                rotation_progress_ratio=angular / angle if angle else None,
                completely_rejected=bool(fraction == 0.0),
                severely_reduced=bool(0.0 < fraction < 0.125 and (distance >= 0.0005 or angle >= 0.005)),
                no_progress=bool(translation_stalled or rotation_stalled))


class PandaTaskEnv(NativePanda):
    CONTROL_NSTEP = 100
    MAX_DPOS = 0.012
    MAX_DROT = 0.10

    def __init__(
        self, task, seed=None, capture_frames=True, scene_path=None, image_size=448,
        target=None,
    ):
        # More frequent Task4 feedback leaves genuine simulation-time margin
        # for target history. Task3 and Task1 keep their existing control rate.
        self.CONTROL_NSTEP = 50 if task == 4 else 100
        self.seed = seed
        rng = np.random.default_rng(seed)
        height = float(rng.uniform(0.03, 0.07)) if task == 3 else 0.03
        sampled_target = int(rng.integers(4)) if task == 4 else 0
        if target is not None and (task != 4 or target not in range(4)):
            raise ValueError("An explicit presentation target is only valid for Task4 drawers 0..3")
        target = sampled_target if target is None else int(target)
        # Independent nuisance stream: the same starting pose can accompany
        # any target history. Preserve this offset after human closure, so the
        # handle approach has varied, correctly supervised starting states.
        start_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
        start_offset = start_rng.uniform(-DRAWER_START_JITTER, DRAWER_START_JITTER) if task == 4 else np.zeros(3)
        approach_offset = start_rng.uniform(-DRAWER_APPROACH_JITTER, DRAWER_APPROACH_JITTER) if task == 4 else np.zeros(3)
        if task == 3:
            xy = rng.uniform([0.345, -0.115], [0.375, -0.085])
            yaw = float(rng.uniform(-np.pi / 36, np.pi / 36))
        else:
            local = np.r_[np.array([-0.045, 0.0]) + rng.uniform([-0.006, -0.010], [0.006, 0.010]), 0.0]
            xy = (CABINET_POSITION + CABINET_ROTATION @ local)[:2]
            yaw = CABINET_YAW
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
        self._object_release_committed = False
        self._object_retention_armed = False
        self._object_free_hold_since = None
        self._lift_start = self.data.xpos[self.obj, 2]
        self._success_dwell = 0
        self._stable_since = None
        self._contact_gap_since = None
        self._contact_gap_start_z = None
        self._contact_gap_descent_m = 0.0
        self._contact_gap_release_allowed = False
        self._contact_gap_closed_hold = False
        self._contact_gap_hold_relative = None
        self._contact_gap_last_loaded_time = None
        self._retention_gripper_open = self.gripper > 0.5
        self.contact_instability_seen = False
        self.failure_events = []
        self._placement_status = {}
        if task == 3:
            self._storage_xy_bounds = self._storage_inner_xy_bounds()
        self._task_milestones = {}
        self.approached_drawer = None
        self.contacted_drawer = None
        self.opened_drawer = None
        self._drawer_contact_seconds = np.zeros(4)
        self.wrong_drawer_opened = False
        self.unsafe_execution_contact = False
        self.finger_table_contact_seen = False
        self._human_close_start = None
        if task == 4:
            self.expert_handle_approach_offset = approach_offset
            # Common robot pose for all four histories, clear of the presented drawer.
            self.observation_hand_position = DRAWER_OBSERVATION_HAND_POSITION + start_offset
            self.execution_hand_position = DRAWER_EXECUTION_HAND_POSITION + start_offset
            self.arm_target = self.drawer_ik(self.observation_hand_position)
            self.data.qpos[self.arm_q] = self.arm_target
            self.data.qvel[:] = 0.0
            self.set_controls(self.arm_target, 1.0)
            joint = self.model.joint(f"drawer_slide{target}")
            self.data.qpos[joint.qposadr[0]] = 0.12
            self.data.qpos[self.object_q:self.object_q + 3] -= 0.12 * HANDLE_ROTATION[:, 2]
            mujoco.mj_forward(self.model, self.data)
            self.rows.clear()
            self.frames.clear()

    def _presentation_substep(self):
        """Declared simulated human closure, disabled before policy execution."""
        if self._human_close_start is None:
            return
        elapsed = self.data.time - self._human_close_start
        # A realistic, slower human closure uses the existing 1.4 s presentation
        # interval; it retains the final visible observation longer without
        # extending K or stretching the simulation/history timestamps.
        duration = 1.2
        progress = np.clip(elapsed / duration, 0.0, 1.0)
        desired = 0.06 * (1.0 + np.cos(np.pi * progress))
        velocity = (
            -0.06 * np.pi / duration * np.sin(np.pi * progress) if progress < 1 else 0.0
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
        control_period = self.model.opt.timestep * self.CONTROL_NSTEP
        for _ in range(round(3.0 / control_period)):
            self._step_pose(self.data.xpos[self.hand].copy(), 1.0)
            self._record_target_visibility()
            yield self.obs()
        self._human_close_start = float(self.data.time)
        self.phase = "presentation_close"
        close_steps = round(1.4 / control_period)
        for index in range(close_steps):
            self.rotation = HANDLE_ROTATION.copy()
            self._step_pose(self.execution_hand_position, 1.0)
            if index == close_steps - 1:
                # The simulated human finishes at the mechanical closed stop.
                # Remove only settled micrometre-scale solver residue, equally
                # for all drawers, before the final closed observation. Without
                # this, the last presented drawer leaks through rasterized edges.
                joints = [self.model.joint(f"drawer_slide{k}") for k in range(4)]
                if any(abs(self.data.qpos[j.qposadr[0]]) > 0.00005
                       or abs(self.data.qvel[j.dofadr[0]]) > 0.001 for j in joints):
                    raise RuntimeError("Simulated human closure has not settled at the closed stop")
                for joint in joints:
                    self.data.qpos[joint.qposadr[0]] = 0.0
                    self.data.qvel[joint.dofadr[0]] = 0.0
                self._human_close_start = None
                self.data.qfrc_applied[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
            self._record_target_visibility()
            yield self.obs()
        self._human_close_start = None
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if (np.linalg.norm(self.data.xpos[self.hand] - self.execution_hand_position) > 0.002
                or np.linalg.norm(rotation_vector(
                    self.data.xmat[self.hand].reshape(3, 3) @ HANDLE_ROTATION.T)) > 0.01):
            raise RuntimeError("Robot has not reached the common drawer execution start pose")
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
        if (hasattr(self, "_collection_max_steps")
                and self.phase not in {"observe_target", "presentation_close"}):
            if self._collection_execution_steps >= self._collection_max_steps:
                raise CollectionStepLimit("robot execution step budget exhausted")
            self._collection_execution_steps += 1
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
        if self.task == 4:
            self.samples[-1]["action"][:6] *= self.executed_pose_fraction
            self.samples[-1]["action"][6] = self.gripper
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
        # Dataset actions are float32 for both contact tasks. Quantize the
        # expert request before execution, including Task4's rotations: doing
        # this only in the writer can change later IK/contact decisions on
        # replay. Policy step(), limits and collision protection are unchanged.
        action = action.astype(np.float32)
        result = self.step(action)
        self.rotation = goal_rotation
        return result

    def _update_events(self):
        two = self.contacts()[0]
        self.had_object_contact |= two
        self.was_lifted |= two and self.data.xpos[self.obj, 2] > self._lift_start + 0.05
        for milestone, reached in (
            ("object_contact", self.had_object_contact),
            ("object_lifted", self.was_lifted),
            ("drawer_selected", self.first_selected_drawer is not None),
            ("drawer_opened", self.task == 4 and any(
                self.data.qpos[self.model.joint(f"drawer_slide{k}").qposadr[0]] >= 0.165
                for k in range(4))),
            ("supported_placement", self._placement_status.get("contained", False)
             and self._placement_status.get("supported", False)
             and self._placement_status.get("released", False)),
        ):
            if reached:
                self._task_milestones.setdefault(milestone, float(self.data.time))
        if self.task == 3 and self.grasp_attempts == 1:
            self.first_grasp_lifted |= (
                two and self.data.xpos[self.obj, 2] > self._lift_start + 0.05
            )
        if self.presentation_complete:
            row = self.rows[-1]
            self.finger_table_contact_seen |= bool(row["finger_table_contacts"])
            blocking_contacts = row["unexpected_robot_contacts_peak"]
            table_contact = row["unpermitted_table_contact"]
            self.unsafe_execution_contact |= bool(table_contact or blocking_contacts)
            for reason, triggered in (
                ("transport_drop", self.transport_dropped),
                ("wrong_drawer_opened", self.wrong_drawer_opened),
                ("unsafe_execution_contact", self.unsafe_execution_contact),
            ):
                if triggered:
                    self._failure(reason)

    def _failure(self, reason):
        if not any(event["reason"] == reason for event in self.failure_events):
            self.failure_events.append(dict(reason=reason, simulation_time=float(self.data.time)))

    def _update_drawer_selection(self):
        for k in range(4):
            handle = self.model.geom(f"handle{k}").id
            selected = self.contacts(handle)[0]
            self._drawer_contact_seconds[k] = (
                self._drawer_contact_seconds[k] + self.model.opt.timestep if selected else 0.0
            )
            confirmed = self._drawer_contact_seconds[k] >= 0.02 - 1e-9
            opened = self.data.qpos[self.model.joint(f"drawer_slide{k}").qposadr[0]] > 0.015
            grasp = self.data.geom_xpos[handle] - 0.1029 * HANDLE_ROTATION[:, 2]
            if np.linalg.norm(self.data.xpos[self.hand] - grasp) < 0.020:
                self.approached_drawer = k
            if confirmed:
                self.contacted_drawer = k
            if opened:
                self.opened_drawer = k
                if k != self.target:
                    self.wrong_drawer_opened = True
                    self._failure("wrong_drawer_opened")
            if self.first_selected_time is None and (confirmed or opened):
                self.first_selected_time = float(self.data.time)
                self.first_selected_drawer = k
                if k != self.target:
                    self._failure("wrong_first_drawer")

    def _storage_inner_xy_bounds(self):
        """The actual inner wall faces, including the bin's world rotation."""
        bottom = self.model.geom("bin_bottom").id
        center = self.data.geom_xpos[bottom, :2]
        half = (np.abs(self.data.geom_xmat[bottom].reshape(3, 3))
                @ self.model.geom_size[bottom])[:2]
        lower, upper = center - half, center + half
        body = self.model.geom_bodyid[bottom]
        for geom in np.flatnonzero(self.model.geom_bodyid == body):
            if geom == bottom:
                continue
            position = self.data.geom_xpos[geom, :2]
            extent = (np.abs(self.data.geom_xmat[geom].reshape(3, 3))
                      @ self.model.geom_size[geom])[:2]
            axis = int(np.argmin(extent))
            if extent[axis] < 0.02 and abs(position[axis] - center[axis]) > half[axis] / 2:
                if position[axis] > center[axis]:
                    upper[axis] = min(upper[axis], position[axis] - extent[axis])
                else:
                    lower[axis] = max(lower[axis], position[axis] + extent[axis])
        return lower, upper

    def _target_xy_contains(self, low, high):
        if self.task == 3:
            lower, upper = self._storage_xy_bounds
        else:
            surface = self.model.geom("receiving_area").id
            center = self.data.geom_xpos[surface, :2]
            half = self.model.geom_size[surface, :2]
            lower, upper = center - half, center + half
        return bool(np.all(low[:2] >= np.asarray(lower) - PLACEMENT_BOUNDARY_EPS_M)
                    and np.all(high[:2] <= np.asarray(upper) + PLACEMENT_BOUNDARY_EPS_M))

    def _update_object_retention(self):
        """Track contact interruption separately from sustained physical loss."""
        gripper_open = self.gripper > 0.5
        opening = gripper_open and not getattr(self, "_retention_gripper_open", gripper_open)
        self._retention_gripper_open = gripper_open
        two = self.contacts()[0]
        z = float(self.data.xpos[self.obj, 2])
        self.had_object_contact |= two
        self.was_lifted |= bool(two and z > self._lift_start + 0.05)
        # Partial lifts also need retention monitoring. This evidence is separate
        # from the unchanged 50 mm task-success milestone above. Contact forces
        # act on geom2; contact.frame rows are the contact axes in world space.
        weight = float(self.model.body_mass[self.obj] * np.linalg.norm(self.model.opt.gravity))
        minimum_load = max(1e-6, weight * 0.01)
        finger_loads = {int(body): 0.0 for body in self.fingers}
        external_upward_load = 0.0
        forces_valid = True
        for index, ct in enumerate(self.data.contact):
            if self.object_geom not in (ct.geom1, ct.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if not np.isfinite(force[:3]).all():
                forces_valid = False
                continue
            other = ct.geom2 if ct.geom1 == self.object_geom else ct.geom1
            body = int(self.model.geom_bodyid[other])
            if body in finger_loads:
                finger_loads[body] += float(np.linalg.norm(force[:3]))
            else:
                world_force = np.asarray(ct.frame).reshape(3, 3).T @ force[:3]
                object_force = world_force if ct.geom2 == self.object_geom else -world_force
                external_upward_load += max(0.0, float(object_force[2]))
        free_hold = bool(
            not gripper_open and forces_valid
            and all(load >= minimum_load for load in finger_loads.values())
            and z > self._lift_start + 0.001
            and external_upward_load < weight * 0.05
        )
        if free_hold:
            if self._object_free_hold_since is None:
                self._object_free_hold_since = float(self.data.time)
        else:
            self._object_free_hold_since = None
        sustained_free_hold = bool(
            self._object_free_hold_since is not None
            and self.data.time - self._object_free_hold_since >= 0.04 - 1e-9
        )
        # Preserve immediate monitoring at the old lift threshold before a legal
        # release. After release, only fresh sustained free-hold evidence re-arms;
        # a supported touch or closing an empty hand must not start another carry.
        legacy_lift = bool(two and z > self._lift_start + 0.05
                           and not self._object_release_committed)
        if sustained_free_hold or legacy_lift:
            self._object_retention_armed = True
            self._object_release_committed = False
        if not self._object_retention_armed or two or self._object_release_committed:
            self._contact_gap_since = None
            self._contact_gap_descent_m = 0.0
            self._contact_gap_release_allowed = False
            self._contact_gap_closed_hold = False
            self._contact_gap_hold_relative = None
            self._contact_gap_last_loaded_time = None
            return
        low, high = map(np.asarray, self.object_bounds())
        rotation = self.data.xmat[self.hand].reshape(3, 3)
        tool = self.data.xpos[self.hand] + 0.1029 * rotation[:, 2]
        relative = rotation.T @ (self.data.xpos[self.obj] - tool)
        finger_contact = any(
            ct.dist <= 0.0001 and self.object_geom in (ct.geom1, ct.geom2)
            and self.model.geom_bodyid[ct.geom2 if ct.geom1 == self.object_geom else ct.geom1] in self.fingers
            for ct in self.data.contact
        )
        loaded_contact = False
        # A later accidental touch cannot establish a new carrying interval.
        # Only inspect loads for a hold established at this gap's closed-hand
        # start, or while that original hold remains physically intact.
        if ((self._contact_gap_since is None and not gripper_open)
                or getattr(self, "_contact_gap_closed_hold", False)):
            for index, ct in enumerate(self.data.contact):
                if ct.dist > 0.0001 or self.object_geom not in (ct.geom1, ct.geom2):
                    continue
                other = ct.geom2 if ct.geom1 == self.object_geom else ct.geom1
                if self.model.geom_bodyid[other] not in self.fingers:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                loaded_contact |= bool(np.linalg.norm(force[:3]) > 1e-9)
        release_window = (self.gripper > 0.5 and self._target_xy_contains(low, high)
                          and self.height <= low[2] <= self.height + 0.10)
        if self._contact_gap_since is None:
            self._contact_gap_since = float(self.data.time)
            self._contact_gap_start_z = z
            self._contact_gap_descent_m = 0.0
            # Decide at the first loss of bilateral contact, including a
            # normal one-finger transition while opening. Falling into the
            # release region later cannot authorize an earlier illegal loss.
            self._contact_gap_release_allowed = bool(release_window)
            self._contact_gap_closed_hold = bool(not gripper_open and loaded_contact)
            self._contact_gap_hold_relative = relative.copy()
            self._contact_gap_last_loaded_time = float(self.data.time) if loaded_contact else None
        if getattr(self, "_contact_gap_closed_hold", False):
            if loaded_contact:
                self._contact_gap_last_loaded_time = float(self.data.time)
            # Retain at most one unloaded solver sample, with no material
            # object/tool slip beyond the existing 1 mm pose tracking budget.
            # Losing this evidence is permanent until a real bilateral regrasp.
            self._contact_gap_closed_hold &= bool(
                not self.transport_dropped
                and np.linalg.norm(relative - self._contact_gap_hold_relative) <= 0.001
                and self.data.time - self._contact_gap_last_loaded_time
                    <= self.model.opt.timestep + 1e-9
            )
            if opening and self._contact_gap_closed_hold and loaded_contact and release_window:
                # Loss of bilateral contact is an instability marker. A real,
                # still-loaded closed-hand hold can nevertheless be deliberately
                # released here; it is not a previously free-falling object.
                self._contact_gap_release_allowed = True
            if gripper_open:
                self._contact_gap_closed_hold = False
        # An unauthorized release cannot gain eligibility by falling or moving
        # into the target. The sole transition above requires the original hold.
        self._contact_gap_release_allowed &= bool(release_window)
        if self._contact_gap_release_allowed and not finger_contact:
            self._object_release_committed = True
            self._object_retention_armed = False
            self._object_free_hold_since = None
            self._contact_gap_since = None
            self._contact_gap_descent_m = 0.0
            self._contact_gap_release_allowed = False
            return
        self.contact_instability_seen = True
        self._contact_gap_descent_m = max(
            self._contact_gap_descent_m, self._contact_gap_start_z - z
        )
        separated = np.linalg.norm(tool - self.data.xpos[self.obj]) > 0.035
        # Following a lowering hand is not a physical fall. This exemption ends
        # as soon as the original loaded hold loses contact or slips relative to it.
        fallen = self._contact_gap_descent_m > 0.015 and not self._contact_gap_closed_hold
        if (not self._contact_gap_release_allowed
                and self.data.time - self._contact_gap_since >= 0.10
                and (separated or fallen)):
            self.transport_dropped = True
            self._failure("transport_drop")

    def outcome(self):
        success = self.success()
        reasons = [event["reason"] for event in self.failure_events]
        quality_checks = dict(
            two_finger_grasp=bool(self.had_object_contact),
            object_lifted=bool(self.was_lifted),
            transport_retained=not self.transport_dropped,
            no_unsafe_contact=not self.unsafe_execution_contact,
            correct_first_drawer=(self.task != 4 or self.first_selected_drawer == self.target),
            no_wrong_drawer_opened=not self.wrong_drawer_opened,
        )
        return dict(
            physical_complete=bool(self._placement_status.get("settled", False)),
            strict_quality_passed=not reasons and all(quality_checks.values()),
            quality_checks=quality_checks,
            success=success,
            terminal_requirements=dict(gripper_open=bool(self.gripper > 0.5)),
            termination_status=("success" if success else "physical_complete_quality_failed"
                                if self._placement_status.get("settled", False) and not (not reasons and all(quality_checks.values()))
                                else "physical_complete_terminal_pending"
                                if self._placement_status.get("settled", False)
                                else "quality_failure" if reasons else "running"),
            failure_events=self.failure_events.copy(),
            placement=self._placement_status.copy(),
            approached_drawer=self.approached_drawer,
            contacted_drawer=self.contacted_drawer,
            opened_drawer=self.opened_drawer,
            first_selected_drawer=self.first_selected_drawer,
            drawer_selection_status=(None if self.task != 4 else "not_selected"
                                     if self.first_selected_drawer is None else "correct"
                                     if self.first_selected_drawer == self.target else "wrong"),
            contact_instability_seen=self.contact_instability_seen,
            object_retention_state=("released" if self._object_release_committed else
                                    "carrying" if self.was_lifted else
                                    "contacted" if self.had_object_contact else "ungrasped"),
            task_progress=dict(
                milestones=self._task_milestones.copy(),
                seconds_since_new_milestone=float(self.data.time - max(
                    [self.execution_start, *self._task_milestones.values()])),
                grasp_attempts_without_lift=self.grasp_attempts if not self.was_lifted else 0,
            ),
            ik_status=self.ik_status,
            ik_rejected_steps=sum(row.get("ik_status", {}).get("execution") in {"hold_unreachable_pose", "hold_joint_limit", "hold_unsafe_pose"}
                                  for row in self.rows if row.get("ik_status")),
            severely_reduced_steps=sum(row.get("pose_tracking", {}).get("severely_reduced", False) for row in self.rows),
            no_progress_steps=sum(row.get("pose_tracking", {}).get("no_progress", False) for row in self.rows),
            motion_no_progress_steps=sum(row.get("pose_tracking", {}).get("no_progress", False) for row in self.rows),
        )

    def _update_placement(self):
        """Called once per physical substep; queries never advance the dwell."""
        low, high = map(np.asarray, self.object_bounds())
        if self.task == 3:
            placed = (
                self._target_xy_contains(low, high)
                and self.height + 0.003 < low[2] < self.height + 0.012
            )
        else:
            placed = (
                self._target_xy_contains(low, high)
                and self.height < low[2] < self.height + 0.010
            )
        velocity = self.data.qvel[self.model.joint("object_free").dofadr[0]:][:6]
        stable = np.linalg.norm(velocity[:3]) < 0.025 and np.linalg.norm(velocity[3:]) < 0.15
        finger_contact = False
        support = False
        surface = self.model.geom("bin_bottom" if self.task == 3 else "receiving_area").id
        for ct in self.data.contact:
            if ct.dist > 0.0001 or self.object_geom not in (ct.geom1, ct.geom2):
                continue
            other = ct.geom2 if ct.geom1 == self.object_geom else ct.geom1
            finger_contact |= self.model.geom_bodyid[other] in self.fingers
            support |= other == surface
        # Detachment is a physical fact. The terminal open-hand requirement
        # is checked separately by success(), not by the support dwell timer.
        released = not finger_contact
        physical = bool(placed and stable and released and support)
        if not physical:
            self._stable_since = None
        elif self._stable_since is None:
            self._stable_since = float(self.data.time)
        dwell = 0.0 if self._stable_since is None else float(self.data.time - self._stable_since)
        self._success_dwell = int((dwell + 1e-8) / 0.2) + 1 if physical else 0
        self._placement_status = dict(
            bounds_low=low.tolist(), bounds_high=high.tolist(), contained=bool(placed),
            released=bool(released), supported=bool(support), stable=bool(stable),
            gripper_open=bool(self.gripper > 0.5),
            linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
            angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
            stable_seconds=dwell, settled=physical and dwell >= 1.0 - 1e-8,
        )

    def success(self):
        valid = bool(
            self._placement_status.get("settled", False)
            and self.was_lifted
            and self.had_object_contact
            and not self.transport_dropped
            and not self.wrong_drawer_opened
            and not self.unsafe_execution_contact
            and self.gripper > 0.5
            and (self.task != 4 or self.first_selected_drawer == self.target)
        )
        return valid

    def close(self):
        self.renderer.close()


def run_task3_expert(env):
    height = env.height
    env.natural_corrections = 0
    for attempt in range(2):
        obj = env.data.xpos[env.obj].copy()
        for pose, phase in [
            (np.r_[obj[:2], height + 0.22], "approach"),
            (np.r_[obj[:2], obj[2] + OBJECT_GRASP_HAND_Z_OFFSETS_M[3]], "descend"),
        ]:
            if not env.move(pose, 1.0, phase, tolerance=0.0015):
                return False
        if env.close_gripper("close"):
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
    return env.settle()


def run_task4_expert(env):
    receiving_xy = env.data.geom_xpos[env.model.geom("receiving_area").id, :2].copy()
    handle = env.model.geom(f"handle{env.target}").id
    grasp = env.data.geom_xpos[handle].copy() - 0.1029 * HANDLE_ROTATION[:, 2]
    # Vary the safe pregrasp waypoint too: a common exact waypoint erases
    # initial-state diversity before the final, precision-critical alignment.
    pregrasp = grasp - 0.02 * HANDLE_ROTATION[:, 2] + CABINET_ROTATION @ env.expert_handle_approach_offset
    for pose, phase in [(pregrasp, "approach"), (grasp, "handle_approach")]:
        if not env.move(pose, 0.25, phase, tolerance=0.0003):
            return False
    env.rotation = HANDLE_ROTATION.copy()
    if not env.close_gripper("handle_close", handle, grasp):
        env.motion_failures.append(
            {"phase": "handle_close", "reason": "missing two-finger contact"}
        )
        return False
    if not env.move(grasp - 0.178 * HANDLE_ROTATION[:, 2], 0.0, "pull"):
        return False
    if env.data.qpos[env.model.joint(f"drawer_slide{env.target}").qposadr[0]] < 0.165:
        return False
    env.hold(round(1.0 / (env.model.opt.timestep * env.CONTROL_NSTEP)), 1.0, "handle_release")
    for pose, phase in [
        (env.data.xpos[env.hand] + CABINET_ROTATION @ np.array([-0.01, -0.08, 0]), "handle_withdraw")
    ]:
        if not env.move(pose, 1.0, phase):
            return False
    if not env.move(
        env.data.xpos[env.hand] + [0, 0, 0.09], 1.0, "handle_clearance_lift"
    ):
        return False
    if not env.orient(
        OBJECT_ROTATION, 1.0, "topdown_orientation"
    ):
        return False
    obj = env.data.xpos[env.obj].copy()
    for pose, phase in [
        # Raise the open fingertips above the real drawer front before crossing
        # it. Turning locally does not by itself provide vertical clearance.
        ([*env.data.xpos[env.hand, :2], obj[2] + 0.19], "object_clearance_lift"),
        (obj + [0, 0, 0.19], "object_approach"),
        (obj + [0, 0, OBJECT_GRASP_HAND_Z_OFFSETS_M[4]], "object_descend"),
    ]:
        if not env.move(pose, 1.0, phase):
            return False
    if not env.close_gripper("object_close"):
        env.motion_failures.append(
            {"phase": "object_close", "reason": "missing two-finger contact"}
        )
        return False
    for pose, phase in [
        (obj + [-0.03, 0, 0.205], "object_lift"),
        ([receiving_xy[0] - 0.02, receiving_xy[1], obj[2] + 0.205], "clear_drawer"),
        ([*receiving_xy, env.height + 0.26], "transport"),
        ([*receiving_xy, env.height + 0.15], "lower"),
    ]:
        if not env.move(pose, 0.0, phase):
            return False
    env.hold(round(1.0 / (env.model.opt.timestep * env.CONTROL_NSTEP)), 1.0, "release")
    if not env.move([*receiving_xy, env.height + 0.26], 1.0, "retreat"):
        return False
    return env.settle()


SCHEMA = CONTACT_QUALITY_SCHEMA
VERSION = CONTACT_QUALITY_VERSION


def trajectory_quality(env, succeeded):
    selected_times = []
    visible_slots = []
    decision_time = None
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
        "no_robot_table_contact": all(not r["unpermitted_table_contact"] for r in rows),
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
        "confirmed_release_and_support": bool(env._placement_status.get("settled")),
        "continuous_contact_monitoring": True,
        "physics_substep_placement_monitoring": True,
        "contact_gap_descent_monitoring": True,
    }
    if env.task == 4:
        handle = [
            r for r in rows if r["phase"] in ("handle_approach", "handle_close", "pull")
        ]
        pull = [r for r in rows if r["phase"] == "pull"]
        sample_times = np.array([r["timestamp"] for r in env.samples])
        if env.first_selected_time is not None:
            # Selection is now timestamped at a physics substep; use the
            # observation preceding that substep, not another full step earlier.
            decision_time = env.first_selected_time - 1e-8
            decision_index = np.searchsorted(sample_times, decision_time, side="right") - 1
            decision_time = float(sample_times[max(decision_index, 0)])
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
                    and r["longest_handle_contact_gap_seconds"] <= 0.04 + 1e-9
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
                "presentation_robot_clear": all(
                    not row["unexpected_robot_contacts_peak"]
                    and not row["table_contacts"]
                    and not row["two_finger_contact"]
                    and not row.get("two_finger_handle_contact", False)
                    for row in env.rows if row["t"] <= env.execution_start + 1e-8
                ),
                "selection_within_history": bool(visible_slots)
                and env.last_visible_time is not None
                and decision_time - env.last_visible_time <= 5.0 + 1e-6,
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
        "policy_state_continuations": getattr(env, "policy_state_continuations", 0),
        "transport_drop": bool(env.transport_dropped),
        "last_target_visible_sim_time": env.last_visible_time,
        "drawer_selected_sim_time": env.first_selected_time,
        "drawer_selection_observation_sim_time": decision_time,
        "sampled_history_sim_times_at_selection": selected_times,
        "target_visible_history_slots_at_selection": visible_slots,
        "target_drawer_diagnostic_only": env.target if env.task == 4 else None,
        "motion_failures": env.motion_failures,
        "timestamps_preserve_control_time": True,
    }
