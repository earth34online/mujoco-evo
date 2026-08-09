import numpy as np
import mujoco

XML = """
<mujoco model="handwritten_panda_pick_place">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <!-- Match LIBERO's 2ms physics integration. With this handwritten
         position-actuated Panda, damping=40 removes substep direction reversals
         while retaining enough authority for the scripted grasp. -->
    <joint damping="40.0" armature="0.02"/>
    <geom friction="0.8 0.1 0.1"/>
  </default>
  <visual>
    <global offwidth="448" offheight="448"/>
    <headlight diffuse="0.62 0.62 0.62" ambient="0.28 0.28 0.28" specular="0.10 0.10 0.10"/>
    <rgba haze="0.70 0.74 0.80 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.38 0.44 0.52" rgb2="0.70 0.75 0.80" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.36 0.38 0.40" rgb2="0.52 0.54 0.56" width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.01"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0.1 -0.55 2.8" dir="0 0 -1" diffuse="0.75 0.75 0.75" ambient="0.15 0.15 0.15"/>
    <light name="fill_light" pos="-0.9 0.8 1.5" dir="0.4 -0.2 -1" diffuse="0.25 0.25 0.25" ambient="0.03 0.03 0.03"/>
    <body name="camera_target" pos="-0.01 0 0.15"/>
    <camera name="front" pos="0.58 -0.82 0.58" mode="targetbody" target="camera_target" fovy="48"/>
    <camera name="overhead" pos="0 0 1.12" mode="targetbody" target="camera_target" fovy="46"/>

    <geom name="floor" type="plane" pos="0 0 -0.005" size="1.0 1.0 0.01" material="floor_mat" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="0 0 0" size="0.45 0.35 0.03" rgba="0.46 0.48 0.51 1"/>

    <body name="panda_base" pos="-0.34 -0.18 0.03">
      <geom name="base_foot" type="cylinder" pos="0 0 0.035" size="0.070 0.035" rgba="0.13 0.15 0.18 1" contype="0" conaffinity="0"/>
      <geom name="base_column" type="cylinder" pos="0 0 0.13" size="0.050 0.095" rgba="0.20 0.23 0.27 1" contype="0" conaffinity="0"/>

      <body name="panda_link1" pos="0 0 0.333">
        <joint name="panda_joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom name="joint1_shell" type="sphere" size="0.052" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 -0.20 0 0 0" size="0.044" rgba="0.82 0.84 0.86 1" contype="0" conaffinity="0"/>

        <body name="panda_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
          <joint name="panda_joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom name="joint2_shell" type="sphere" size="0.047" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 -0.316 0" size="0.041" rgba="0.84 0.86 0.88 1" contype="0" conaffinity="0"/>

          <body name="panda_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
            <joint name="panda_joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom name="joint3_shell" type="sphere" size="0.044" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0.0825 0 0" size="0.038" rgba="0.78 0.81 0.84 1" contype="0" conaffinity="0"/>

            <body name="panda_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
              <joint name="panda_joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom name="joint4_shell" type="sphere" size="0.041" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
              <geom name="link4_geom" type="capsule" fromto="0 0 0 -0.0825 0.384 0" size="0.035" rgba="0.84 0.86 0.88 1" contype="0" conaffinity="0"/>

              <body name="panda_link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0">
                <joint name="panda_joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom name="joint5_shell" type="sphere" size="0.038" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>

                <body name="panda_link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
                  <joint name="panda_joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom name="joint6_shell" type="sphere" size="0.035" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
                  <geom name="link6_geom" type="capsule" fromto="0 0 0 0.088 0 0" size="0.029" rgba="0.84 0.86 0.88 1" contype="0" conaffinity="0"/>

                  <body name="panda_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
                    <joint name="panda_joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <!-- LIBERO keeps one link7 between joint7 and right_hand. -->
                    <geom name="link7_geom" type="capsule" fromto="0 0 0 0 0 0.1065" size="0.030" rgba="0.76 0.79 0.82 1" contype="0" conaffinity="0"/>

                    <!-- LIBERO / robosuite Panda hand mounting. The flange keeps
                         its -45 degree mounting angle and has no cylinder between
                         the fingers. -->
                    <body name="eef" pos="0 0 0.1065" quat="0.92388 0 0 -0.382683">
                      <geom name="gripper_palm" type="box" pos="0 0 0.017" size="0.040 0.043 0.017" rgba="0.30 0.32 0.36 1" contype="0" conaffinity="0"/>
                      <!-- Same hand-relative placement used by robosuite's
                           eye_in_hand camera. -->
                      <camera name="wrist" mode="fixed" pos="0.05 0 0" quat="0 0.707108 0.707108 0" fovy="75"/>

                      <!-- Independent Panda fingers. Only the high-friction red
                           pads collide with the cube. -->
                      <body name="left_finger_body" pos="0 0.018 0.055">
                        <joint name="left_finger_joint" axis="0 1 0" type="slide" range="0 0.022"/>
                        <geom name="left_finger" type="box" pos="0 0 -0.006" size="0.010 0.006 0.012" rgba="0.50 0.52 0.55 1" contype="0" conaffinity="0"/>
                        <geom name="left_finger_tip" type="box" pos="0 0 0" size="0.010 0.006 0.006" rgba="0.92 0.20 0.15 1" contype="2" conaffinity="2" friction="5 0.05 0.0001" condim="4" solref="0.01 0.5"/>
                      </body>
                      <body name="right_finger_body" pos="0 -0.018 0.055">
                        <joint name="right_finger_joint" axis="0 -1 0" type="slide" range="0 0.022"/>
                        <geom name="right_finger" type="box" pos="0 0 -0.006" size="0.010 0.006 0.012" rgba="0.50 0.52 0.55 1" contype="0" conaffinity="0"/>
                        <geom name="right_finger_tip" type="box" pos="0 0 0" size="0.010 0.006 0.006" rgba="0.92 0.20 0.15 1" contype="2" conaffinity="2" friction="5 0.05 0.0001" condim="4" solref="0.01 0.5"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="cube" pos="0.12 -0.08 0.046">
      <joint name="cube_free" type="free"/>
      <!-- 32mm cube: the 36mm closed pad gap gives the fingers room to
           establish contact before they apply clamping force. -->
      <geom name="cube_geom" type="box" size="0.016 0.016 0.016" mass="0.05" rgba="0.12 0.42 0.92 1" contype="2" conaffinity="3" friction="1.2 0.05 0.0001" condim="4"/>
    </body>

    <body name="goal" pos="-0.15 0.12 0.035">
      <geom name="goal_geom" type="cylinder" size="0.04 0.005" rgba="0.10 0.80 0.25 0.55" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="panda_motor1" joint="panda_joint1" kp="750" ctrlrange="-2.8973 2.8973" forcerange="-180 180"/>
    <position name="panda_motor2" joint="panda_joint2" kp="850" ctrlrange="-1.7628 1.7628" forcerange="-180 180"/>
    <position name="panda_motor3" joint="panda_joint3" kp="700" ctrlrange="-2.8973 2.8973" forcerange="-150 150"/>
    <position name="panda_motor4" joint="panda_joint4" kp="850" ctrlrange="-3.0718 -0.0698" forcerange="-180 180"/>
    <position name="panda_motor5" joint="panda_joint5" kp="650" ctrlrange="-2.8973 2.8973" forcerange="-120 120"/>
    <position name="panda_motor6" joint="panda_joint6" kp="700" ctrlrange="-0.0175 3.7525" forcerange="-120 120"/>
    <position name="panda_motor7" joint="panda_joint7" kp="600" ctrlrange="-2.8973 2.8973" forcerange="-100 100"/>
    <position name="left_finger_motor" joint="left_finger_joint" kp="1000" ctrlrange="0 0.022" forcerange="-20 20"/>
    <position name="right_finger_motor" joint="right_finger_joint" kp="1000" ctrlrange="0 0.022" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""


class PickPlaceEnv:
    ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
    ARM_ACTUATORS = tuple(f"panda_motor{i}" for i in range(1, 8))
    # Neutral Panda posture used by robosuite / LIBERO.
    HOME_QPOS = np.array(
        [0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0,
         0.0, np.pi - 0.2, np.pi / 4.0],
        dtype=np.float64,
    )
    CONTROL_NSTEP = 100
    MAX_JOINT_TARGET_DELTA = 0.04

    MAX_DPOS = 0.012
    FINGER_TRAVEL = 0.022
    GRASP_OFFSET = 0.055
    TABLE_TOP = 0.03
    CUBE_HALF = 0.016
    CUBE_SUPPORT_Z = TABLE_TOP + CUBE_HALF
    MIN_EEF_Z = TABLE_TOP + 0.062

    def __init__(self, image_size=448):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
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
        self.eef_body_id = self.model.body("eef").id
        self.cube_body_id = self.model.body("cube").id
        self.finger_joints = ("left_finger_joint", "right_finger_joint")
        self.finger_actuators = ("left_finger_motor", "right_finger_motor")
        self.gripper = 1.0
        self.attached = False
        self.release_counter = 0
        self.tool_z_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.target_eef = np.array([0.105, -0.18, 0.225], dtype=np.float64)
        self.arm_target = self.HOME_QPOS.copy()

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # LIBERO-style placement sampling stays inside the robot's validated
        # task workspace. Sampling the whole table creates unreachable scenes,
        # which are reset failures rather than useful expert demonstrations.
        cube_xy = rng.uniform([0.025, -0.10], [0.085, -0.045])
        goal_xy = rng.uniform([-0.15, 0.08], [-0.06, 0.13])
        self.model.body("goal").pos[:] = [goal_xy[0], goal_xy[1], 0.035]

        cube_qadr = self.model.joint("cube_free").qposadr[0]
        self.data.qpos[cube_qadr:cube_qadr + 7] = [
            cube_xy[0], cube_xy[1], self.CUBE_SUPPORT_Z,
            1.0, 0.0, 0.0, 0.0,
        ]

        self.data.qpos[self.arm_qpos_ids] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self.tool_z_ref[:] = self.data.xmat[self.eef_body_id].reshape(3, 3)[:, 2]
        self.target_eef[:] = self.data.body("eef").xpos
        arm_targets = self._solve_ik(self.target_eef, self.HOME_QPOS)
        self.data.qpos[self.arm_qpos_ids] = arm_targets
        for joint_name in self.finger_joints:
            self.data.qpos[self.model.joint(joint_name).qposadr[0]] = self.FINGER_TRAVEL

        self.gripper = 1.0
        self.attached = False
        self.release_counter = 0
        self.arm_target = arm_targets.copy()
        self._set_actuator_targets(arm_targets)
        self._apply_grasp_force()
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        eef = self.data.body("eef").xpos.copy()
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()
        state = np.concatenate([
            eef, cube, goal,
            np.array([self.gripper], dtype=np.float32),
        ]).astype(np.float32)

        front = self.render("front")
        overhead = self.render("overhead")
        wrist = self.render("wrist")
        return {
            "state": state,
            "image": front,
            "image_front": front,
            "image_overhead": overhead,
            "image_wrist": wrist,
        }

    def render(self, camera="front"):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -self.MAX_DPOS, self.MAX_DPOS)
        self.gripper = float(np.clip(action[3], 0.0, 1.0))

        current_eef = self.data.body("eef").xpos.copy()
        self.target_eef[:] = np.clip(
            current_eef + dpos,
            [-0.22, -0.30, self.MIN_EEF_Z],
            [0.20, 0.22, 0.38],
        )
        arm_targets = self._solve_ik(
            self.target_eef,
            self.arm_target.copy(),
        )
        self._set_actuator_targets(arm_targets)
        self._apply_grasp_force()
        mujoco.mj_step(self.model, self.data, nstep=self.CONTROL_NSTEP)
        self._update_grasp_logic()
        mujoco.mj_forward(self.model, self.data)

        obs = self.obs()
        done = self.success()
        return obs, done

    def step_video(self, action, frames_per_step=4, cameras=("front", "overhead", "wrist")):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -self.MAX_DPOS, self.MAX_DPOS)
        self.gripper = float(np.clip(action[3], 0.0, 1.0))
        current_eef = self.data.body("eef").xpos.copy()
        self.target_eef[:] = np.clip(
            current_eef + dpos,
            [-0.22, -0.30, self.MIN_EEF_Z],
            [0.20, 0.22, 0.38],
        )
        arm_targets = self._solve_ik(
            self.target_eef,
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
            if step_in_sweep in render_steps:
                mujoco.mj_forward(self.model, self.data)
                for cam in cameras:
                    frames[cam].append(self.render(cam).copy())

        mujoco.mj_forward(self.model, self.data)
        self._update_grasp_logic()
        obs = self.obs()
        done = self.success()
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

        finger_target = self.FINGER_TRAVEL * self.gripper
        for actuator_name in self.finger_actuators:
            self.data.ctrl[self.model.actuator(actuator_name).id] = finger_target

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

            pos_err = target - self.data.body("eef").xpos

            R = self.data.xmat[self.eef_body_id].reshape(3, 3)
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
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.eef_body_id)
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
        if self.gripper < 0.5 and not self.attached:
            self.attached = (
                self._has_two_sided_grasp_contact()
                or self._tips_enclose_cube()
            )
            if self.attached:
                self.release_counter = 0
        if self.attached:
            if self.gripper > 0.8:
                self.release_counter += 1
            else:
                self.release_counter = 0

            if self.release_counter >= 3:
                self.attached = False
                self.release_counter = 0
                self.data.xfrc_applied[self.cube_body_id] = 0.0
        else:
            self.release_counter = 0
            if self.gripper > 0.8:
                self.data.xfrc_applied[self.cube_body_id] = 0.0

    def _apply_grasp_force(self):
        self.data.xfrc_applied[self.cube_body_id] = 0.0
        if not self.attached or self.gripper >= 0.5:
            return
        eef = self.data.body("eef").xpos
        eef_R = self.data.xmat[self.eef_body_id].reshape(3, 3)
        target = eef + eef_R @ np.array([0.0, 0.0, self.GRASP_OFFSET])
        cube = self.data.body("cube").xpos
        cube_vadr = self.model.joint("cube_free").dofadr[0]
        cube_vel = self.data.qvel[cube_vadr:cube_vadr + 3]
        force = 40.0 * (target - cube) - 2.8 * cube_vel
        self.data.xfrc_applied[self.cube_body_id, :3] = np.clip(force, -4.0, 4.0)
    def _has_two_sided_grasp_contact(self):
        cube_id = self.model.geom("cube_geom").id
        contacted = set()
        pad_names = {"left_finger_tip", "right_finger_tip"}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist > 1e-3:
                continue
            names = {
                self.model.geom(contact.geom1).name,
                self.model.geom(contact.geom2).name,
            }
            if cube_id in (contact.geom1, contact.geom2):
                contacted.update(names & pad_names)
        return contacted == pad_names

    def _tips_enclose_cube(self, tol=0.025, z_tol=0.030):
        """Fallback for near-grasp: both red pads are beside the cube at cube
        height while the gripper is closed, even if the contact solver reports
        a sub-millimeter gap."""
        cube = self.data.body("cube").xpos
        tips = []
        for pad in ("left_finger_tip", "right_finger_tip"):
            gid = self.model.geom(pad).id
            tips.append(self.data.geom_xpos[gid].copy())
        for t in tips:
            if abs(t[2] - cube[2]) > z_tol:
                return False
            if np.linalg.norm(t[:2] - cube[:2]) > tol:
                return False
        sep = tips[0][:2] - tips[1][:2]
        mid = 0.5 * (tips[0][:2] + tips[1][:2])
        return float(np.dot(sep, cube[:2] - mid)) < 0.0

    def success(self):
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()
        xy_ok = np.linalg.norm(cube[:2] - goal[:2]) < 0.04
        z_ok = abs(cube[2] - self.CUBE_SUPPORT_Z) < 0.012
        return bool(xy_ok and z_ok)


def scripted_expert(obs):
    state = obs["state"]
    eef = state[0:3]
    cube = state[3:6]
    goal = state[6:9]
    gripper = state[9]

    safe_z = 0.18
    grasp_z = cube[2] + PickPlaceEnv.GRASP_OFFSET
    place_z = 0.10  # lower release height so the cube can settle before timeout
    xy_tol = 0.025
    z_tol = 0.018

    cube_xy = cube[:2]
    goal_xy = goal[:2]
    eef_xy = eef[:2]

    if gripper > 0.5:
        if np.linalg.norm(eef_xy - cube_xy) > xy_tol:
            target = np.array([cube[0], cube[1], safe_z], dtype=np.float32)
            grip_cmd = 1.0
        elif eef[2] > grasp_z + z_tol:
            target = np.array([cube[0], cube[1], grasp_z], dtype=np.float32)
            grip_cmd = 1.0
        else:
            target = eef.copy()
            grip_cmd = 0.0
    else:
        if eef[2] < safe_z - z_tol:
            target = np.array([eef[0], eef[1], safe_z], dtype=np.float32)
            grip_cmd = 0.0
        elif np.linalg.norm(eef_xy - goal_xy) > xy_tol:
            target = np.array([goal[0], goal[1], safe_z], dtype=np.float32)
            grip_cmd = 0.0
        elif eef[2] > place_z + z_tol:
            target = np.array([goal[0], goal[1], place_z], dtype=np.float32)
            grip_cmd = 0.0
        else:
            target = eef.copy()
            grip_cmd = 1.0

    dpos = np.clip(target - eef, -0.012, 0.012)
    return np.r_[dpos, grip_cmd].astype(np.float32)

class ScriptedExpertPolicy:

    def __init__(self, env):
        self.env = env
        self.phase = "approach"
        self.phase_steps = 0
        self.previous_dpos = np.zeros(3, dtype=np.float32)

    def _set_phase(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.phase_steps = 0
            self.previous_dpos[:] = 0.0

    def _move(self, target, gripper):
        eef = self.env.data.body("eef").xpos.copy()
        error = np.asarray(target, dtype=np.float64) - eef
        distance = np.linalg.norm(error)
        if distance < 1e-8:
            raw = np.zeros(3, dtype=np.float64)
        else:
            raw = error * (min(self.env.MAX_DPOS, 0.65 * distance) / distance)

        low_pass = 0.45 * raw + 0.55 * self.previous_dpos
        dpos = self.previous_dpos + np.clip(
            low_pass - self.previous_dpos, -0.004, 0.004
        )
        self.previous_dpos = dpos.astype(np.float32)
        return np.r_[self.previous_dpos, gripper].astype(np.float32)

    def __call__(self, obs):
        self.phase_steps += 1
        state = obs["state"]
        eef, cube, goal = state[:3], state[3:6], state[6:9]
        safe_z = 0.18
        grasp_z = cube[2] + self.env.GRASP_OFFSET
        place_z = 0.10

        if self.phase == "approach":
            target = np.array([cube[0], cube[1], safe_z])
            if np.linalg.norm(eef - target) < 0.014:
                self._set_phase("descend")
                target = np.array([cube[0], cube[1], grasp_z])
            return self._move(target, 1.0)

        if self.phase == "descend":
            target = np.array([cube[0], cube[1], grasp_z])
            if (np.linalg.norm(eef[:2] - cube[:2]) < 0.012 and
                    eef[2] - cube[2] < 0.073):
                self._set_phase("close")
                return self._move(eef, 0.0)
            return self._move(target, 1.0)

        if self.phase == "close":
            if self.env.attached:
                self._set_phase("lift")
                return self._move(np.array([eef[0], eef[1], safe_z]), 0.0)
            if self.phase_steps > 8:
                self._set_phase("approach")
                return self._move(np.array([cube[0], cube[1], safe_z]), 1.0)
            return self._move(eef, 0.0)

        if self.phase == "lift":
            target = np.array([eef[0], eef[1], safe_z])
            if eef[2] > safe_z - 0.012:
                self._set_phase("transfer")
                target = np.array([goal[0], goal[1], safe_z])
            return self._move(target, 0.0)

        if self.phase == "transfer":
            target = np.array([goal[0], goal[1], safe_z])
            if np.linalg.norm(eef - target) < 0.014:
                self._set_phase("lower")
                target = np.array([goal[0], goal[1], place_z])
            return self._move(target, 0.0)

        if self.phase == "lower":
            target = np.array([goal[0], goal[1], place_z])
            if np.linalg.norm(eef - target) < 0.010:
                self._set_phase("release")
                return self._move(eef, 1.0)
            return self._move(target, 0.0)

        if self.phase == "release":
            if self.phase_steps >= 5:
                self._set_phase("settle")
            return self._move(eef, 1.0)

        return self._move(eef, 1.0)
