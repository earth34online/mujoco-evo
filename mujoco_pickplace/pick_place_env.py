import numpy as np
import mujoco

XML = """
<mujoco model="handwritten_panda_pick_place">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <default>
    <joint damping="4.0" armature="0.02"/>
    <geom friction="0.8 0.1 0.1"/>
  </default>
  <visual>
    <global offwidth="448" offheight="448"/>
    <headlight diffuse="0.44 0.44 0.44" ambient="0.11 0.11 0.11" specular="0.05 0.05 0.05"/>
    <rgba haze="0.70 0.74 0.80 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.38 0.44 0.52" rgb2="0.70 0.75 0.80" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.36 0.38 0.40" rgb2="0.52 0.54 0.56" width="256" height="256"/>
    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.01"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0.1 -0.55 2.8" dir="0 0 -1" diffuse="0.54 0.54 0.54" ambient="0.08 0.08 0.08"/>
    <light name="fill_light" pos="-0.9 0.8 1.5" dir="0.4 -0.2 -1" diffuse="0.15 0.15 0.15" ambient="0.01 0.01 0.01"/>
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
                <geom name="link5_geom" type="capsule" fromto="0 0 0 0 0 0.10" size="0.032" rgba="0.76 0.79 0.82 1" contype="0" conaffinity="0"/>

                <body name="panda_link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
                  <joint name="panda_joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom name="joint6_shell" type="sphere" size="0.035" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
                  <geom name="link6_geom" type="capsule" fromto="0 0 0 0.088 0 0" size="0.029" rgba="0.84 0.86 0.88 1" contype="0" conaffinity="0"/>

                  <body name="panda_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
                    <joint name="panda_joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom name="joint7_shell" type="sphere" size="0.032" rgba="0.12 0.14 0.17 1" contype="0" conaffinity="0"/>
                    <geom name="link7_geom" type="capsule" fromto="0 0 0 0 0 0.1065" size="0.026" rgba="0.76 0.79 0.82 1" contype="0" conaffinity="0"/>

                    <body name="eef" pos="0 0 0.1065" quat="0.92388 0 0 -0.382683">
                      <geom name="eef_geom" type="sphere" size="0.030" rgba="0.92 0.20 0.15 1" contype="0" conaffinity="0"/>
                      <geom name="gripper_palm" type="box" pos="0 0 -0.032" size="0.040 0.020 0.011" rgba="0.80 0.82 0.85 1" contype="0" conaffinity="0"/>
                      <camera name="wrist" pos="0 0 -0.020" fovy="66"/>

                      <body name="left_finger_body" pos="0 0.014 -0.074">
                        <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.026"/>
                        <geom name="left_finger" type="box" size="0.010 0.008 0.038" rgba="0.10 0.12 0.15 1" contype="0" conaffinity="0"/>
                      </body>
                      <body name="right_finger_body" pos="0 -0.014 -0.074">
                        <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.026"/>
                        <geom name="right_finger" type="box" size="0.010 0.008 0.038" rgba="0.10 0.12 0.15 1" contype="0" conaffinity="0"/>
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

    <body name="cube" pos="0.12 -0.08 0.07">
      <joint name="cube_free" type="free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05" rgba="0.12 0.42 0.92 1"/>
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
    <position name="left_finger_motor" joint="left_finger_joint" kp="100" ctrlrange="0 0.026" forcerange="-12 12"/>
    <position name="right_finger_motor" joint="right_finger_joint" kp="100" ctrlrange="0 0.026" forcerange="-12 12"/>
  </actuator>
</mujoco>
"""


class PickPlaceEnv:
    ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
    ARM_ACTUATORS = tuple(f"panda_motor{i}" for i in range(1, 8))
    HOME_QPOS = np.array(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        dtype=np.float64,
    )

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
        self.finger_joints = ("left_finger_joint", "right_finger_joint")
        self.finger_actuators = ("left_finger_motor", "right_finger_motor")
        self.gripper = 1.0
        self.attached = False
        self.target_eef = np.array([0.0, -0.18, 0.22], dtype=np.float64)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        cube_xy = rng.uniform([-0.18, -0.12], [0.18, 0.05])
        goal_xy = rng.uniform([-0.18, 0.08], [0.18, 0.18])
        self.model.body("goal").pos[:] = [goal_xy[0], goal_xy[1], 0.035]

        cube_qadr = self.model.joint("cube_free").qposadr[0]
        self.data.qpos[cube_qadr:cube_qadr + 7] = [
            cube_xy[0], cube_xy[1], 0.07,
            1.0, 0.0, 0.0, 0.0,
        ]

        self.data.qpos[self.arm_qpos_ids] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self.target_eef[:] = [0.0, -0.18, 0.22]
        arm_targets = self._solve_ik(self.target_eef, self.HOME_QPOS)
        self.data.qpos[self.arm_qpos_ids] = arm_targets
        for joint_name in self.finger_joints:
            self.data.qpos[self.model.joint(joint_name).qposadr[0]] = 0.026

        self.gripper = 1.0
        self.attached = False
        self._set_actuator_targets(arm_targets)
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
        dpos = np.clip(action[:3], -0.03, 0.03)
        self.gripper = float(np.clip(action[3], 0.0, 1.0))

        current_eef = self.data.body("eef").xpos.copy()
        self.target_eef[:] = np.clip(
            current_eef + dpos,
            [-0.22, -0.30, 0.07],
            [0.20, 0.22, 0.38],
        )
        arm_targets = self._solve_ik(
            self.target_eef,
            self.data.qpos[self.arm_qpos_ids].copy(),
        )
        self._set_actuator_targets(arm_targets)
        mujoco.mj_step(self.model, self.data, nstep=30)
        self._update_grasp_logic()
        mujoco.mj_forward(self.model, self.data)

        obs = self.obs()
        done = self.success()
        return obs, done

    def _set_actuator_targets(self, arm_targets):
        for actuator_name, target in zip(self.arm_actuators, arm_targets):
            self.data.ctrl[self.model.actuator(actuator_name).id] = target

        finger_target = 0.026 * self.gripper
        for actuator_name in self.finger_actuators:
            self.data.ctrl[self.model.actuator(actuator_name).id] = finger_target

    def _solve_ik(self, target, seed_q):
        original_q = self.data.qpos[self.arm_qpos_ids].copy()
        q = np.clip(np.asarray(seed_q, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        target = np.asarray(target, dtype=np.float64)
        identity = np.eye(len(self.arm_joints))

        for _ in range(80):
            self.data.qpos[self.arm_qpos_ids] = q
            mujoco.mj_forward(self.model, self.data)
            error = target - self.data.body("eef").xpos
            if np.linalg.norm(error) < 5e-4:
                break

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.eef_body_id)
            jac = jacp[:, self.arm_dof_ids]
            damping = 2e-3
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(3))
            delta = jac_pinv @ error
            nullspace = identity - jac_pinv @ jac
            delta += nullspace @ (0.025 * (self.HOME_QPOS - q))
            q += np.clip(delta, -0.14, 0.14)
            q = np.clip(q, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

        self.data.qpos[self.arm_qpos_ids] = original_q
        mujoco.mj_forward(self.model, self.data)
        return q

    def _update_grasp_logic(self):
        eef = self.data.body("eef").xpos.copy()
        cube = self.data.body("cube").xpos.copy()

        if self.gripper < 0.5 and np.linalg.norm(eef - cube) < 0.075:
            self.attached = True
        if self.gripper > 0.8:
            self.attached = False

        if self.attached:
            joint = self.model.joint("cube_free")
            qadr = joint.qposadr[0]
            vadr = joint.dofadr[0]
            self.data.qpos[qadr:qadr + 3] = eef + np.array([0.0, 0.0, -0.060])
            self.data.qvel[vadr:vadr + 6] = 0.0

    def success(self):
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()
        xy_ok = np.linalg.norm(cube[:2] - goal[:2]) < 0.04
        z_ok = cube[2] < 0.10
        return bool(xy_ok and z_ok)


def scripted_expert(obs):
    state = obs["state"]
    eef = state[0:3]
    cube = state[3:6]
    goal = state[6:9]
    gripper = state[9]

    safe_z = 0.18
    grasp_z = cube[2] + 0.035
    place_z = goal[2] + 0.060
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

    dpos = np.clip(target - eef, -0.025, 0.025)
    return np.r_[dpos, grip_cmd].astype(np.float32)