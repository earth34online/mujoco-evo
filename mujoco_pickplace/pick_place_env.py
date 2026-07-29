import numpy as np
import mujoco

XML = """
<mujoco model="simple_pick_place">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual><global offwidth="448" offheight="448"/></visual>

  <worldbody>
    <light name="main_light" pos="0 0 2.5" dir="0 0 -1"/>
    <camera name="front" pos="0.8 -1.0 0.7" xyaxes="1 0 0 0 0.45 0.9"/>

    <geom name="table" type="box" pos="0 0 0" size="0.45 0.35 0.03" rgba="0.65 0.65 0.65 1"/>

    <body name="cube" pos="0.12 -0.08 0.07">
      <joint name="cube_free" type="free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05" rgba="0.1 0.4 0.9 1"/>
    </body>

    <body name="goal" pos="-0.15 0.12 0.035">
      <geom name="goal_geom" type="cylinder" size="0.04 0.005" rgba="0.1 0.8 0.25 0.45" contype="0" conaffinity="0"/>
    </body>

    <body name="eef" pos="0 -0.18 0.22">
      <geom name="eef_geom" type="sphere" size="0.025" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

class PickPlaceEnv:
    def __init__(self, image_size=448):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, image_size, image_size)
        self.gripper = 1.0
        self.attached = False

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        cube_xy = rng.uniform([-0.18, -0.12], [0.18, 0.05])
        goal_xy = rng.uniform([-0.18, 0.08], [0.18, 0.18])

        self.model.body("eef").pos[:] = [0.0, -0.18, 0.22]
        self.model.body("goal").pos[:] = [goal_xy[0], goal_xy[1], 0.035]

        qadr = self.model.joint("cube_free").qposadr[0]
        self.data.qpos[qadr:qadr + 7] = [
            cube_xy[0], cube_xy[1], 0.07,
            1.0, 0.0, 0.0, 0.0,
        ]

        self.gripper = 1.0
        self.attached = False
        mujoco.mj_forward(self.model, self.data)
        return self.obs()

    def obs(self):
        eef = self.model.body("eef").pos.copy()
        cube = self.data.body("cube").xpos.copy()
        goal = self.model.body("goal").pos.copy()

        state = np.concatenate([
            eef, cube, goal,
            np.array([self.gripper], dtype=np.float32),
        ]).astype(np.float32)

        return {"state": state, "image": self.render()}

    def render(self):
        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -0.03, 0.03)
        self.gripper = float(np.clip(action[3], 0.0, 1.0))

        eef_pos = self.model.body("eef").pos
        eef_pos[:] = eef_pos + dpos
        eef_pos[:] = np.clip(
            eef_pos,
            [-0.30, -0.25, 0.06],
            [0.30, 0.25, 0.35],
        )

        self._update_grasp_logic()
        mujoco.mj_step(self.model, self.data, nstep=5)

        obs = self.obs()
        done = self.success()
        return obs, done

    def _update_grasp_logic(self):
        eef = self.model.body("eef").pos.copy()
        cube = self.data.body("cube").xpos.copy()

        if self.gripper < 0.5 and np.linalg.norm(eef - cube) < 0.06:
            self.attached = True

        if self.gripper > 0.8:
            self.attached = False

        if self.attached:
            qadr = self.model.joint("cube_free").qposadr[0]
            self.data.qpos[qadr:qadr + 3] = eef + np.array([0.0, 0.0, -0.045])
            self.data.qvel[:] = 0.0

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

    safe_z = 0.20
    grasp_z = cube[2] + 0.035
    place_z = goal[2] + 0.060
    xy_tol = 0.018
    z_tol = 0.012

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
