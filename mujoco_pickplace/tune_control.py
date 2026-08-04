# tune_control.py
#
# 控制参数扫描：定位位置控制器欠阻尼震荡的问题，找到(阻尼, kp, nstep)合适组合。
# 通过 patch pick_place_env.XML（joint damping）与 PickPlaceEnv.step（nstep）实现。
#
# 用法（WSL mujoco 环境）：
#   MUJOCO_GL=egl /home/user/miniconda3/envs/mujoco/bin/python tune_control.py
#
import os
import re
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "egl"
import pick_place_env as pp
from pick_place_env import PickPlaceEnv


def build_env(joint_damping=None, nstep=30):
    xml = pp.XML
    if joint_damping is not None:
        xml = xml.replace('<joint damping="4.0" armature="0.02"/>',
                          f'<joint damping="{joint_damping}" armature="0.02"/>')
    pp.XML = xml
    env = PickPlaceEnv()
    _orig_step = PickPlaceEnv.step

    def step(self, action, _nstep=nstep):
        action = np.asarray(action, dtype=np.float32)
        dpos = np.clip(action[:3], -0.03, 0.03)
        self.gripper = float(np.clip(action[3], 0.0, 1.0))
        current_eef = self.data.body("eef").xpos.copy()
        self.target_eef[:] = np.clip(current_eef + dpos, [-0.22, -0.30, 0.07], [0.20, 0.22, 0.38])
        arm_targets = self._solve_ik(self.target_eef, self.data.qpos[self.arm_qpos_ids].copy())
        self._set_actuator_targets(arm_targets)
        mujoco.mj_step(self.model, self.data, nstep=_nstep)
        self._update_grasp_logic()
        mujoco.mj_forward(self.model, self.data)
        obs = self.obs()
        done = self.success()
        return obs, done

    PickPlaceEnv.step = step
    return env


def eef(env):
    return env.data.body("eef").xpos.copy()


def test_single(env, dpos):
    e0 = eef(env)
    env.step(np.r_[dpos, 1.0])
    e1 = eef(env)
    return np.linalg.norm(e1 - e0)


def test_hold(env, steps=8):
    e0 = eef(env)
    drifts = []
    for _ in range(steps):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        e1 = eef(env)
        drifts.append(np.linalg.norm(e1 - e0))
        e0 = e1
    return max(drifts)


def main():
    print("sweep: (joint_damping, nstep) -> |actual-want| mean over 4 cmds | hold max drift")
    for damping in (4.0, 10.0, 20.0, 40.0):
        for nstep in (30, 10, 5, 2):
            env = build_env(joint_damping=damping, nstep=nstep)
            errs = []
            for dpos in ([0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0.02, 0.01, -0.01]):
                env.reset(seed=20)
                d = test_single(env, np.asarray(dpos))
                want = np.linalg.norm(dpos)
                errs.append(abs(d - want))
            env.reset(seed=21)
            hold = test_hold(env)
            print(f"  damping={damping:5.1f} nstep={nstep:3d} -> err-mean={np.mean(errs):.4f}  |  hold-max={hold:.4f}")


if __name__ == "__main__":
    main()
