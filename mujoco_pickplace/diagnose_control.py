# diagnose_control.py
#
# 控制环节诊断：验证 "命令 dpos -> 实际 eef 位移" 的映射是否可靠、
# 保持时是否漂移、子步内是否震荡。每个测试都重新 reset，避免状态污染。
#
# 用法（在 WSL 中，mujoco 环境）：
#   MUJOCO_GL=egl /home/user/miniconda3/envs/mujoco/bin/python diagnose_control.py
#
import os
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "egl"
from pick_place_env import PickPlaceEnv


def fresh_env(seed):
    env = PickPlaceEnv()
    env.reset(seed=seed)
    return env


def eef(env):
    return env.data.body("eef").xpos.copy()


def step(env, action):
    obs, done = env.step(np.asarray(action, dtype=np.float32))
    return obs, done


def test_single_command(seed, dpos):
    """干净状态下，单条 dpos 命令的 eef 位移。"""
    env = fresh_env(seed)
    e0 = eef(env)
    obs, done = step(env, np.r_[dpos, 1.0])
    e1 = eef(env)
    print(f"[single] seed={seed} cmd={dpos} -> eef delta={np.round(e1 - e0, 4)} | |d|={np.linalg.norm(e1 - e0):.4f} (want {np.linalg.norm(dpos):.4f})")
    return env, np.linalg.norm(e1 - e0)


def test_hold(seed, steps=10):
    """保持不动（dpos=0），看 eef 是否漂移。"""
    env = fresh_env(seed)
    # 先给一小段正常运动进入"准稳态"，再测保持
    for i in range(3):
        step(env, np.r_[0.0, 0.02, 0.0, 1.0])
    drifts = []
    e0 = eef(env)
    for i in range(steps):
        step(env, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        e1 = eef(env)
        drifts.append(np.linalg.norm(e1 - e0))
        e0 = e1
    print(f"[hold]  seed={seed} per-step eef drift = {np.round(drifts, 4)} | max={max(drifts):.4f}")
    return env, max(drifts)


def test_substep_oscillation(seed, dpos, use_nullspace=True):
    """手动复刻 step() 内部控制，观察 30 个子步内 eef 是否震荡。"""
    import pick_place_env as pp
    orig_solve = pp.PickPlaceEnv._solve_ik

    def solve_patched(self, target, seed_q):
        # 去掉 nullspace 项（只测是否 nullspace 引起漂移）
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
            if use_nullspace:
                nullspace = identity - jac_pinv @ jac
                delta += nullspace @ (0.025 * (self.HOME_QPOS - q))
            q += np.clip(delta, -0.14, 0.14)
            q = np.clip(q, self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        self.data.qpos[self.arm_qpos_ids] = original_q
        mujoco.mj_forward(self.model, self.data)
        return q

    pp.PickPlaceEnv._solve_ik = solve_patched
    env = fresh_env(seed)
    e0 = eef(env)
    env.target_eef[:] = e0 + np.asarray(dpos, dtype=np.float64)
    arm_t = env._solve_ik(env.target_eef, env.data.qpos[env.arm_qpos_ids].copy())
    env._set_actuator_targets(arm_t)
    traj = []
    for s in range(30):
        mujoco.mj_step(env.model, env.data)
        traj.append(eef(env))
    traj = np.array(traj)
    disp = np.linalg.norm(traj - e0, axis=1)
    osc = np.linalg.norm(traj - traj[-1], axis=1)
    tag = "nullspace=ON" if use_nullspace else "nullspace=OFF"
    print(f"[substep] {tag} cmd={dpos}")
    print(f"   disp/substep = {np.round(disp, 4)}")
    print(f"   final@0.6s |d|={disp[-1]:.4f}  (want {np.linalg.norm(dpos):.4f})")
    print(f"   max |eef - final| within window = {osc.max():.4f}  (oscillation amplitude)")
    return env, disp[-1], osc.max()


if __name__ == "__main__":
    print("=" * 70)
    print("CONTROL DIAGNOSIS")
    print("=" * 70)
    print("\n--- 1) single-command response (clean reset each time) ---")
    for dpos in ([0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0.02, 0.01, -0.01]):
        test_single_command(seed=10, dpos=np.asarray(dpos, dtype=np.float64))

    print("\n--- 2) hold test (dpos=0) ---")
    test_hold(seed=11)

    print("\n--- 3) substep oscillation: nullspace ON vs OFF ---")
    env, d_on, o_on = test_substep_oscillation(seed=12, dpos=[0.02, 0.02, 0], use_nullspace=True)
    env, d_off, o_off = test_substep_oscillation(seed=12, dpos=[0.02, 0.02, 0], use_nullspace=False)
    print(f"\n   SUMMARY nullspace ON : final={d_on:.4f} amp={o_on:.4f}")
    print(f"           nullspace OFF: final={d_off:.4f} amp={o_off:.4f}")
