# eval_open_loop.py
#
# 开环策略测试：加载训练好的 Evo-1 模型，对每个 episode 只做一次推理，
# 得到 action chunk 后【不回传观测】整段执行，判断：
#   1) 模型生成的 action chunk 本身是否平滑（抖动是否来自模型）；
#   2) 开环执行该 chunk 是否让任务有进展 / 成功。
#
# 这个脚本不启动 websocket server，直接进程内加载模型推理，
# 需要 Evo1 conda 环境（有 torch + InternVL3-1B + CUDA）。
#
# 用法（WSL Evo1 环境）：
#   MUJOCO_GL=egl /home/user/miniconda3/envs/Evo1/bin/python eval_open_loop.py \
#       --ckpt <dir> --num-episodes 8
#
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVO_ROOT = os.path.join(PROJECT_ROOT, "Evo-1", "Evo_1")
sys.path.insert(0, EVO_ROOT)

from pick_place_env import PickPlaceEnv
from scripts.Evo1_server import load_model_and_normalizer, decode_image_from_list

PROMPT = "pick up the blue cube and place it on the green target"


def infer_chunk(model, normalizer, obs):
    """One model forward on the current observation -> denormalized [horizon, 24] chunk."""
    images_rgb = [obs["image_front"], obs["image_overhead"], obs["image_wrist"]]
    # Mirror the server pipeline exactly: the client flips RGB->BGR, the server
    # decodes BGR->RGB. Net effect is the model sees the same RGB as training.
    images = [decode_image_from_list(img[..., ::-1]) for img in images_rgb]

    state = torch.tensor(obs["state"], dtype=torch.float32, device="cuda")
    state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device="cuda")], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    image_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
    action_mask = torch.tensor([[1, 1, 1, 1] + [0] * 20], dtype=torch.int32, device="cuda")

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=PROMPT,
            state_input=norm_state,
            action_mask=action_mask,
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
    return action.cpu().numpy()


def action_jitter_metrics(chunk):
    """How jumpy is the model's action chunk on its own?

    Reports position (dx,dy,dz) and gripper (dim 3) jitter separately, because a
    binary-ish gripper inflates the all-dims mean and hides whether the *arm*
    motion is actually smooth.
    """
    rows = chunk[:, :4]
    pos = rows[:, :3]
    grp = rows[:, 3]
    pos_diffs = np.abs(np.diff(pos, axis=0))
    grp_diffs = np.abs(np.diff(grp))
    return {
        "mean_abs_diff_all": float(np.abs(np.diff(rows, axis=0)).mean()),
        "mean_abs_diff_pos": float(pos_diffs.mean()),
        "max_abs_diff_pos": float(pos_diffs.max()),
        "mean_abs_diff_gripper": float(grp_diffs.mean()),
        "dx_sign_flips": int((np.diff(np.sign(pos[:, 0])) != 0).sum()),
        "gripper_first_last": (float(grp[0]), float(grp[-1])),
    }


def main():
    parser = argparse.ArgumentParser(description="Open-loop Evo-1 policy test in MuJoCo.")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "ckpt",
                                             "evo1_mujoco_panda7_multiview_h10_clean_ds", "step_best"))
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt} ...", flush=True)
    model, normalizer = load_model_and_normalizer(args.ckpt)
    print("Model loaded.", flush=True)

    env = PickPlaceEnv()
    total_success = 0
    for ep in range(args.num_episodes):
        obs = env.reset(seed=args.seed_base + ep)
        chunk = infer_chunk(model, normalizer, obs)
        horizon = chunk.shape[0]
        jit = action_jitter_metrics(chunk)

        # open-loop: execute the whole chunk without re-inference
        ee = [obs["state"][:3].copy()]
        done = False
        for i in range(horizon):
            a = chunk[i, :4].astype(np.float32)
            a[3] = 1.0 if a[3] > 0.5 else 0.0
            obs, done = env.step(a)
            ee.append(obs["state"][:3].copy())
            if done:
                break
        ee = np.array(ee)
        vel = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        flips = int((np.diff(np.sign(np.diff(ee[:, 0]))) != 0).sum())

        total_success += int(done)
        print(
            f"[ep {ep}] chunk={chunk.shape} "
            f"jitter(all={jit['mean_abs_diff_all']:.4f}, pos={jit['mean_abs_diff_pos']:.4f}, "
            f"gripper={jit['mean_abs_diff_gripper']:.4f}, dx_flips={jit['dx_sign_flips']}/{horizon-1}, "
            f"gripper[{jit['gripper_first_last'][0]:.2f}->{jit['gripper_first_last'][1]:.2f}]) "
            f"openloop_success={done} steps={len(ee)} vel_mean={vel.mean():.4f} eef_x_flips={flips}",
            flush=True,
        )

    print(f"\nOpen-loop success: {total_success}/{args.num_episodes}", flush=True)


if __name__ == "__main__":
    main()
