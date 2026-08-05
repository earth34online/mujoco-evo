# analyze_jitter_source.py
#
# 回答"抖动来自模型还是控制"：
#   1) 对同一观测，分别用"模型"和"scripted_expert"生成 10 步 action chunk；
#   2) 对比两者的原始位置动作序列（是否反向、是否平滑、幅度）；
#   3) 统计"反向模式"（相邻动作方向相反 / 符号翻转）指标。
#
# 如果模型 chunk 明显振荡、专家 chunk 平滑 => 抖动来自模型生成；
# 如果专家 chunk 也振荡 => 控制/数据层有问题。
#
# 用法（WSL Evo1 环境）：
#   MUJOCO_GL=egl /home/user/miniconda3/envs/Evo1/bin/python analyze_jitter_source.py
#
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVO_ROOT = os.path.join(PROJECT_ROOT, "Evo-1", "Evo_1")
sys.path.insert(0, EVO_ROOT)

from pick_place_env import PickPlaceEnv, scripted_expert
from scripts.Evo1_server import load_model_and_normalizer, decode_image_from_list

PROMPT = "pick up the blue cube and place it on the green target"


def infer_chunk(model, normalizer, obs):
    images = [decode_image_from_list(img[..., ::-1]) for img in
              [obs["image_front"], obs["image_overhead"], obs["image_wrist"]]]
    state = torch.tensor(obs["state"], dtype=torch.float32, device="cuda").unsqueeze(0)
    state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device="cuda")], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)
    image_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
    action_mask = torch.tensor([[1, 1, 1, 1] + [0] * 20], dtype=torch.int32, device="cuda")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(images=images, image_mask=image_mask, prompt=PROMPT,
                                     state_input=norm_state, action_mask=action_mask)
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
    return action.cpu().numpy()[:, :4]


def reverse_metric(actions):
    """相邻动作方向相反的比例 + 符号翻转次数（位置维）。"""
    pos = actions[:, :3]
    # 相邻动作内积 < 0 => 方向相反
    dots = np.sum(pos[1:] * pos[:-1], axis=1)
    opposite = (dots < 0).mean()
    # 每维符号翻转次数
    flips = sum(int((np.diff(np.sign(pos[:, d])) != 0).sum()) for d in range(3))
    return opposite, flips


def main():
    ckpt = os.path.join(PROJECT_ROOT, "ckpt", "evo1_mujoco_panda7_multiview_h10_clean_v2", "step_best")
    print(f"Loading model from {ckpt} ...", flush=True)
    model, normalizer = load_model_and_normalizer(ckpt)
    env = PickPlaceEnv()

    print(f"\n{'seed':>4} | {'model opp%':>10} {'model flips':>11} {'model |a|mean':>12} | "
          f"{'expert opp%':>11} {'expert flips':>12} {'expert |a|mean':>13}", flush=True)
    for seed in range(5):
        obs = env.reset(seed=seed)
        model_chunk = infer_chunk(model, normalizer, obs)

        # expert 10 步 chunk：滚动执行 expert，收集前 10 个动作
        expert_actions = []
        o = obs
        for _ in range(10):
            a = scripted_expert(o)
            expert_actions.append(a)
            o, _ = env.step(a)
        expert_chunk = np.array(expert_actions)

        m_opp, m_flips = reverse_metric(model_chunk)
        e_opp, e_flips = reverse_metric(expert_chunk)
        print(f"{seed:4d} | {m_opp:10.2f} {m_flips:11d} {np.abs(model_chunk[:, :3]).mean():12.4f} | "
              f"{e_opp:11.2f} {e_flips:12d} {np.abs(expert_chunk[:, :3]).mean():13.4f}", flush=True)

        if seed == 0:
            print("\n  model chunk dx,dy,dz (raw):")
            for r in model_chunk:
                print(f"    {r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}  grip={r[3]:+.2f}")
            print("  expert chunk dx,dy,dz (raw):")
            for r in expert_chunk:
                print(f"    {r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}  grip={r[3]:+.2f}")


if __name__ == "__main__":
    main()
