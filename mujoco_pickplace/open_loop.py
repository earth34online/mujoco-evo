# 开环测试
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVO_ROOT = os.path.join(PROJECT_ROOT, "Evo-1", "Evo_1")
sys.path.insert(0, EVO_ROOT)

from pick_place_env import PickPlaceEnv
from scripts.Evo1_server import load_model_and_normalizer, decode_image_from_list

PROMPT = "pick up the blue cube and place it on the green target"
CAMERA_KEYS = ("front",)

EXPECTED_GRIPPER = {
    "approach": 1.0,
    "descend": 1.0,
    "close": 0.0,
    "lift": 0.0,
    "transfer": 0.0,
    "lower": 0.0,
    "release": 1.0,
    "settle": 1.0,
}

def infer_chunk(model, normalizer, obs, num_samples=1):
    """One model forward on the current observation -> denormalized [horizon, 24] chunk."""
    front = obs["image_front"]
    images_rgb = [front, np.zeros_like(front), np.zeros_like(front)]
    # Mirror the server pipeline exactly: the client flips RGB->BGR, the server
    # decodes BGR->RGB. Net effect is the model sees the same RGB as training.
    image_size = int(model.config.get("image_size", 448))
    images = [decode_image_from_list(img[..., ::-1], image_size=image_size) for img in images_rgb]

    use_state = bool(model.config.get("use_state", True))
    norm_state = None
    if use_state:
        state = torch.tensor(obs["robot_state"], dtype=torch.float32, device="cuda")
        state = state.unsqueeze(0)
        if state.shape != (1, 8):
            raise ValueError(
                "Expected 8-D robot proprioception, got "
                f"{tuple(state.shape)}"
            )
        norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    image_mask = torch.tensor([1, 0, 0], dtype=torch.int32, device="cuda")
    action_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 1] + [0] * 17], dtype=torch.int32, device="cuda")

    num_samples = max(1, int(num_samples))
    samples = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_samples):
            action = model.run_inference(
                images=images,
                image_mask=image_mask,
                prompt=PROMPT,
                state_input=norm_state,
                action_mask=action_mask,
            )
            action = action.reshape(1, -1, 24)
            action = normalizer.denormalize_action(action[0])
            samples.append(action.float())
    return torch.stack(samples, dim=0).mean(dim=0).cpu().numpy()


def action_jitter_metrics(chunk):
    rows = chunk[:, :7]
    pos = rows[:, :3]
    grp = rows[:, 6]
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


def _sample_replay_indices(total_frames, stride=1, max_frames=None):
    """Return deterministic replay indices, optionally capped per episode.

    Defaults preserve the original full-frame replay behavior.  The optional cap
    is only for fast smoke tests: it keeps coverage across the whole trajectory
    instead of looking only at the prefix.
    """
    stride = max(1, int(stride))
    indices = np.arange(0, total_frames, stride, dtype=np.int64)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        pick = np.linspace(0, len(indices) - 1, int(max_frames), dtype=np.int64)
        indices = indices[pick]
    return indices

def assert_gripper_labels(phases, actions_gt, episode_index):
    """Fail fast if expert.phase and action[6] disagree."""
    if actions_gt.ndim != 2 or actions_gt.shape[1] < 7:
        raise AssertionError(
            f"episode {episode_index}: expected actions shaped [N, >=7], "
            f"got {actions_gt.shape}"
        )

    if len(phases) != len(actions_gt):
        raise AssertionError(
            f"episode {episode_index}: phase/action length mismatch: "
            f"{len(phases)} vs {len(actions_gt)}"
        )

    unknown_phases = sorted(
        set(str(phase) for phase in phases) - set(EXPECTED_GRIPPER)
    )
    if unknown_phases:
        raise AssertionError(
            f"episode {episode_index}: unknown expert phases: "
            f"{unknown_phases}"
        )

    for frame_index, (phase, action) in enumerate(
        zip(phases, actions_gt)
    ):
        phase = str(phase)
        expected = EXPECTED_GRIPPER[phase]
        actual = float(action[6])

        if abs(actual - expected) > 1e-6:
            raise AssertionError(
                f"gripper label mismatch: "
                f"episode={episode_index}, "
                f"frame={frame_index}, "
                f"phase={phase}, "
                f"value={actual}, "
                f"expected={expected}"
            )

    print(
        f"[PASS] episode {episode_index}: "
        f"gripper labels match expert phases",
        flush=True,
    )

def assert_normalization_equivalence(
    normalizer,
    states,
    actions_gt,
):
    states = torch.as_tensor(states, dtype=torch.float32, )
    if states.ndim != 2 or states.shape[1] != 8:
        raise AssertionError(f"Expected states [N, 8], got {tuple(states.shape)}")

    # ===== Training state path =====
    state_min_8 = normalizer.state_min[:8].cpu()
    state_max_8 = normalizer.state_max[:8].cpu()

    train_state_8 = torch.clamp(
        2.0 * (states - state_min_8) / (state_max_8 - state_min_8 + 1e-8) - 1.0,
        -1.0,
        1.0,
    )
    train_state_24 = torch.zeros((states.shape[0], 24), dtype=torch.float32, )
    train_state_24[:, :8] = train_state_8
    raw_state_24 = torch.zeros((states.shape[0], 24), dtype=torch.float32, )
    raw_state_24[:, :8] = states
    server_state_24 = normalizer.normalize_state(raw_state_24).cpu()
    state_diff = torch.max(torch.abs(train_state_24 - server_state_24)).item()

    print(
        f"[state pipeline] "
        f"max_abs_diff={state_diff:.10g}",
        flush=True,
    )
    assert state_diff < 1e-6, (
        "train/server state mismatch: "
        f"max_diff={state_diff}"
    )

    # ===== Action roundtrip =====
    actions_gt = torch.as_tensor(actions_gt, dtype=torch.float32, )
    if (actions_gt.ndim != 2 or actions_gt.shape[1] != 7):
        raise AssertionError(
            f"Expected actions [N, 7], "
            f"got {tuple(actions_gt.shape)}"
        )
    action_min_7 = normalizer.action_min[:7].cpu()
    action_max_7 = normalizer.action_max[:7].cpu()
    normalized_7 = torch.clamp(
        2.0 * (actions_gt - action_min_7) / (action_max_7 - action_min_7 + 1e-8) - 1.0,
        -1.0,
        1.0,
    )
    inactive_diff = torch.max(torch.abs(normalized_7[:, 3:6])).item()
    assert inactive_diff < 1e-6, (
        "inactive rotation dimensions are not neutral: "
        f"max_abs={inactive_diff}"
    )
    normalized_24 = torch.zeros((actions_gt.shape[0], 24), dtype=torch.float32, )
    normalized_24[:, :7] = normalized_7
    recovered = normalizer.denormalize_action(normalized_24)[:, :7].cpu()
    action_diff = torch.max(torch.abs(recovered - actions_gt)).item()

    print(
        f"[action roundtrip] "
        f"max_abs_diff={action_diff:.10g}",
        flush=True,
    )
    assert action_diff < 1e-6, (
        "action normalization roundtrip mismatch: "
        f"max_diff={action_diff}"
    )
    print(
        "[PASS] state/action normalization invariants",
        flush=True,
    )

def run_replay_test(ckpt, replay_dir, num_episodes, replay_stride=1,
                    max_frames_per_episode=None, num_samples=1):
    """Offline open-loop check: feed the recorded observations of held-out test
    episodes to the policy and compare the first predicted action of each chunk
    against the recorded expert action (ground truth)."""
    import pandas as pd
    import av

    model, normalizer, _ = load_model_and_normalizer(ckpt)
    image_size = int(model.config.get("image_size", 448, ))
    image_pipeline_checked = False
    normalization_checked = False
    with open(os.path.join(replay_dir, "meta", "episodes.jsonl"), "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = sorted(rows, key=lambda r: r["episode_index"])[:num_episodes]

    def read_frames(path):
        frames = []

        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

        return frames

    pos_errs, grip_acc, grip_errs = [], [], []
    phase_pos_err = {}
    phase_grip_acc = {}
    phase_grip_err = {}
    phase_grip_counts = {}
    for row in rows:
        table = pd.read_parquet(os.path.join(replay_dir, row["data_path"]))
        frames = {camera: read_frames(os.path.join(replay_dir, row["video_paths"][camera]))
                  for camera in CAMERA_KEYS}
        
        if not image_pipeline_checked:
            if not frames["front"]:
                raise AssertionError("Decoded front video contains no frames")
            assert_image_pipeline_equivalence(
                frames["front"][0],
                image_size=image_size,
            )
            image_pipeline_checked = True
            
        states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        actions_gt = np.stack(table["action"].to_numpy()).astype(np.float32)
        if not normalization_checked:
            assert_normalization_equivalence(normalizer, states, actions_gt, )
            normalization_checked = True
            
        if "expert.phase" not in table.columns:
            raise AssertionError(
                f"episode {row['episode_index']} "
                "is missing expert.phase"
            )
        phases = (table["expert.phase"].astype(str).to_numpy())
        assert_gripper_labels(phases,actions_gt,episode_index=row["episode_index"],)
        replay_indices = _sample_replay_indices(
            len(table),
            stride=replay_stride,
            max_frames=max_frames_per_episode,
        )
        print(
            f"[replay ep {row['episode_index']}] "
            f"{len(replay_indices)}/{len(table)} sampled frames",
            flush=True,
        )

        for i, t in enumerate(replay_indices):
            obs = {
                "robot_state": states[t],
                "image_front": frames["front"][t],
            }
            chunk = infer_chunk(model, normalizer, obs, num_samples=num_samples)
            horizon = min(len(chunk), 14)
            gt_indices = np.minimum(
                np.arange(t, t + horizon, dtype=np.int64),
                len(actions_gt) - 1,
            )
            pred_chunk = chunk[:horizon, :7]
            gt_chunk = actions_gt[gt_indices, :7]
            chunk_pos_errs = np.linalg.norm(
                pred_chunk[:, :3] - gt_chunk[:, :3],
                axis=1,
            )
            chunk_grip_acc = (
                (pred_chunk[:, 6] > 0.5) == (gt_chunk[:, 6] > 0.5)
            ).astype(np.int32)
            chunk_grip_errs = np.abs(pred_chunk[:, 6] - gt_chunk[:, 6])

            pos_errs.extend(chunk_pos_errs.astype(float).tolist())
            grip_acc.extend(chunk_grip_acc.astype(int).tolist())
            grip_errs.extend(chunk_grip_errs.astype(float).tolist())

            for offset, gt_index in enumerate(gt_indices):
                pred = pred_chunk[offset]
                gt = gt_chunk[offset]
                pos_err = float(chunk_pos_errs[offset])
                grip_ok = int(chunk_grip_acc[offset])
                grip_abs_err = float(chunk_grip_errs[offset])
                pred_open = bool(pred[6] > 0.5)
                gt_open = bool(gt[6] > 0.5)
                phase = str(phases[gt_index])
                phase_pos_err.setdefault(phase, []).append(pos_err)
                phase_grip_acc.setdefault(phase, []).append(grip_ok)
                phase_grip_err.setdefault(phase, []).append(grip_abs_err)
                counts = phase_grip_counts.setdefault(
                    phase,
                    {
                        "gt_open": 0,
                        "gt_closed": 0,
                        "pred_open": 0,
                        "pred_closed": 0,
                        "false_open": 0,
                        "false_closed": 0,
                    },
                )
                counts["gt_open" if gt_open else "gt_closed"] += 1
                counts["pred_open" if pred_open else "pred_closed"] += 1
                if pred_open and not gt_open:
                    counts["false_open"] += 1
                elif (not pred_open) and gt_open:
                    counts["false_closed"] += 1
            if (i + 1) % 8 == 0 or i + 1 == len(replay_indices):
                print(
                    f"  frames {i + 1:3d}/{len(replay_indices):3d} "
                    f"latest_horizon_ADE={chunk_pos_errs.mean():.4f} "
                    f"latest_horizon_grip_acc={chunk_grip_acc.mean():.3f}",
                    flush=True,
                )

    pos_errs = np.asarray(pos_errs)
    grip_acc = np.asarray(grip_acc)
    grip_errs = np.asarray(grip_errs)
    print(f"\nReplay open-loop over {len(rows)} held-out episodes, {len(pos_errs)} horizon actions:", flush=True)
    print(f"  position action ADE: mean={pos_errs.mean():.4f}  median={np.median(pos_errs):.4f}  p90={np.percentile(pos_errs, 90):.4f}", flush=True)
    print(f"  gripper accuracy: {grip_acc.mean():.3f}   mean |gripper err|: {grip_errs.mean():.3f}", flush=True)
    for phase in sorted(phase_pos_err):
        errs = np.asarray(phase_pos_err[phase])
        g_acc = np.asarray(phase_grip_acc[phase])
        g_err = np.asarray(phase_grip_err[phase])
        counts = phase_grip_counts[phase]
        print(
            f"  phase {phase:<9s} steps={len(errs):4d}  "
            f"ADE={errs.mean():.4f}  grip_acc={g_acc.mean():.3f}  "
            f"|g_err|={g_err.mean():.3f}  "
            f"gt(open/closed)={counts['gt_open']}/{counts['gt_closed']}  "
            f"pred(open/closed)={counts['pred_open']}/{counts['pred_closed']}  "
            f"miss(open/close)={counts['false_closed']}/{counts['false_open']}",
            flush=True,
        )
    return pos_errs.mean(), grip_acc.mean()


def main():
    parser = argparse.ArgumentParser(description="Open-loop Evo-1 policy test in MuJoCo.")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "ckpt",
                                             "evo1_mujoco_pickplace_stage1", "step_best"))
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--replay-dir", type=str, default=None,
                        help="Held-out dataset dir; replay recorded observations "
                             "and compare predictions against expert actions.")
    parser.add_argument("--replay-stride", type=int, default=1,
                        help="Evaluate every Nth frame in replay mode. Default 1 "
                             "keeps the original full replay behavior.")
    parser.add_argument("--max-frames-per-episode", type=int, default=None,
                        help="Optional cap for fast replay smoke tests. Frames are "
                             "sampled evenly across the trajectory.")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Average multiple stochastic flow-matching samples per "
                             "observation. Default 1 preserves original behavior.")
    args = parser.parse_args()

    if args.replay_dir:
        run_replay_test(
            args.ckpt, args.replay_dir, args.num_episodes,
            replay_stride=args.replay_stride,
            max_frames_per_episode=args.max_frames_per_episode,
            num_samples=args.num_samples,
        )
        return

    print(f"Loading model from {args.ckpt} ...", flush=True)
    model, normalizer, _ = load_model_and_normalizer(args.ckpt)
    print("Model loaded.", flush=True)

    env = PickPlaceEnv()
    total_success = 0
    for ep in range(args.num_episodes):
        obs = env.reset(seed=args.seed_base + ep)
        chunk = infer_chunk(model, normalizer, obs, num_samples=args.num_samples)
        horizon = chunk.shape[0]
        jit = action_jitter_metrics(chunk)

        ee = [obs["robot_state"][:3].copy()]
        done = False
        for i in range(horizon):
            a = chunk[i, :7].astype(np.float32)
            a[6] = 1.0 if a[6] > 0.5 else 0.0
            obs, done = env.step(a)
            ee.append(obs["robot_state"][:3].copy())
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
