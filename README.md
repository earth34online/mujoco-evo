# MuJoCo + Evo-1 Pick-and-Place Demo

This repository is a minimal end-to-end demo for:

```text
MuJoCo pick-and-place task
-> scripted expert data collection
-> Evo-1/LeRobot-style dataset conversion
-> Evo-1 DeepSpeed training
-> Evo-1 websocket inference server
-> MuJoCo rollout evaluation
```

The goal is not to rewrite Evo-1. MuJoCo only replaces the benchmark environment/client layer, similar to how Evo-1 uses `LIBERO_evaluation/libero_client_4tasks.py` or `MetaWorld_evaluation/mt50_evo1_client_prompt.py`.

## Repository Layout

```text
mujoco_pickplace/                 MuJoCo task, data collection, conversion, evaluation client
Evo-1/Evo_1/                      Minimal Evo-1 files needed for training/server inference
.gitignore                        Excludes datasets, checkpoints, videos, caches, logs
```

Generated data and training outputs are intentionally not tracked:

```text
Mujoco_training_dataset/
ckpt/
mujoco_pickplace/outputs/
mujoco_pickplace/logs/
__pycache__/
```

## 1. Collect MuJoCo Demonstrations

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py --raw-dir /home/user/mujoco+evo/Mujoco_training_dataset/raw_mujoco_panda7_multiview_fixed --num-episodes 100 --max-steps 80
```

`collect_data.py` runs `scripted_expert()` in `pick_place_env.py` and only keeps
episodes that succeed AND pass quality gates (smooth, no wild action spikes).
This keeps the training data clean: the current controller is critically damped
(`joint damping=10`, `CONTROL_NSTEP=10`, i.e. 0.2 s of sim per action) so the
recorded arm motion is smooth and non-overshooting.

> Note: the original controller (`damping=4.0`, `nstep=30` = 0.6 s open-loop
> hold) was underdamped and made the arm ring permanently — every rendered
> frame showed the eef at a random point of the oscillation, which is the
> "trembling" seen in old videos. `tune_control.py` documents the fix.

## 2. Convert Data to LeRobot v2.1 / v3.0

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python convert_to_evo_lerobot.py --raw-dir .../raw_mujoco_panda7_multiview_fixed --out-dir .../MuJoCo_Panda7_Multiview_Clean --format both
```

- `--format v21` (default): LeRobot **v2.1** layout with a standards-compliant
  `meta/info.json` (`codebase_version: v2.1`, `robot_type: panda`, features
  schema). This is what the Evo-1 loader requires.
- `--format v30`: LeRobot **v3.0** consolidated layout (shared chunk parquet +
  concatenated videos + parquet meta), readable by the lerobot-main in
  `Evo-1/so100_evo1/lerobot-main`.
- `--format both`: write both.

The training config points at:

```text
Evo-1/Evo_1/dataset/config.yaml  ->  .../MuJoCo_Panda7_Multiview_Clean
```

> IMPORTANT: the LeRobot window cache is keyed by the config *name*, not the
> dataset path. When switching to a different dataset, delete
> `Mujoco_training_dataset/cache/mujoco_pickplace/mujoco_pickplace/MuJoCo_PickPlace_Dataset`
> before training, otherwise the loader reuses stale windows.

## 3. Check Evo-1 Dataset Loading

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

Expected shapes:

```text
images: [3, 3, 448, 448]
state: [24]
action: [50, 24]
```

The raw MuJoCo state/action are smaller, then padded by the Evo-1 loader to 24 dimensions.

## 4. Train with Evo-1

Run training from the Evo-1 directory (see `mujoco_pickplace/train_fixed.sh`):

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
bash train_fixed.sh          # or run the accelerate launch it wraps
```

`train_fixed.sh` exports `LD_PRELOAD=/home/user/miniconda3/envs/Evo1/lib/libstdc++.so.6`
(the system libstdc++ lacks `CXXABI_1.3.15` for flash_attn) and trains on the
clean dataset with `horizon=10`, `per_action_dim=24`, `max_steps=4000`.

## 5. Open-Loop Policy Test (before closed-loop eval)

Before wiring up the websocket loop, check whether the *model's own actions*
are smooth (isolates model jitter from control jitter):

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_open_loop.py --ckpt /home/user/mujoco+evo/ckpt/evo1_mujoco_panda7_multiview_h10_clean_v2/step_best --num-episodes 8
```

It feeds one observation, generates the action chunk, then executes the whole
chunk open-loop. The position jitter (`pos` metric) of the trained model is
~0.008-0.014 (the old under-trained model measured 0.05-0.13).

Result (step_best, 4000 steps on the clean data): closed-loop success **6/10
(0.60)** on the eval client vs 0/10 for the old model.

## 6. Start Evo-1 Inference Server

Terminal 1:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py --ckpt-dir /home/user/mujoco+evo/ckpt/<run>/step_best --port 9000
```

`--ckpt-dir` selects which checkpoint to serve.

## 7. Evaluate in MuJoCo (closed loop)

Terminal 2:

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python eval_policy_client.py --horizon 10 --save-video
```

The eval client executes `--horizon` actions of each received chunk, then
re-infers. Videos are rendered with `PickPlaceEnv.step_video`, which spreads the
0.2 s per action over several rendered frames, so playback is ~real-time and
smooth (the old videos advanced 0.6 s per frame at 30 fps = 6x and jumped).

```text
mujoco_pickplace/outputs/eval_videos/
```