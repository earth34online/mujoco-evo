# MuJoCo Pick-and-Place Client

This folder is the MuJoCo side of the project. It plays the same role as Evo-1's benchmark clients such as `LIBERO_evaluation/libero_client_4tasks.py`: create the simulation environment, render observations, send them to the Evo-1 server, receive action chunks, execute actions, and report rollout results.

## 1. Collect Data

`collect_data.py` uses the stateful, contact-aware scripted expert. This
expert is a teacher that generates demonstrations; it is not a visualization
of an already-trained Evo-1 checkpoint.


```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

New episodes use the project-owned `mujoco-evo-episodes/1.0` format. It is
inspired by LeRobot v2.1/v3 organization and LIBERO episode semantics, but is
not labeled as an official LeRobot dataset:

```text
../Mujoco_training_dataset/MuJoCo_Panda7_Multiview_V2/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.front/episode_000000.mp4
  videos/chunk-000/observation.images.overhead/episode_000000.mp4
  videos/chunk-000/observation.images.wrist/episode_000000.mp4
  meta/dataset.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
```

Only successful two-pad-contact episodes passing smoothness gates are saved.
Writes are atomic and existing episodes are appended, never overwritten. One
action spans 0.2 seconds, so per-action videos and rows are both 5 FPS.

## 2. Legacy NPZ Conversion

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python convert_to_evo_lerobot.py
```

The converter below remains for existing NPZ archives and legacy Evo-1
experiments. The old collector is preserved as `collect_data_legacy_npz.py`.
Neither is the default path for new demonstrations.

```text
/home/user/mujoco+evo/Mujoco_training_dataset/MuJoCo_Panda7_Multiview_Small
├── data/chunk-000/*.parquet
├── videos/chunk-000/observation.images.{front,overhead,wrist}/*.mp4
└── meta/*.jsonl, *.json
```

## 3. Check Dataset Loading

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

Expected output:

```text
dataset length: ...
images (3, 3, 448, 448) float32
state (24,) float32
action (50, 24) float32
```

> Note: Evo-1 loader pads the 10-dim state and 4-dim action to 24 dimensions with zeros.

## 4. Train with Evo-1

The Evo-1 source tree is kept unchanged. Training uses its original entry point and is not started by the cleanup/check workflow:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/train.py --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_panda7_multiview_h10_clean_v2
```

## 5. Start Evo-1 Inference Server

**Terminal 1:**

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

The unchanged `scripts/Evo1_server.py` reads its checkpoint path internally:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_panda7_multiview_h10_clean_v2/step_best
```

## 6. Evaluate in MuJoCo

**Terminal 2:**

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python eval_policy_client.py
```

Full CLI flags:

```text
--server-url    WebSocket server URL
--num-episodes  Number of evaluation episodes (default: 10)
--max-steps     Max environment steps per episode (default: 100)
--horizon       Actions to unroll per server call (default: 3)
--render        Show real-time OpenCV window (requires GUI)
--save-video    Save each rollout as MP4 (default)
--video-dir     Output directory for videos (default: outputs/eval_videos/<timestamp>)
```

## Environment Summary

| Requirement | Conda Env |
|---|---|
| Collect / Convert / Evaluate | `mujoco` |
| Check dataset / Train / Server | `Evo1` |
