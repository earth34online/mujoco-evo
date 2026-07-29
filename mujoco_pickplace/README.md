# MuJoCo Pick-and-Place Client

This folder is the MuJoCo side of the project. It plays the same role as Evo-1's benchmark clients such as `LIBERO_evaluation/libero_client_4tasks.py`: create the simulation environment, render observations, send them to the Evo-1 server, receive action chunks, execute actions, and report rollout results.

## 1. Collect Data

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

Defaults are `NUM_EPISODES = 100` and `MAX_STEPS = 300` per episode. Output:

```text
../Mujoco_training_dataset/raw_mujoco_pickplace/episode_000000.npz
../Mujoco_training_dataset/raw_mujoco_pickplace/episode_000001.npz
```

## 2. Convert Data

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python convert_to_evo_lerobot.py
```

Converted dataset path (referenced by `Evo-1/Evo_1/dataset/config.yaml`):

```text
/home/user/mujoco+evo/Mujoco_training_dataset/MuJoCo_PickPlace_Dataset
├── data/chunk-000/*.parquet
├── videos/chunk-000/observation.images.image/*.mp4
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

## 4. Start Evo-1 Inference Server

**Terminal 1:**

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

The current `scripts/Evo1_server.py` reads its checkpoint path internally. Make sure it points to:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_final
```

## 5. Evaluate in MuJoCo

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
--max-steps     Max environment steps per episode (default: 300)
--horizon       Actions to unroll per server call (default: 15, must be ≤ training horizon 50)
--render        Show real-time OpenCV window (requires GUI)
--save-video    Save each rollout as MP4 (default)
--video-dir     Output directory for videos (default: outputs/eval_videos)
```

## Environment Summary

| Requirement | Conda Env |
|---|---|
| Collect / Convert / Evaluate | `mujoco` |
| Check dataset / Train / Server | `Evo1` |