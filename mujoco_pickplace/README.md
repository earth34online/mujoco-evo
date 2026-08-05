# MuJoCo Pick-and-Place Client

This folder is the MuJoCo side of the project. It plays the same role as Evo-1's benchmark clients: create the environment, render observations, send them to the Evo-1 server, receive action chunks, execute actions, and report rollout results.

## 1. Collect Data

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

Only successful two-pad-contact episodes passing smoothness gates are saved.
Writes are atomic and existing episodes are appended, never overwritten.

## 2. Check Dataset Loading

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

## 3. Train with Evo-1

The Evo-1 source tree is kept unchanged. Training uses its original entry point and is not started by the cleanup/check workflow:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/train.py
```

## 4. Start Evo-1 Inference Server

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
