# MuJoCo + Evo-1 Pick-and-Place Demo

```text
MuJoCo pick-and-place task
-> scripted expert data collection
-> Evo-1/LeRobot-style dataset conversion
-> Evo-1 DeepSpeed training
-> Evo-1 websocket inference server
-> MuJoCo rollout evaluation
```

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
python collect_data.py
```

> Note: the original controller (`damping=4.0`, `nstep=30` = 0.6 s open-loop
> hold) was underdamped and made the arm ring permanently — every rendered
> frame showed the eef at a random point of the oscillation, which is the
> "trembling" seen in old videos. The historical tuning script has been removed; the current controller is in `pick_place_env.py`.

## 2. Convert Data to LeRobot v2.1 / v3.0

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python convert_to_evo_lerobot.py
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
Evo-1/Evo_1/dataset/config.yaml  ->  .../MuJoCo_Panda7_Multiview_V2
```

> IMPORTANT: the LeRobot window cache is keyed by the config *name*, not the
> dataset path. When switching to a different dataset, delete
> `Mujoco_training_dataset/cache/mujoco_pickplace/mujoco_pickplace/MuJoCo_PickPlace_Dataset`
> before training, otherwise the loader reuses stale windows.

## 3. Check Evo-1 Dataset Loading

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python check_dataset.py
```

Expected shapes:

```text
images: [3, 3, 448, 448]
state: [24]
action: [50, 24]
```

## 4. Train with Evo-1

Use Evo-1 original `train.py`; the MuJoCo wrapper scripts were removed and the Evo-1 tree is unchanged:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/train.py
```


## 5. Open-Loop Policy Test (before closed-loop eval) (choice)

Before wiring up the websocket loop, check whether the *model's own actions*
are smooth (isolates model jitter from control jitter):

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_open_loop.py
```

## 6. Start Evo-1 Inference Server

Terminal 1:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

## 7. Evaluate in MuJoCo (closed loop)

Terminal 2:

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python eval_policy_client.py
```
