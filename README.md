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
python collect_data.py
```

Current defaults:

```text
NUM_EPISODES = 100
MAX_STEPS = 300
```

The script uses `scripted_expert()` in `pick_place_env.py` to collect successful demonstration-style trajectories.

## 2. Convert Data to Evo-1 Format

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python convert_to_evo_lerobot.py
```

Output path:

```text
/home/user/mujoco+evo/Mujoco_training_dataset/MuJoCo_PickPlace_Dataset
```

This path is referenced by:

```text
Evo-1/Evo_1/dataset/config.yaml
```

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

Run training from the Evo-1 directory, following Evo-1's original README style:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_mujoco_pickplace_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

Stage 2 can follow Evo-1's README pattern, usually enabling `--finetune_vlm` and training for many more steps after Stage 1.

## 5. Start Evo-1 Inference Server

Terminal 1:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

Current `scripts/Evo1_server.py` uses this checkpoint path internally:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_final
```

## 6. Evaluate in MuJoCo

Terminal 2:

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python eval_policy_client.py
```

Optional realtime visualization:

```bash
python eval_policy_client.py
```

```text
mujoco_pickplace/outputs/eval_videos/
```