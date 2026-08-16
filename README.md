# MuJoCo + Evo-1 Pick-and-Place Demo

```text
MuJoCo pick-and-place task
-> scripted expert data collection
-> LeRobot-style parquet/mp4/meta dataset in cache/
-> Evo-1 DeepSpeed training
-> Evo-1 websocket inference server
-> MuJoCo rollout evaluation
```

## Current Canonical State

- Dataset root: `/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace`
- No `npz` intermediate pipeline remains.
- `collect_data.py` writes the final dataset directly.
- Training uses `accelerate launch` with `Evo-1/Evo_1/scripts/train.py`.
- Server default checkpoint: `/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best`
- Evaluation client saves videos under `mujoco_pickplace/outputs/eval_videos/<timestamp>/task1/`

## Repository Layout

```text
mujoco_pickplace/                 MuJoCo task, data collection, dataset check, evaluation client
Evo-1/Evo_1/                      Minimal Evo-1 files needed for training/server inference
Mujoco_training_dataset/cache/     Current canonical dataset location
ckpt/                              Training checkpoint outputs
```

## 1. Collect MuJoCo Demonstrations

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

This writes direct LeRobot-like episodes to:

```text
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace
```

The old npz collection and conversion scripts have been removed.

## 2. Check Evo-1 Dataset Loading

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

Expected shapes:

```text
images: [3, 3, 224, 224]
state: [24]
action: [14, 24]
state mask sum: 8
action mask sum: 56
```

## 3. Train with Evo-1

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.1 \
  --weight_decay 1e-3 --batch_size 16 --image_size 224 --max_steps 5000 \
  --log_interval 20 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 \
  --num_layers 8 --horizon 14 --finetune_vlm --finetune_action_head --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 --state_dim 24 --use_state \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

Default checkpoint root:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

## 4. Start Evo-1 Inference Server

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

## 5. Evaluate in MuJoCo

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```
