# MuJoCo Pick-and-Place Client

This folder is the MuJoCo side of the project. It creates the environment, collects demonstration data, checks Evo-1 dataset loading, sends observations to the Evo-1 server, and renders evaluation rollouts.

## Current Canonical State

- Dataset root: `/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace`
- `check_dataset.py` validates the dataset through the Evo-1 loader.
- `eval_policy_client.py` is the closed-loop evaluator and saves MP4 videos.

## 1. Collect Data

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

Only successful two-pad-contact episodes passing the quality gates are saved.

## 2. Check Dataset Loading

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

Expected output:

```text
dataset length: ...
images torch.Size([3, 3, 224, 224])
state torch.Size([24])
action torch.Size([14, 24])
state_mask sum: 8
action_mask sum: 56
```

## 3. Train with Evo-1

Configure Accelerate/DeepSpeed once, following the upstream Evo-1 setup:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate config
```

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.1 \
  --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 8000 \
  --log_interval 20 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 \
  --num_layers 8 --horizon 14 --finetune_action_head --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 --state_dim 24 --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

## 4. Start Evo-1 Inference Server

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

The server default checkpoint is:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best
```

## 5. Evaluate in MuJoCo

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```

Common flags:

```text
--server-url
--num-episodes
--max-steps
--horizon
--render
--save-video
--video-dir
```
