#!/usr/bin/env bash
# train_fixed.sh
#
# Retrain Evo-1 on the clean (critically-damped control) pick-and-place data.
# Run from inside WSL (Evo1 conda env). The previous model was trained on data
# collected with the underdamped controller (nstep=30), so its actions jittered.
#
# IMPORTANT: clear the LeRobot window cache before training, otherwise the
# loader reuses windows cached from the old dataset (they share the config-keyed
# cache subdir):
#   rm -rf /home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/mujoco_pickplace/MuJoCo_PickPlace_Dataset
#
set -e
export PATH=/home/user/miniconda3/envs/Evo1/bin:$PATH
cd /home/user/mujoco+evo/Evo-1/Evo_1

# flash_attn needs the conda libstdc++ (system one lacks CXXABI_1.3.15).
export LD_PRELOAD=/home/user/miniconda3/envs/Evo1/lib/libstdc++.so.6

accelerate launch --num_processes 1 --num_machines 1 --use_deepspeed --deepspeed_config_file ds_config.json \
    scripts/train.py \
    --run_name Evo1_mujoco_panda7_h10_clean_v2 \
    --action_head flowmatching \
    --use_augmentation \
    --lr 3e-5 \
    --batch_size 16 \
    --image_size 448 \
    --max_steps 4000 \
    --log_interval 20 \
    --ckpt_interval 1000 \
    --warmup_steps 200 \
    --grad_clip_norm 1.0 \
    --num_layers 8 \
    --horizon 10 \
    --finetune_action_head \
    --disable_wandb \
    --vlm_name OpenGVLab/InternVL3-1B \
    --dataset_config_path dataset/config.yaml \
    --per_action_dim 24 \
    --state_dim 24 \
    --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_panda7_multiview_h10_clean_v2
