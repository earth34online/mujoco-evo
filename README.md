# MuJoCo + Evo-1 拾取放置演示

π-MEM K=6 短期记忆版本的架构、环境、训练与评估方式见 [PI_MEM_README.md](PI_MEM_README.md)。

```text
MuJoCo 拾取放置任务
-> 脚本专家数据采集
-> cache/ 中的 LeRobot 风格 parquet、mp4 与 meta 数据集
-> Evo-1 DeepSpeed 训练
-> Evo-1 WebSocket 推理服务
-> MuJoCo 回放评估
```

## 当前标准状态

- 数据集根目录：`/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace`
- 训练通过 `accelerate launch` 启动 `Evo-1/Evo_1/scripts/train.py`。
- 训练入口默认启用 LoRA；π-MEM 正式参数和关闭 LoRA 的兼容方式见 `PI_MEM_README.md`。
- 服务端默认 checkpoint：`/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best`
- 评估客户端把视频保存到 `mujoco_pickplace/outputs/eval_videos/<时间戳>/task1/`。

## 仓库结构

```text
mujoco_pickplace/                  MuJoCo 任务、数据采集、数据检查与评估客户端
Evo-1/Evo_1/                       训练和服务端推理所需的最小 Evo-1 文件
Mujoco_training_dataset/cache/     当前标准数据集位置
ckpt/                              训练 checkpoint 输出
```

## 1. 采集 MuJoCo 示范

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

采集默认删除目标数据集后从 episode 0 重写。只有需要保留现有 episode 并继续编号时才显式使用 `python collect_data.py --append`。

数据会直接写成 LeRobot 风格 episode，保存于：

```text
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace
```

旧的 npz 采集与转换脚本已经移除。

## 2. 检查 Evo-1 数据加载

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

原始单帧工程预期形状：

```text
images: [3, 3, 448, 448]
state: [24]
action: [14, 24]
state mask sum: 8
action mask sum: 56
```

## 3. 使用 Evo-1 训练

先按照上游 Evo-1 方式配置一次 Accelerate/DeepSpeed：

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate config
```

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend no --use_deepspeed --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.1 \
  --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 16000 \
  --log_interval 50 --ckpt_interval 2500 --warmup_steps 3000 --grad_clip_norm 1.0 \
  --num_layers 8 --horizon 14 --finetune_action_head --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 --state_dim 24 --use_state \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1_random
```

默认 checkpoint 根目录：

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

## 4. 启动 Evo-1 推理服务

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

## 5. 在 MuJoCo 中评估

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```
