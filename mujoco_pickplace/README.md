# MuJoCo 拾取放置客户端

本文件是原始 `mujoco_pickplace/README.md` 的中文对应版本；原文件保持不变。π-MEM 的 K=6 用法、训练与评估方式见上级目录的 [PI_MEM_README.md](../PI_MEM_README.md)。

本目录是项目的 MuJoCo 端，负责创建环境、采集示范数据、检查 Evo-1 数据加载、向 Evo-1 服务端发送观测并渲染评估回放。

## 当前标准状态

- 数据集根目录：`/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace`
- `check_dataset.py` 通过 Evo-1 loader 验证数据集。
- `eval_policy_client.py` 是闭环评估客户端，并保存 MP4 视频。

## 1. 采集数据

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

只有通过质量门槛、两个夹爪接触垫都成功接触的 episode 才会保存。
采集默认清除目标数据集并从 episode 0 重写；只有需要保留已有数据并继续编号时才显式添加 `--append`。

## 2. 检查数据加载

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py
```

原始单帧工程预期输出：

```text
dataset length: ...
images torch.Size([3, 3, 224, 224])
state torch.Size([24])
action torch.Size([14, 24])
state_mask sum: 8
action_mask sum: 56
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
  --log_interval 20 --ckpt_interval 2500 --warmup_steps 3000 --grad_clip_norm 1.0 \
  --num_layers 8 --horizon 14 --finetune_action_head --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 --state_dim 24 \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1_random
```

## 4. 启动 Evo-1 推理服务

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

服务端默认 checkpoint：

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best
```

## 5. 在 MuJoCo 中评估

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```

常用参数：

```text
--server-url        服务端地址
--num-episodes      episode 数量
--max-steps         最大控制步数
--horizon           每次执行的动作视野
--render            显示渲染窗口
--save-video        保存视频
--video-dir         视频目录
```
