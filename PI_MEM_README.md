# π-MEM K=6 短期记忆

本文件是仓库中唯一的 π-MEM 实现说明。论文 PDF 与逐页中文解读仅保存在本地 `paper/` 目录；该目录已由 `.gitignore` 排除，不随源码上传。

```text
MuJoCo 5 Hz 历史观测
-> K=6 图像、状态与历史掩码
-> InternVL 因果时空注意力
-> ViT 第 20 层后仅保留当前帧 token
-> 完整 14 层语言主干与 flow-matching 动作头
-> 预测 14 步，执行 4 步后重新规划
```

## 当前配置

- 记忆长度：`K=6`；历史间隔：`1.0 s`。
- 时间层：ViT 第 `4/8/12/16/20` 层。
- 上层压缩：第 20 层后丢弃过去帧 token；ViT 其余层继续运行。
- 语言主干：完整保留并运行 `14` 层，只关闭无用的 hidden-state 收集和 KV cache。
- 输入：`images [B,6,3,3,448,448]`、`state [B,6,24]`、`history_mask [B,6]`。
- 输出：`14 × 24` 动作；在线默认执行前 4 步后重新规划。
- 夹爪：0.4/0.6 滞回，连续 2 步确认后切换开合状态。
- 缓存：窗口索引版本 2；源 parquet、视频或关键 meta 改变时自动失效。
- 微调：训练入口默认启用 LoRA，`rank=8`、`alpha=16`，目标为时间 ViT 与动作头，并训练目标范围内的偏置、归一化和时间层缩放参数。
- 批处理：物理 batch 直接执行一次 batched ViT 和一次 batched 语言主干前向，不再把 batch 8 拆成 8 次 B=1 VLM。
- 优化器：单卡默认使用 CUDA fused AdamW + BF16，不启用 CPU optimizer offload；DeepSpeed ZeRO-2 仅保留为多卡可选路径。
- 旧 checkpoint 可作为 LoRA 初始化权重加载；要获得可靠的历史策略仍需使用 K=6 数据继续训练。

## 论文机制与实现位置

| π-MEM 短期机制 | 实现位置 |
|---|---|
| 每 4 个 ViT 层插入一次时间注意力 | `model/internvl3/temporal_vision_encoder.py` |
| 相同 patch 跨时间、因果下三角掩码 | `_causal_temporal_attention` |
| 复用原层 QKV、投影、归一化与 layer scale | `_space_time_layer` |
| 固定相对正弦时间编码，当前时刻为零 | `fixed_relative_temporal_encoding` |
| K=1 与原视觉路径严格一致 | `extract_temporal_feature` 单帧分支 |
| 第 20 层后删除历史视觉 token | `drop_past_after_layer` |
| K 个连续本体状态 token | `FlowmatchingActionHead._encode_state_history` |
| 左填充帧不参与视觉与动作注意力 | Dataset、时空编码器和 action head |

## 数据与模型流程

数据集按 episode 时间戳从早到晚选出 6 个观测；时间戳不可用时才退回固定步数。episode 开头不足 6 帧的槽位重复最早图像以维持固定 shape，但对应 `history_mask=False`。单样本推理会直接删除左填充帧；批量训练只删除整个 batch 共同无效的前缀，其余样本按各自 `history_mask` 屏蔽，样本之间不会共享时间注意力。状态 token 使用相同历史掩码。

每个有效相机、时刻先执行空间注意力；第 4/8/12/16/20 个 ViT 层再对同相机、同 patch 的历史执行因果时间注意力。第 20 层融合完成后只保留当前帧视觉 token，上层 ViT 继续处理当前帧。训练时 `[B,K,V,C,H,W]` 一次进入 ViT，之后所有 prompt 一次进入 14 层语言主干；batch 维和时间维始终独立。当前视觉语言 token 与 6 个状态 token 进入 flow-matching 动作头，产生 14 步动作。

时序样本对所有时刻使用同一组随机裁剪、旋转和颜色增强参数，避免增强过程制造不存在的运动。默认保留原始 5 Hz 控制时间轴；只有显式添加 `--compact-static-frames` 才压缩静止帧。

## 仓库结构

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py  时空视觉编码
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py       K 帧预处理与语言融合
Evo-1/Evo_1/model/action_head/flow_matching.py          状态记忆与动作头
Evo-1/Evo_1/model/lora.py                               LoRA 线性层、MHA 与推理合并
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py      K=6 数据窗口
Evo-1/Evo_1/scripts/train.py                            训练入口
Evo-1/Evo_1/scripts/Evo1_server.py                      推理服务
mujoco_pickplace/collect_data.py                        默认覆盖、显式追加的数据采集
mujoco_pickplace/eval_policy_client.py                  在线闭环客户端
```

## 环境状态

当前 `Evo1` 环境已经具备：

| 组件 | 已验证状态 |
|---|---|
| Python / PyTorch | Python 3.10.20；PyTorch 2.12.0.dev + CUDA 12.8 |
| C++ / Ninja | Conda GCC/G++ 14.3；Ninja 1.13 |
| 优化器 | 单卡 CUDA fused AdamW；DeepSpeed 0.19.2 仅作多卡可选 |
| CUDA 开发库 | `libcurand-dev 10.3.9.90`、`libcufile-dev 1.13.1.3` |
| 异步 I/O | `libaio 0.3.113`，多卡 DeepSpeed 兼容性检查通过 |
| FlashAttention | 2.8.3.post1；真实 InternVL3-1B、K=6、batch 8 训练与推理均确认使用快路径 |

当前只训练约 1.49M 参数，Adam 一、二阶状态仅为十几 MB 量级。单张 8 GB GPU 使用 ZeRO-2 没有第二张卡可分片，反而增加包装器和通信缓冲；真实生产入口测试中曾在反向传播触发 OOM。当前单卡正式路径改用 CUDA fused AdamW，并通过分段 checkpoint 时间注意力与 ViT MLP、以 BF16 执行 LoRA 矩阵乘法来压低峰值；FP32 LoRA 主参数、梯度和 optimizer state 仍保留。`ds_config_pi_mem.json` 不删除，供以后多卡训练使用，其中也不启用 CPU optimizer offload。FlashAttention 的 PyTorch 回退只用于明确报错和环境诊断，不能当作正式训练快路径。正式训练必须先 `conda activate Evo1`，否则可能加载系统旧版 `libstdc++` 并触发 ABI 回退。

## 1. 安全采集数据

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

只有需要保留现有数据并继续编号时，才显式添加：

```bash
python collect_data.py --append
```

`--overwrite` 可用于显式表达默认行为，并与 `--append` 互斥。追加模式下，写入器会拒绝覆盖已经存在的 episode 文件。若要在不影响当前数据集的情况下生成一套新数据，也可以使用 `--dataset-dir`，并同步修改 `Evo-1/Evo_1/dataset/config.yaml`。

## 2. 检查数据

```bash
conda activate Evo1
cd /home/user/mujoco+evo
python mujoco_pickplace/check_dataset.py --require-evo
```

预期关键形状：

```text
images: [6, 3, 3, 448, 448]
state: [6, 24]
history_mask: [6]
action: [14, 24]
```

## 3. 单卡继续训练

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend no --mixed_precision bf16 scripts/train.py \
  --run_name evo1_pi_mem --action_head flowmatching --use_flash_attn \
  --dataset_config_path dataset/config.yaml --vlm_name OpenGVLab/InternVL3-1B \
  --use_augmentation --image_size 448 --batch_size 8 --lr 1e-5 --dropout 0.1 --weight_decay 1e-3 --fused_adamw \
  --max_steps 16000 --warmup_steps 1000 --log_interval 50 --ckpt_interval 2000 --grad_clip_norm 1.0 \
  --num_layers 8 --num_workers 4 --horizon 14 --per_action_dim 24 --state_dim 24 --use_state \
  --memory_frames 6 --memory_stride_seconds 1.0 --memory_stride_steps 5 --temporal_layer_interval 4 \
  --temporal_drop_past_after_layer 20 \
  --gradient_checkpointing --compact_masked_views --finetune_temporal_vision --finetune_action_head \
  --use_lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --lora_targets vision,action \
  --lora_train_bias_norm --disable_wandb --disable_swanlab --resume --resume_pretrain \
  --resume_path /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2
```

启动日志应明确出现以下两行；若 attention backend 不是这两个值，不应把该次运行视为正式快路径：

```text
Attention backend: vision=flash-attn, language=flash_attention_2
VLM forward mode=batched: one vision/language forward per physical batch
Optimizer=AdamW, fused=True
Prepared optimizer=AcceleratedOptimizer; base optimizer=AdamW; DeepSpeed ZeRO stage=None
```

### 路径与恢复方式

LoRA 默认已经开启；命令仍显式写出参数以便审计。当前训练 `1.49M` 参数，其中动作头约 `1.21M`、五个时间 ViT 层约 `0.28M`，语言主干冻结。LoRA-B 从零开始，首次前向与基础模型一致。只有明确需要恢复旧的非 LoRA 训练方式时才添加 `--no-use_lora`。

`lora_targets=vision,action` 不微调语言主干。如果以后确实需要语言适配，可显式改为 `vision,language,action`，但会增加反向计算和显存。`--fp32_temporal_parameters` 只服务 `--no-use_lora` 的原权重微调路径；LoRA 参数自身已保持 FP32，所以默认 LoRA 命令不再需要该参数。

如果是 π-MEM 训练意外中断后继续，保留 `--resume` 和 `--resume_path /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2/step_8000`，删除 `--resume_pretrain`，并继续使用同一个 `--save_dir`。此时 `--max_steps` 表示最终全局步数，而不是再训练多少步。


## 4. 推理与评估

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py \
  --memory-frames 6 --memory-stride-steps 5 --horizon 4 \
  --gripper-debounce-steps 2
```

客户端用 base64 JPEG 发送 `[K,V]` 图像并只发送真实相机；服务端仍兼容旧单帧 payload。K=1 严格走原单帧视觉路径。K>1 使用旧 checkpoint 只能证明结构可运行，不能代替 K=6 历史数据训练。
