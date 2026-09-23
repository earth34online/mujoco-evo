# π-MEM K=6 短期记忆

```text
MuJoCo 5 Hz 历史观测
-> K=6 图像、状态与历史掩码
-> InternVL 因果时间 value -> 空间注意力的组合结构
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
- 输出：`14 × 24` 动作；自由空间和抬升/搬运阶段默认执行前 4 步，接近抓取平面或即将由开变闭时每步重新规划。控制器不再把指关节位置误当成“已经抓稳”的确认信号。
- 夹爪：0.4/0.6 滞回；闭合立即执行以保留 `success_random` 的夹取时机，只有开爪/释放需要连续 2 步强信号，防止搬运途中单帧误开爪。
- 当前任务几何：方块边长 `50 mm`、静止中心高度 `55 mm`；抓取 Z 容差没有放宽。50 mm 抓取平面下专家与 Stage2 运行时使用经随机回归验证的 `4 mm` X 几何补偿，旧 Stage1 checkpoint 仍保留原 `6 mm` 契约。
- 缓存：窗口索引版本 3；除源 parquet、视频和关键 meta 指纹外，还保存每个窗口的短期记忆事件标签。
- 微调：训练入口默认启用 LoRA，`rank=8`、`alpha=16`。动作头继续使用普通 LoRA；五个时间 ViT 层只在时间匹配/value 融合和组合记忆输出上使用独立低秩残差，空间 Q/K、视觉偏置、归一化和 layer scale 保持 Stage1 冻结值。
- 批处理：物理 batch 直接执行一次 batched ViT 和一次 batched 语言主干前向，不再把 batch 8 拆成 8 次 B=1 VLM。
- 优化器：正式单卡 LoRA 直接用一个 Python 进程启动 Accelerate + CUDA fused AdamW + BF16；DeepSpeed ZeRO-2 仅保留为多卡可选路径。worker 内的自动回退只能关闭 ZeRO，不能撤销外层 launcher 已建立的 NCCL 进程组。

## 数据与模型流程

数据集按 episode 时间戳从早到晚选出 6 个观测；时间戳不可用时才退回固定步数。episode 开头不足 6 帧的槽位重复最早图像以维持固定 shape，但对应 `history_mask=False`。单样本推理会直接删除左填充帧；批量训练只删除整个 batch 共同无效的前缀，其余样本按各自 `history_mask` 屏蔽，样本之间不会共享时间注意力。状态 token 使用相同历史掩码。

时序样本对所有时刻使用同一组增强参数，避免增强过程制造不存在的运动。MuJoCo 固定相机的笛卡尔动作标签与像素几何绑定，因此 `dataset/config.yaml` 默认设置 `preserve_spatial_calibration: true`：保留同步颜色增强，但禁用未同步变换动作标签的随机裁剪和旋转。默认保留原始 5 Hz 控制时间轴；只有显式添加 `--compact-static-frames` 才压缩静止帧。

数据窗口会按模型真正收到的 6 个历史索引分类，只有所采样历史里实际包含 `recover`，且当前仍处在恢复后的 `recover/approach/descend/close`，才标为 `post_failure_correction`；采样间隔中出现但没有进入模型输入的恢复帧不会误标。训练默认按各事件在现有数据中的实际频次做逆平方根加权，不规定恢复 episode 数量或固定失败比例，也不要求为了凑比例而延长采集。

## 仓库结构

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py  时空视觉编码
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py       K 帧预处理与语言融合
Evo-1/Evo_1/model/action_head/flow_matching.py          状态记忆与动作头
Evo-1/Evo_1/model/lora.py                               普通 LoRA、时间路径独立 LoRA 与推理合并
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
| FlashAttention | 2.8.3.post1；新组合结构已用真实 InternVL3-1B、K=6、batch 1 完成训练单步与评估快路径验证 |

Adam 一、二阶状态仅为十几 MB 量级，单张 8 GB GPU 使用 ZeRO-2 没有第二张卡可分片，反而增加包装器和通信缓冲。当前单卡正式路径用 CUDA fused AdamW，并通过分段 checkpoint 时间注意力与 ViT MLP、以 BF16 执行 LoRA 矩阵乘法来压低峰值；FP32 LoRA 主参数、梯度和 optimizer state 仍保留。正式命令直接运行 `python scripts/train.py`，从源头避免 `accelerate launch -> deepspeed_launcher -> torch.distributed/NCCL`。训练脚本在 `Accelerator` 初始化前关闭单卡 LoRA 的 DeepSpeed wrapping 仍作为旧命令兼容保护，但 worker 已经无法撤销外层 launcher 创建的单 rank 进程组，因此不能以日志中的 `ZeRO stage=None` 单独证明 launcher 已移除。`ds_config_pi_mem.json` 不删除，供以后多卡训练使用，其中也不启用 CPU optimizer offload。FlashAttention 的 PyTorch 回退只用于明确报错和环境诊断，不能当作正式训练快路径。

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

方块从 `60 mm` 改成 `50 mm` 后不能对旧数据使用 `--append`。新采集元数据会记录 `cube_side_m=0.050`、`cube_support_z_m=0.055` 和 `expert_grasp_x_bias_m=0.004`；写入器会拒绝把不同几何混入同一数据集，严格数据检查和正式数据加载也会拒绝旧 60 mm 数据。应使用默认覆盖模式重新采集，使图像尺寸、下探平面和动作标签属于同一个任务分布。

## 2. 检查数据

```bash
conda activate Evo1
cd /home/user/mujoco+evo
python mujoco_pickplace/check_dataset.py --require-evo --require-strict-grasp-quality
```

`--require-evo` 禁止在缺少 Evo 数据接口依赖时把 K=6 shape 检查静默跳过；`--require-strict-grasp-quality` 要求 `stable-grasp-v3` 的闭合位置、双指接触、接触时方块位移/倾斜和抓持持续证据。这两个开关只用于训练前严格验收专家数据，不进入训练或评估策略。

预期关键形状：

```text
images: [6, 3, 3, 448, 448]
state: [6, 24]
history_mask: [6]
action: [14, 24]
```

## 3. 训练

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=bf16
python scripts/train.py \
  --run_name evo1_pi_mem --action_head flowmatching --use_flash_attn \
  --dataset_config_path dataset/config.yaml --vlm_name OpenGVLab/InternVL3-1B \
  --use_augmentation --image_size 448 --batch_size 4 --lr 1e-5 --dropout 0.1 --weight_decay 1e-3 --fused_adamw \
  --max_steps 32000 --warmup_steps 1500 --log_interval 20 --ckpt_interval 2000 --grad_clip_norm 1.0 \
  --num_layers 8 --num_workers 4 --horizon 14 --per_action_dim 24 --state_dim 24 --use_state \
  --memory_frames 6 --memory_stride_seconds 1.0 --memory_stride_steps 5 --temporal_layer_interval 4 \
  --temporal_drop_past_after_layer 20 \
  --gradient_checkpointing --compact_masked_views --finetune_action_head \
  --use_lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --lora_targets vision,action \
  --lora_train_bias_norm --disable_wandb --disable_swanlab --resume --resume_pretrain \
  --resume_path /home/user/mujoco+evo/ckpt/archive/evo1_mujoco_pickplace_stage1_random_stepbest \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2
```

数据路径只由 `dataset/config.yaml` 决定；训练从已验收的随机任务阶段一 `step_best` 归档初始化，并写入阶段二目录。

```text
Attention backend: vision=flash-attn, language=flash_attention_2
VLM forward mode=batched: one vision/language forward per physical batch
Accelerate runtime: distributed_type=DistributedType.NO, num_processes=1, ... torch_distributed_initialized=False, LOCAL_RANK=None, WORLD_SIZE=None
Optimizer=AdamW, fused=True
Prepared optimizer=AcceleratedOptimizer; base optimizer=AdamW; DeepSpeed ZeRO stage=None
```

`step_best` 沿用 MINT 原始判定：从全局 step 0 起用单 batch loss 更新内存中的 `best_loss`，但只有 `step > max(1000, warmup_steps)` 且再次刷新全局最低 loss 时才写入。该逻辑按要求保持不变；它代表训练 loss 最低点，不保证机器人成功率最高，因此正式验收仍需横向比较 `step_28000` 等周期 checkpoint、`step_best` 和 `step_final` 的同 seed 行为结果。

### 路径与恢复方式

LoRA 默认已经开启；命令仍显式写出 rank、alpha 和目标范围以便审计。当前 InternVL3-1B 的五个时间层新增 `245,760` 个 temporal-only LoRA 参数，完整模型实测总可训练参数为 `1,458,640`，共享视觉参数可训练数为 `0`；动作头继续使用 LoRA 及其少量偏置/归一化支持参数，语言主干冻结。LoRA-B 从零开始，因此初始化时 temporal 增量为零。只有明确需要恢复旧的非 LoRA 训练方式时才添加 `--no-use_lora`。

`lora_targets=vision,action` 不微调语言主干。`separate_temporal_lora` 默认开启；它不是复制整套 ViT attention，而是给时间匹配/value 融合和组合记忆输出增加低秩残差。空间 Q/K 不读取这些残差。`--finetune_temporal_vision` 和 `--fp32_temporal_parameters` 只影响 `--no-use_lora` 的原权重微调路径，因此正式 LoRA 命令不写这两个无效参数。如果以后确实需要语言适配，可显式改为 `vision,language,action`，但会增加反向计算和显存。

当前命令里的 `--resume --resume_pretrain --resume_path /home/user/mujoco+evo/ckpt/archive/evo1_mujoco_pickplace_stage1_random_stepbest` 表示：只从已验收的随机任务 Stage1 `step_best` 载入模型权重，新的 temporal/action LoRA 从零增量开始；不继承 Stage1 的 optimizer、scheduler、global step 或 best loss。这正是新结构首次训练应使用的初始化方式，不应删除。

已有的 Stage2 checkpoint 不能代替这个初始化：其冻结 action-base 与 `stage1_random_stepbest` 不同。新一轮必须按上述 `resume_path` 开始；只有在新一轮已经产生带 `memory_frames=6`、`pi_mem_attention_mode=composed`、`separate_temporal_lora=true` 的 checkpoint 后，中断续训才把 `resume_path` 改为该新 checkpoint 并去掉 `--resume_pretrain`。

### 小规模真实模型调起验证

已从 `/home/user/mujoco+evo/ckpt/archive/evo1_mujoco_pickplace_stage1_random_stepbest` 严格载入全部基础权重，只允许新 LoRA 键缺失，并完成一次 K=6、batch 1 的完整训练损失前向、反向与 AdamW 更新，再完成一次 `predict_action` 评估：

```text
vision forward calls=1, language forward calls=1
loss=1.67512763
temporal layers 4/8/12/16/20: QKV LoRA gradients finite and non-zero
action-head LoRA gradient: finite and non-zero
optimizer update: temporal LoRA max delta=9.99998974e-06
evaluation output=[1,14,24], finite=true
CUDA peak allocated=2467.6 MiB, peak reserved=2548.0 MiB
```

这只证明训练和评估入口、权重恢复、梯度链路与输出 shape 可以正常工作，不代表完整训练的显存峰值，也不代替新 checkpoint 的 100 episode 成功率验收。

兼容性方面，现有 `stage2/step_best` 已用当前服务端完成严格 state-dict 加载；它按自身配置选择 `composed` 和 temporal-only LoRA，32 个 LoRA 模块可正常合并，K=6 推理输出为 `[14,24]` 且全部有限。这只证明旧 Stage2 文件完整且能启动，不改变其初始 action-base 选错的事实，因此不应将它作为新一轮续训起点。


## 4. 推理与评估

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py \
  --ckpt-dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2/step_best
```

不写 `--ckpt-dir` 时也默认使用上述阶段二 `step_best`。从 ModelScope 云端下载的 checkpoint 可能在 `config.json` 中保留 `/mnt/workspace/modelscope_cache/OpenGVLab/InternVL3-1B`；该路径在本机不存在时，服务端会自动改用可移植模型 ID `OpenGVLab/InternVL3-1B`。若模型位于其他目录，显式添加 `--vlm-name /实际模型目录`。

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```

客户端正式命令不需要重复写默认参数：`num_episodes=100`、`max_steps=250`、`memory_frames=6`、`memory_stride_steps=5`；不传 `--start-seed` 时每次运行随机生成 seed。图像通过无损 base64 PNG 发送，避免 JPEG 改变像素边界后累积成夹取偏差。

连接建立后，客户端会先读取 checkpoint 协议，而不再用一套评估参数强行解释所有权重：

- 旧 Stage1（配置中没有 `memory_frames`）：服务端只取当前帧，恢复 1 个真实视角 + 2 个黑色占位视角、固定 1024-token 上下文、Stage1 action-head 的原始无 padding-mask 交叉注意力和 32 步 flow solver；客户端自动使用 `horizon=14`、不逐步重规划、`6 mm` 运行时抓取几何。
- π-MEM Stage2：保留 K=6 历史、紧凑视角、带 mask 的 action context 和 50 步 flow solver；客户端自动使用 `horizon=4`、抓取平面逐步重规划、当前 50 mm 环境的 `4 mm` 抓取几何。

命令行显式传入 `--horizon`、`--precision-replan` 或 `--no-precision-replan` 仍可覆盖 checkpoint 建议。禁用 precision replan 时会完整执行指定 horizon，不再暗中因夹爪跳变截断动作块。π-MEM 默认启用时，仍会在抓取平面每步重规划，并在夹爪跳变后用新观测重规划。

回归证据：对原先失败的 seed `79341034`，同一份 `stage1_random_stepbest` 在恢复上述协议后于第 100 步成功；首次 attachment 与真实双指接触同在第 36 步，运输期间未掉落，闭爪未确认发愣连续步数为 0。这是定向回归，不冒充 100-episode 成功率。

### 50 mm 方块与现有 checkpoint

现有 `step_28000`、`step_best` 和 `step_final` 都由 2026-09-17 采集的旧 60 mm 图像数据训练，不能因为代码能加载就视为已经适配 50 mm。一次不更新权重的可行性诊断中，`step_28000` 能按 K=6/4-step/precision-replan 协议正常启动，但在随机 seed `4009580131` 的 50 mm 场景里首次闭爪时没有双指接触或 attachment，夹爪相对真实方块约有 `7–9 mm` X 误差和 `22–24 mm` Y 误差。这个量级远大于 3→4 mm 的物理补偿，证明旧视觉策略发生了尺寸分布偏移，不能靠吸附、放宽 Z 容差或控制后处理掩盖。

因此可以直接加载旧权重做兼容性评估，但不能把它当作 50 mm 方案的最终结果。要让 50 mm 成为正式任务，需要用当前严格专家重新覆盖采集数据，再从已验收的 Stage1 初始化进行 Stage2 训练或针对新几何微调。80 个随机专家种子的无模型回归为 `80/80` 成功、`0` 次恢复、`0` 次释放前掉落，抓取前方块最大位移 `0.39 mm`；这证明新专家标签本身稳定，不代表旧神经网络权重已经学会新外观。

`step_best/step_final` 旧视频中的搬运掉落发生在策略连续给出开爪信号、通过两步释放防抖之后；方块不是在持续闭爪命令下自行脱落。同一运行时代码下 `step_28000` 的旧 60 mm 评估明显更好，因此目前证据支持晚期 checkpoint 的行为退化/过拟合可能，而不支持改大保持力、永久锁住 attachment 或继续增加释放防抖。后续应通过相同 seed 横向评估周期 checkpoint 判断；不能用环境特权状态替模型修正错误开爪。

服务端构造模型时保持 checkpoint 中的 `finetune_vlm`、`finetune_action_head` 等结构字段原值；评估冻结由 `eval()` 和 `no_grad()` 完成。不要在加载前重写这些字段，因为全视觉 LoRA 是否注入由 `finetune_vlm` 决定，改写会让合法 checkpoint 的 state dict 拓扑不匹配。当前 temporal-only LoRA 的 QKV 与组合输出残差均保留：这是为避免 shared spatial/temporal LoRA 梯度干扰而采用的有意扩展，不是待删除的旧逻辑。
