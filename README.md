## Task1：拾取放置演示

当前默认运行整个 Task1、Task3、Task4 任务集合：一次采集、一次联合训练、同一个模型服务依次评估全部任务。Task1 是集合中的普通任务。下方保留已有阶段的命令与路径；当前多任务训练和评估命令见最后一节。

```text
MuJoCo 拾取放置任务
-> 脚本专家数据采集
-> cache/ 中的 LeRobot 风格 parquet、mp4 与 meta 数据集
-> Evo-1 共享训练入口
-> Evo-1 WebSocket 推理服务
-> MuJoCo 回放评估
```

### 当前标准状态

- 数据集根目录：`/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace`
- 训练默认读取 `Evo-1/Evo_1/dataset/config.yaml` 中的全部任务数据，单卡直接运行 `python scripts/train.py`。
- 训练入口默认不启用 LoRA，且本 task 不需要启动 LoRA；π-MEM 正式参数和关闭 LoRA 的兼容方式见下方 Task2。
- 当前共享服务默认 checkpoint：`/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3/step_best`；已有 Task1 权重仍可通过 `--ckpt-dir` 显式加载。
- 评估默认依次执行 Task1、Task3、Task4，视频保存在 `mujoco_pickplace/outputs/eval_videos/<时间戳>/taskN/`，并保存集合汇总。

### 仓库结构

```text
mujoco_pickplace/                  MuJoCo 任务、数据采集、数据检查与评估客户端
Evo-1/Evo_1/                       训练和服务端推理所需的最小 Evo-1 文件
Mujoco_training_dataset/cache/     当前标准数据集位置
ckpt/                              训练 checkpoint 输出
```

### 1. 采集 MuJoCo 示范

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

采集默认遍历 Task1、Task3、Task4，并分别清理所选数据集中的 `data/`、`videos/`、`meta/` 后重写。不会删除整个 cache 根目录。每个任务默认采集 250 条通过严格验收的轨迹；`--num-episodes` 指定每个任务的数量。保留现有 episode 并继续编号时使用 `python collect_data.py --append`。

Task1 数据继续写成 LeRobot 风格 episode，保存在原路径；Task3、Task4 由同一次采集自动写入同一 cache 根目录中配置的数据集：

```text
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace
```

### 2. 检查 Evo-1 数据加载

```bash
conda activate Evo1
cd /home/user/mujoco+evo/mujoco_pickplace
python check_dataset.py --require-evo --require-strict-grasp-quality
```

当前默认 K=6 的三个任务加载形状：

```text
images: [6, 3, 3, 448, 448]
state: [6, 24]
action: [14, 24]
state mask sum: 8 per valid history frame
action mask sum: 56 for Task1/Task3, 98 for Task4
```

### 3. 使用 Evo-1 训练

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
accelerate config
```

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
rm -rf -- /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1_random/step_*
accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend no --use_deepspeed --deepspeed_config_file ds_config.json scripts/train.py \
  --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.1 \
  --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 16000 \
  --log_interval 50 --ckpt_interval 2500 --warmup_steps 3000 --grad_clip_norm 1.0 \
  --num_layers 8 --horizon 14 --finetune_action_head --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 --state_dim 24 --use_state --memory_frames 1 \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1_random
```

上述历史命令的 checkpoint 根目录（保留原路径记录）：

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

### 4. 启动 Evo-1 推理服务

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py --ckpt-dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1_random/step_best
```

### 5. 在 MuJoCo 中评估

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```

## Task2：π-MEM K=6 短期记忆


```text
-> K=6 图像、状态与历史掩码
-> InternVL 因果时间 value -> 空间注意力的组合结构
-> ViT 第 20 层后仅保留当前帧 token
-> 完整 14 层语言主干与 flow-matching 动作头
-> 预测 14 步，执行 4 步后重新规划
```

### 当前配置

- 记忆长度：`K=6`；历史间隔：`1.0 s`。
- 时间层：ViT 第 `4/8/12/16/20` 层。
- 上层压缩：第 20 层后丢弃过去帧 token；ViT 其余层继续运行。
- 语言主干：完整保留并运行 `14` 层，只关闭无用的 hidden-state 收集和 KV cache。
- 输入：`images [B,6,3,3,448,448]`、`state [B,6,24]`、`history_mask [B,6]`。
- 输出：`14 × 24` 动作；自由空间和抬升/搬运阶段默认执行前 4 步，接近抓取平面或即将由开变闭时每步重新规划。
- 夹爪：Task1 保留 0.4/0.6 滞回、闭合立即执行、开爪连续 2 步强信号的 `success_random` 行为；Task3 的二值开闭立即执行，与专家时序一致；Task4 保留连续开度。
- 缓存：窗口索引版本 8；除源 parquet、视频和关键 meta 指纹外，还保存每个窗口的短期记忆事件标签；真实失败后的纠正示范保留历史前缀，但不把失败动作作为训练目标。
- 微调：训练入口默认不启用 LoRA。显式 `--use_lora` 时使用 `rank=8`、`alpha=16`；动作头使用普通 LoRA，五个时间 ViT 层使用独立低秩残差。
- 批处理：物理 batch 直接执行一次 batched ViT 和一次 batched 语言主干前向，不再把 batch 8 拆成 8 次 B=1 VLM。
- 优化器：正式单卡 LoRA 直接用一个 Python 进程启动 Accelerate + CUDA fused AdamW + BF16；DeepSpeed ZeRO-2 仅保留为多卡可选路径。worker 内的自动回退只能关闭 ZeRO，不能撤销外层 launcher 已建立的 NCCL 进程组。

### 数据与模型流程

数据集按 episode 时间戳从早到晚选出 6 个观测；时间戳不可用时才退回固定步数。episode 开头不足 6 帧的槽位重复最早图像以维持固定 shape，但对应 `history_mask=False`。单样本推理会直接删除左填充帧；批量训练只删除整个 batch 共同无效的前缀，其余样本按各自 `history_mask` 屏蔽，样本之间不会共享时间注意力。状态 token 使用相同历史掩码。

若仅用旧权重初始化新训练，显式使用原有 `--resume --resume_pretrain --resume_path` 方式。MuJoCo 固定相机的笛卡尔动作标签与像素几何绑定，因此 `dataset/config.yaml` 默认设置 `preserve_spatial_calibration: true`：保留同步颜色增强，但禁用未同步变换动作标签的随机裁剪和旋转。只有显式添加 `--compact-static-frames` 才压缩静止帧。

### 仓库结构

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py  时空视觉编码
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py       K 帧预处理与语言融合
Evo-1/Evo_1/model/action_head/flow_matching.py          状态记忆与动作头
Evo-1/Evo_1/model/lora.py                               普通 LoRA、时间路径独立 LoRA 与推理合并
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py      K=6 数据窗口与按任务归一化
Evo-1/Evo_1/scripts/train.py                            训练入口
Evo-1/Evo_1/scripts/Evo1_server.py                      推理服务
mujoco_pickplace/collect_data.py                        默认覆盖、显式追加的数据采集
mujoco_pickplace/eval_policy_client.py                  在线闭环客户端
```

### 环境状态

当前 `Evo1` 环境已经具备：

| 组件 | 已验证状态 |
|---|---|
| Python / PyTorch | Python 3.10.20；PyTorch 2.12.0.dev + CUDA 12.8 |
| C++ / Ninja | Conda GCC/G++ 14.3；Ninja 1.13 |
| 优化器 | 单卡 CUDA fused AdamW；DeepSpeed 0.19.2 仅作多卡可选 |
| CUDA 开发库 | `libcurand-dev 10.3.9.90`、`libcufile-dev 1.13.1.3` |
| 异步 I/O | `libaio 0.3.113`，多卡 DeepSpeed 兼容性检查通过 |
| FlashAttention | 本机扩展因 `CXXABI_1.3.15` 缺失未能加载；单步训练与评估启动验证使用 PyTorch attention 回退 |

Adam 一、二阶状态仅为十几 MB 量级，单张 8 GB GPU 使用 ZeRO-2 没有第二张卡可分片，反而增加包装器和通信缓冲。当前单卡正式路径用 CUDA fused AdamW，并通过分段 checkpoint 时间注意力与 ViT MLP、以 BF16 执行 LoRA 矩阵乘法来压低峰值；FP32 LoRA 主参数、梯度和 optimizer state 仍保留。正式命令直接运行 `python scripts/train.py`，从源头避免 `accelerate launch -> deepspeed_launcher -> torch.distributed/NCCL`。训练脚本在 `Accelerator` 初始化前关闭单卡 LoRA 的 DeepSpeed wrapping 仍作为旧命令兼容保护，但 worker 已经无法撤销外层 launcher 创建的单 rank 进程组，因此不能以日志中的 `ZeRO stage=None` 单独证明 launcher 已移除。`ds_config_pi_mem.json` 不删除，供以后多卡训练使用，其中也不启用 CPU optimizer offload。FlashAttention 的 PyTorch 回退只用于明确报错和环境诊断，不能当作正式训练快路径。

### 1. 安全采集数据

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python collect_data.py
```

只有需要保留现有数据并继续编号时，才显式添加：

```bash
python collect_data.py --append
```

### 2. 检查数据

```bash
conda activate Evo1
cd /home/user/mujoco+evo
MUJOCO_GL=egl python mujoco_pickplace/check_dataset.py --require-evo --require-strict-grasp-quality
```

`--require-evo` 禁止在缺少 Evo 数据接口依赖时把 K=6 shape 检查静默跳过；`--require-strict-grasp-quality` 要求 `stable-grasp-v3` 的闭合位置、双指接触、接触时方块位移/倾斜和抓持持续证据。这两个开关只用于训练前严格验收专家数据，不进入训练或评估策略。

预期关键形状：

```text
images: [6, 3, 3, 448, 448]
state: [6, 24]
history_mask: [6]
action: [14, 24]
```

### 3. 训练

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=bf16
rm -rf -- /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2/step_*
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

#### 路径与恢复方式

LoRA 默认关闭；上面的 Stage2 命令显式写出 `--use_lora`，因为从旧 Stage1 权重初始化并训练时间专用适配器时需要该结构。当前 LoRA 配置实测总可训练参数为 `1,458,640`。不传 `--use_lora` 时使用普通参数微调路径；不能把包含独立时间 LoRA 的 `step_28000` 直接作为无 LoRA 模型严格加载。

`lora_targets=vision,action` 不微调语言主干。`separate_temporal_lora` 默认开启。空间 Q/K 不读取这些残差。K=1 新训练没有时间注意力执行，因此视觉 LoRA 使用已有空间投影路径，并在保存配置中记录 `separate_temporal_lora=false`，避免装入没有梯度的时间适配器；服务端仍按各 checkpoint 保存的结构加载旧权重。`--finetune_temporal_vision` 和 `--fp32_temporal_parameters` 只影响无 LoRA 的原权重微调路径。如果以后确实需要语言适配，可显式改为 `vision,language,action`，但会增加反向计算和显存。


### 4. 推理与评估

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py --ckpt-dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage2/step_28000
```

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```

## Task3、Task4

```text
任务定义中的场景与语言指令
-> 同一 Panda 机器人组中的多个数据集
-> 同一 EVO + π-MEM 联合训练入口
-> 一个 checkpoint 与 WebSocket 服务
-> 按任务区分视频的 MuJoCo 闭环评估
```

### 当前配置

| 任务 | 指令与环境 | 数据目录（不覆盖 Task1） | 动作 |
| --- | --- | --- | --- |
| Task1 | 原有 50 mm Cube pick-and-place | `Mujoco_training_dataset/cache/mujoco_pickplace`
| Task3 | 实体筷架上的筷子抓取入盒，桌高随机 | `Mujoco_training_dataset/cache/mujoco_chopstick`
| Task4 | 观察目标、关闭抽屉、记住位置、拉开取物并放置 | `Mujoco_training_dataset/cache/mujoco_drawer_memory`

### 1. 采集 MuJoCo

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python collect_data.py
```

可选纠正补采：`--recovery-actions` 读取真实失败日志；下面示例追加 2 条纠正示范，需已有同版本正常示范。

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python collect_data.py --task 3 --append --num-episodes 2 \
  --recovery-actions outputs/step_best/task3/actions.jsonl
```

### 2. 检查数据

```bash
conda activate Evo1
cd /home/user/mujoco+evo
MUJOCO_GL=egl python mujoco_pickplace/check_dataset.py --require-evo --require-strict-grasp-quality
```

```bash
conda activate Evo1
cd /home/user/mujoco+evo
MUJOCO_GL=egl python mujoco_pickplace/open_loop_test.py
```

- `--audit-only`：只检查数据，不加载模型。
- `--replay-steps 600`：重放最多 600 个记录动作，核对动作前状态。
- `--no-memory-ablation`：跳过历史消融。
- `--drawer-selection-probe --closed-loop-max-steps 40`：追加四抽屉短段选择检查。

### 3. 训练

`resume_path` 不会自动指向其他模型。`--resume --resume_pretrain` 表示只加载初始化权重并开始新的训练进度；普通 `--resume` 还需要恢复兼容的优化器、调度器和进度，恢复失败不会自动降级为只加载权重。无 LoRA 的权重初始化可继续使用同一个 Stage3 保存根目录，新运行仍会按既有规则覆盖该目录内相应的 checkpoint 标签；跨结构 LoRA 初始化保留基础模型目录保护。正式无 LoRA 训练去掉 `--use_lora`、`--lora_rank`、`--lora_alpha`、`--lora_dropout`、`--lora_targets`、`--lora_train_bias_norm`；若继续使用上述普通 Stage1 初始化点，保留显式 `--resume --resume_pretrain --resume_path`。不加载初始化权重时去掉三个 resume 参数。单卡仍直接使用 `python scripts/train.py`。

仅在开始全新训练、且初始化或续训来源不在清理目录中时，单独执行下面的预检查与清理。正式数据或配置检查失败时，`&&` 会阻止删除旧权重；普通续训不要执行这一段。清理范围只包含该 Stage3 目录中的 `step_*`。

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
MUJOCO_GL=egl python ../../mujoco_pickplace/check_dataset.py --require-evo && \
  rm -rf -- /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3/step_*
```

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=bf16
python scripts/train.py \
  --run_name evo1_pi_mem --action_head direct --drawer_intent_supervision --memory_event_sampling --use_flash_attn \
  --dataset_config_path dataset/config.yaml --vlm_name OpenGVLab/InternVL3-1B \
  --use_augmentation --image_size 448 --batch_size 2 --lr 1e-5 --dropout 0.1 --weight_decay 1e-3 --fused_adamw \
  --max_steps 80000 --warmup_steps 1000 --log_interval 20 --ckpt_interval 5000 --grad_clip_norm 1.0 \
  --num_layers 8 --num_workers 4 --horizon 14 --per_action_dim 24 --state_dim 24 --use_state \
  --memory_frames 6 --memory_stride_seconds 1.0 --memory_stride_steps 5 --temporal_layer_interval 4 \
  --temporal_drop_past_after_layer 20 \
  --gradient_checkpointing --compact_masked_views --finetune_action_head \
  --disable_wandb --disable_swanlab \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3
```

### 4. 推理与评估

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python eval_policy_client.py
```