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

以下保留 Task1 早期实验的 Accelerate/DeepSpeed 命令及保存路径。这些参数对应当时仅含 Task1 的数据配置；当前默认配置已包含全部任务，正式联合训练使用最后一节的单进程命令。

早期实验按照上游 Evo-1 (详见 Evo-1 文件夹内README.md)方式配置 Accelerate/DeepSpeed：

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
MuJoCo 5 Hz 历史观测
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
- 夹爪：0.4/0.6 滞回；闭合立即执行以保留 `success_random` 的夹取时机，只有开爪/释放需要连续 2 步强信号，防止搬运途中单帧误开爪。
- 缓存：窗口索引版本 7；除源 parquet、视频和关键 meta 指纹外，还保存每个窗口的短期记忆事件标签。
- 微调：训练入口默认不启用 LoRA。显式 `--use_lora` 时使用 `rank=8`、`alpha=16`；动作头使用普通 LoRA，五个时间 ViT 层使用独立低秩残差。
- 批处理：物理 batch 直接执行一次 batched ViT 和一次 batched 语言主干前向，不再把 batch 8 拆成 8 次 B=1 VLM。
- 优化器：正式单卡 LoRA 直接用一个 Python 进程启动 Accelerate + CUDA fused AdamW + BF16；DeepSpeed ZeRO-2 仅保留为多卡可选路径。worker 内的自动回退只能关闭 ZeRO，不能撤销外层 launcher 已建立的 NCCL 进程组。

### 数据与模型流程

数据集按 episode 时间戳从早到晚选出 6 个观测；时间戳不可用时才退回固定步数。episode 开头不足 6 帧的槽位重复最早图像以维持固定 shape，但对应 `history_mask=False`。单样本推理会直接删除左填充帧；批量训练只删除整个 batch 共同无效的前缀，其余样本按各自 `history_mask` 屏蔽，样本之间不会共享时间注意力。状态 token 使用相同历史掩码。

时序样本对所有时刻使用同一组增强参数，避免增强过程制造不存在的运动。MuJoCo 固定相机的笛卡尔动作标签与像素几何绑定，因此 `dataset/config.yaml` 默认设置 `preserve_spatial_calibration: true`：保留同步颜色增强，但禁用未同步变换动作标签的随机裁剪和旋转。默认保留原始 5 Hz 控制时间轴；只有显式添加 `--compact-static-frames` 才压缩静止帧。

数据窗口会按模型真正收到的 6 个历史索引分类，只有所采样历史里实际包含 `recover`，且当前仍处在恢复后的 `recover/approach/descend/close`，才标为 `post_failure_correction`；采样间隔中出现但没有进入模型输入的恢复帧不会误标。训练默认按各事件在现有数据中的实际频次做逆平方根加权，不规定恢复 episode 数量或固定失败比例，也不要求为了凑比例而延长采集。

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

### 3. 训练

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

#### 路径与恢复方式

LoRA 默认关闭；上面的 Stage2 命令显式写出 `--use_lora`，因为从旧 Stage1 权重初始化并训练时间专用适配器时需要该结构。当前 LoRA 配置实测总可训练参数为 `1,458,640`。不传 `--use_lora` 时使用普通参数微调路径；不能把包含独立时间 LoRA 的 `step_28000` 直接作为无 LoRA 模型严格加载。

`lora_targets=vision,action` 不微调语言主干。`separate_temporal_lora` 默认开启；它不是复制整套 ViT attention，而是给时间匹配/value 融合和组合记忆输出增加低秩残差。空间 Q/K 不读取这些残差。`--finetune_temporal_vision` 和 `--fp32_temporal_parameters` 只影响无 LoRA 的原权重微调路径。如果以后确实需要语言适配，可显式改为 `vision,language,action`，但会增加反向计算和显存。


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

默认采集与评估直接遍历 Task1、Task3、Task4；默认训练配置也包含全部任务。Task1 与其他任务共用同一采集循环、数据写入和评估循环，只在任务定义中保留各自的场景、专家与验收规则。

正式代码统一位于 `mujoco_pickplace/`，不依赖 `tests/`。沿用现有 `collect_data.py`、`eval_policy_client.py`、`episode_dataset.py` 和 EVO 的 `scripts/train.py`、`scripts/Evo1_server.py`。新增的 Python 源文件只有 `tasks.py`（任务定义、历史与共享调度）和 `task_env.py`（新任务环境、专家与物理验收）；各任务场景位于 `assets/task3_scene.xml`、`assets/task4_scene.xml`。

| 任务 | 指令与环境 | 数据目录（不覆盖 Task1） | 动作 |
| --- | --- | --- | --- |
| Task1 | 原有 50 mm Cube pick-and-place | `Mujoco_training_dataset/cache/mujoco_pickplace` | 位移、夹爪；保留原固定腕姿态 |
| Task3 | 贴桌筷子抓取入盒，桌高随机 | `Mujoco_training_dataset/cache/mujoco_chopstick` | 位移、夹爪；筷子方向限制在已验证的 ±5° |
| Task4 | 观察目标、关闭抽屉、记住位置、拉开取物并放置 | `Mujoco_training_dataset/cache/mujoco_drawer_memory` | 位移、世界坐标系旋转向量增量、夹爪 |

三个任务使用同一个 Panda embodiment、8 维机器人本体状态、7 维动作和同一个模型。位移单位为米，旋转增量单位为弧度，夹爪 `0=关闭、1=打开`。Task1/Task3 使用二值夹爪命令并保留原有开爪防抖；Task4 使用连续开度目标，接近把手的 `0.25` 原样执行，不经过二值阈值转换。状态沿用 Task1 的 hand quaternion 到 axis-angle 的定义，不包含物体坐标、抽屉编号或专家阶段。Task1/Task3 的旋转维度通过各数据集的动作掩码屏蔽；Task4 三个旋转维度参与训练和执行。

数据中的 `timestamp` 保留采集时序，用于按真实控制时间选择历史；新写入的 `video_timestamp` 从 MP4 第 0 帧开始，等于 `frame_index / fps`。加载器对 `mujoco-evo-episodes` 使用视频时间寻帧，已有未写入该列的数据也按帧号与 fps 恢复正确映射，无需重写原始数据或视频。其他 LeRobot 数据保持其原有时间语义。派生缓存索引版本已升级为 7；加载时拒绝旧索引并在原缓存目录重新构建，不复用旧的错帧窗口。

训练加载器、检查入口及新任务数据写入共用 `episode_dataset.py` 中的接触质量校验规则，要求完整的必需检查项均通过，并验证无运输掉落、动作失败及历史时序缺失。Task1 的训练入口复用检查入口原有的 `precision-grasp-stable-v3` 规则，双指接触、抓取定位与深度、夹持保持、倾斜及角速度阈值均不改变。

`dataset/config.yaml` 在同一机器人组中列出三个数据集，按数据集保存归一化统计以保护 Task1 原有范围。派生缓存使用新的 `mujoco_panda_multitask` 命名空间和源文件指纹，数据组合变化时重新核对 manifest；指纹同时包含 K、历史采样间隔、动作块长度及归一化统计文件，显式缓存目录也不能复用不同配置的窗口。原生 episode 清单须与实际 parquet 和总帧数对应，配置要求的相机视频及语言指令缺失时直接报错。模型服务通过 `task_key` 选择同一 checkpoint 内的相应统计，不切换权重；未知统计键会报错。Task1 的评估统计键统一为 `mujoco_pickplace`，兼容已有同名单任务 checkpoint。

Task3 的筷子长 240 mm，截面约 5–7 mm，自然接触桌面；桌高只在新任务内随机为 30–70 mm，不改变 Task1。专家只有在真实闭合失败后才尝试自然重试，没有故意失败和固定纠正比例。已有少量通过验收的样例都是首次成功，不能据此宣称模型已经学会失败纠正。

Task4 有完整柜体、四个被动滑动抽屉、横向 U 形把手、180 mm 行程与限位。层间距 120 mm、柜底垫座 40 mm；各抽屉有实际承重和碰撞的 30 mm 垫层，物体真实落在垫层上，为 Panda 手掌提供边缘间隙。观察与关抽屉属于明确的模拟人类展示流程：展示时施加关抽屉的力，执行前清零；执行期间只驱动机器人，不给抽屉或物体施加辅助力。模型从视觉历史选择抽屉，评估入口不调用专家来选择或纠错。这是参考 π-MEM 的模拟任务，不是完全相同的实物实验。

`panda.xml` 在本地新增新任务专用的命名夹爪与接触配置，保留 Task1 原有默认配置、碰撞形状与开口。Task3/Task4 通过这些配置使用真实接触，未新增吸附、焊接或物体重力补偿。`mujoco_menagerie` 仍不纳入 Git 上传范围。现有本地资产必须保留这些已写入的配置；正式运行不读取测试目录中的资产副本。

### 1. 采集 MuJoCo 示范

采集继续使用已有入口，例如：

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
MUJOCO_GL=egl python collect_data.py
```

可用 `--dataset-dir` 一次指定整套数据的根目录；仅选择一个任务时该参数指向其数据集目录。`--task 1` 或 `--tasks 3 4` 只用于明确选择子集。新任务会拒绝不合格轨迹；成功必须经过真实抓取、抬升、运输保持和稳定放置。

### 2. 检查数据

```bash
conda activate Evo1
cd /home/user/mujoco+evo
python mujoco_pickplace/check_dataset.py --require-evo --require-strict-grasp-quality
```

检查入口默认一次检查 Task1、Task3、Task4，无需分别运行命令。原始检查覆盖所有 episode；EVO 接口检查还核对全部原生视频的帧数、帧率和分辨率，训练解码缺失帧时报错，不用最后一帧补齐。EVO 接口检查使用同一份训练配置，在独立检查缓存中为每个 episode 构造一个窗口，并实际解码每个任务的代表样本。Task1 保留原有严格抓取阈值；新任务检查 `mujoco-panda-contact` 的完整接触轨迹、真实抬升、运输保持和稳定放置，Task4 还检查把手方向、正确抽屉和选择时的视觉历史。三个数据集的动作掩码、场景配置和组合指纹都按默认训练配置核对。只检查原始数据时加 `--raw-only` 并去掉 `--require-evo`；任务子集和 `--dataset-dir` 的含义与采集入口一致。

需要检查已训练模型的接触精度时，在上面的检查命令中增加 `--policy-url ws://localhost:9000 --policy-report /home/user/mujoco+evo/mujoco_pickplace/outputs/contact_precision.json`，对应现有服务器地址。检查每个所选任务的第一条已验收专家轨迹，报告关键阶段的毫米级位移误差、旋转误差和夹爪误差；Task4 另外比较专家、替换模型位移、替换模型位移与旋转的短段物理执行，记录拉动距离与接触保持，并保持当前画面和全部本体状态相同，仅交换不同抽屉的展示历史，检查模型动作对视觉历史的响应。该对照不等于正确选屉或操作成功。报告写入文件，原有终端输出格式不变。这是专家观测输入下的有界检查，不是闭环成功率评估；阶段标签和真实抽屉位置只用于检查，不发给模型。

### 3. 训练

训练继续使用现有 `scripts/train.py`。默认数据配置就是 `dataset/config.yaml`，其中同一 `mujoco_panda` 机器人组列出全部任务；无需切换另一份配置，输出仍沿用 `/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3`；Task 编号不是训练 stage。默认不使用 LoRA，本次只执行了显式 LoRA、batch=1、单步的小规模验证，没有启动完整训练，也没有改动 `step_best` 的选择逻辑。

现有 π-MEM 采样对 Task1/Task3 的下探与闭合，以及 Task4 的 `handle_approach/handle_close/pull/object_descend/object_close` 在接触精度类别中按阶段区分，避免长接近阶段稀释短暂接触动作；`handle_approach` 在有效历史含目标展示时仍用于记忆选择检查；关抽屉时仍可见的目标画面，使用既有质量记录中经过可见性检查的时间点补充记忆采样分类，标记只用于采样，不进入模型输入；无效补齐帧不作为历史证据。权重继续按已有的类别频次平方根计算，不设置失败比例。模拟人展示、关闭抽屉阶段只提供历史画面，不监督执行动作；保留真实仿真时间与原始数据。派生缓存版本为 7，训练与接口检查会拒绝旧索引并按各自原路径重建，不需要重采这套专家数据。

`resume_path` 不会自动指向其他模型。`--resume --resume_pretrain` 表示只加载初始化权重并开始新的训练进度；普通 `--resume` 还需要恢复兼容的优化器、调度器和进度，恢复失败不会自动降级为只加载权重。无 LoRA 的权重初始化可继续使用同一个 Stage3 保存根目录，新运行仍会按既有规则覆盖该目录内相应的 checkpoint 标签；跨结构 LoRA 初始化保留基础模型目录保护。旧 `step_28000` 含独立时间 LoRA，已验证可作为同结构 LoRA 验证的初始化；不能直接作为无 LoRA 模型严格加载或随意删除适配器键。已有普通 Stage1 初始化点 `ckpt/archive/evo1_mujoco_pickplace_stage1_random_stepbest` 的全部基础参数键和形状也已核对兼容，但初始化方式必须显式指定。联合训练使用默认的 `dataset/config.yaml` 和 `/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3`。不按任务切换权重或建立另一套训练工程。正式无 LoRA 训练去掉 `--use_lora`、`--lora_rank`、`--lora_alpha`、`--lora_dropout`、`--lora_targets`、`--lora_train_bias_norm`；若继续使用上述普通 Stage1 初始化点，保留显式 `--resume --resume_pretrain --resume_path`。不加载初始化权重时去掉三个 resume 参数。单卡仍直接使用 `python scripts/train.py`。

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=bf16
python scripts/train.py \
  --run_name evo1_pi_mem --action_head flowmatching --use_flash_attn \
  --dataset_config_path dataset/config.yaml --vlm_name OpenGVLab/InternVL3-1B \
  --use_augmentation --image_size 448 --batch_size 2 --lr 1e-5 --dropout 0.1 --weight_decay 1e-3 --fused_adamw \
  --max_steps 80000 --warmup_steps 1000 --log_interval 20 --ckpt_interval 2000 --grad_clip_norm 1.0 \
  --num_layers 8 --num_workers 4 --horizon 14 --per_action_dim 24 --state_dim 24 --use_state \
  --memory_frames 6 --memory_stride_seconds 1.0 --memory_stride_steps 5 --temporal_layer_interval 4 \
  --temporal_drop_past_after_layer 20 \
  --gradient_checkpointing --compact_masked_views --finetune_action_head \
  --disable_wandb --disable_swanlab \
  --save_dir /home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage3
```

### 4. 推理与评估

Task1 默认预算为原有的 250 步，Task3 为 300 步，Task4 为 600 步；只有显式 `--max-steps` 才覆盖预算。Task3/Task4 的执行契约为每次推理执行 1 个控制动作以取得接触反馈，启动日志沿用原有 logging 格式打印实际 horizon；每步继续显示动作、夹爪值、reward 和 done。选择新任务时，显式指定 `--horizon` 必须为 1；其他值会在启动前报错。Task1 可通过 `--task 1 --horizon N` 设置动作块长度，保留原有精细重规划规则。Task3 评估单独记录手指与桌面的接触及穿透深度，不再仅因该接触否定成功；仍要求双指真实抓取、抬升、运输无掉落、整根筷子进入盒内并稳定释放。手掌、机械臂和其他异常接触仍保留检查。专家数据采集继续严格拒绝手指碰桌；Task4 的接触、正确抽屉和稳定放置标准保持原样。模型训练的 `--horizon 14` 仍是预测动作块长度，与评估执行步数是不同参数。

联合训练得到新的 Stage3 checkpoint 后，使用同一个现有服务器：

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