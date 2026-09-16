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
- 缓存：窗口索引版本 3；除源 parquet、视频和关键 meta 指纹外，还保存每个窗口的短期记忆事件标签。
- 微调：训练入口默认启用 LoRA，`rank=8`、`alpha=16`。动作头继续使用普通 LoRA；五个时间 ViT 层只在时间匹配/value 融合和组合记忆输出上使用独立低秩残差，空间 Q/K、视觉偏置、归一化和 layer scale 保持 Stage1 冻结值。
- 批处理：物理 batch 直接执行一次 batched ViT 和一次 batched 语言主干前向，不再把 batch 8 拆成 8 次 B=1 VLM。
- 优化器：正式单卡 LoRA 直接用一个 Python 进程启动 Accelerate + CUDA fused AdamW + BF16；DeepSpeed ZeRO-2 仅保留为多卡可选路径。worker 内的自动回退只能关闭 ZeRO，不能撤销外层 launcher 已建立的 NCCL 进程组。
- 旧 Stage2 checkpoint 缺少结构标记，服务端会按原来的并联相加注意力和 shared LoRA 结构严格加载，便于基线复核；新训练保存 `pi_mem_attention_mode=composed` 与 `separate_temporal_lora=true`。旧 checkpoint 不会因为新代码自动获得新记忆效果。

## 数据与模型流程

数据集按 episode 时间戳从早到晚选出 6 个观测；时间戳不可用时才退回固定步数。episode 开头不足 6 帧的槽位重复最早图像以维持固定 shape，但对应 `history_mask=False`。单样本推理会直接删除左填充帧；批量训练只删除整个 batch 共同无效的前缀，其余样本按各自 `history_mask` 屏蔽，样本之间不会共享时间注意力。状态 token 使用相同历史掩码。

第 4/8/12/16/20 个 ViT 层按 π-MEM 的组合顺序运行：先对同相机、同 patch 的历史做因果时间注意力，得到 time-mixed value；再用冻结的 Stage1 空间 Q/K 对这些 value 做空间注意力；共享 output projection 只执行一次。它不是“空间输出 + 时间输出”的并联残差。独立 temporal QKV LoRA 只改变时间匹配与 value 融合，空间 Q/K 始终来自冻结基础权重；memory-output LoRA 只存在于 K>1 组合路径。混合 batch 中仅当前帧有效的样本会逐样本屏蔽这些 adapter，K=1 与无历史样本都严格退化为原图像编码器。第 20 层融合完成后只保留当前帧视觉 token，上层 ViT 继续处理当前帧。训练时 `[B,K,V,C,H,W]` 一次进入 ViT，之后所有 prompt 一次进入 14 层语言主干；batch 维和时间维始终独立。当前视觉语言 token 与 6 个状态 token 进入 flow-matching 动作头，产生 14 步动作。

时序样本对所有时刻使用同一组增强参数，避免增强过程制造不存在的运动。MuJoCo 固定相机的笛卡尔动作标签与像素几何绑定，因此 `dataset/config.yaml` 默认设置 `preserve_spatial_calibration: true`：保留同步颜色增强，但禁用未同步变换动作标签的随机裁剪和旋转。默认保留原始 5 Hz 控制时间轴；只有显式添加 `--compact-static-frames` 才压缩静止帧。

数据窗口会按模型真正收到的 6 个历史索引分类，只有所采样历史里实际包含 `recover`，且当前仍处在恢复后的 `recover/approach/descend/close`，才标为 `post_failure_correction`；采样间隔中出现但没有进入模型输入的恢复帧不会误标。训练默认按各事件在现有数据中的实际频次做逆平方根加权，不规定恢复 episode 数量或固定失败比例，也不要求为了凑比例而延长采集。

现有 250 条 Stage2 专家 episode 中只有 1 条包含实际恢复，按 K=6 窗口统计只有 38 个 `post_failure_correction` 样本。自适应采样能提高这些已有证据被看到的频率，但不会凭空创造新的错误修正轨迹。因此本轮代码能够保证记忆结构、梯度和在线重规划链路正确，不能在新 checkpoint 实测前把“发现错误后必然快速修正”当成已经达成。若后续恢复指标仍不足，应保留严格专家门槛并采集自然出现的失败种子，而不是强行规定固定失败比例。

## 专家轨迹与抓取回归

- 抓取关闭门槛保持严格：XY 误差不超过 `6 mm`，手爪不能在目标抓取平面上方超过 `1 mm`；没有通过调宽抓取 Z 容差换取表面成功率。
- 恢复 `success_random` 的严格几何抓取辅助：仅在手爪已闭合、XY 不超过 `6 mm`、Z 不超过 `4 mm` 时允许保持力接管。正式专家数据验收仍额外要求首次 attachment 当帧存在真实双指接触，因此人工辅助不能让劣质专家视频通过。
- 搬运到下放阶段仍使用原 `10 mm` 高度就绪门槛。控制目标在安全高度上增加 `3 mm` 余量，用来抵消工作空间边缘的 IK 稳态误差；这不是放宽判定，也不改变抓取深度。
- 专家状态机对 transfer 的 X 对齐使用单向锁存，避免在 X-only 与完整 XY 目标之间来回切换；下放每步最大 Z 位移限制为 `6 mm`，避免阶段切换冲击。
- 80 个随机任务种子的物理回归为 `80/80` 完成、`0` 次恢复、`0` 次中途掉落；首次 attachment 最大 XY 误差 `2.673 mm`，抓前方块最大位移 `1.700 mm`，attachment 最大倾斜 `0.398°`，抓持阶段最大倾斜 `6.548°`。这验证的是专家与环境链路，不等同于尚未重新训练的策略成功率。

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
  --use_augmentation --image_size 448 --batch_size 6 --lr 1e-5 --dropout 0.1 --weight_decay 1e-3 --fused_adamw \
  --max_steps 24000 --warmup_steps 1500 --log_interval 20 --ckpt_interval 2000 --grad_clip_norm 1.0 \
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

`step_best` 沿用 MINT 原始判定：从全局 step 0 起用单 batch loss 更新内存中的 `best_loss`，但只有 `step > max(1000, warmup_steps)` 且再次刷新全局最低 loss 时才写入。

### 路径与恢复方式

LoRA 默认已经开启；命令仍显式写出 rank、alpha 和目标范围以便审计。当前 InternVL3-1B 的五个时间层新增 `245,760` 个 temporal-only LoRA 参数，完整模型实测总可训练参数为 `1,458,640`，共享视觉参数可训练数为 `0`；动作头继续使用 LoRA 及其少量偏置/归一化支持参数，语言主干冻结。LoRA-B 从零开始，因此初始化时 temporal 增量为零。只有明确需要恢复旧的非 LoRA 训练方式时才添加 `--no-use_lora`。

`lora_targets=vision,action` 不微调语言主干。`separate_temporal_lora` 默认开启；它不是复制整套 ViT attention，而是给时间匹配/value 融合和组合记忆输出增加低秩残差。空间 Q/K 不读取这些残差。`--finetune_temporal_vision` 和 `--fp32_temporal_parameters` 只影响 `--no-use_lora` 的原权重微调路径，因此正式 LoRA 命令不写这两个无效参数。如果以后确实需要语言适配，可显式改为 `vision,language,action`，但会增加反向计算和显存。

当前命令里的 `--resume --resume_pretrain --resume_path /home/user/mujoco+evo/ckpt/archive/evo1_mujoco_pickplace_stage1_random_stepbest` 表示：只从已验收的随机任务 Stage1 `step_best` 载入模型权重，新的 temporal/action LoRA 从零增量开始；不继承 Stage1 的 optimizer、scheduler、global step 或 best loss。这正是新结构首次训练应使用的初始化方式，不应删除。

现有 `stage2/step_best`、`step_final`、`step_16000`、`step_20000` 是旧的并联注意力 + shared spatial/temporal LoRA 结构，只作为基线评估，不能直接续训到新结构。开始新训练前，应先由操作者归档现有阶段二目录或明确选择一个空目录；本文不自动改名、移动或覆盖 checkpoint 路径。只有在已经产生 `separate_temporal_lora=true` 的新 checkpoint 后，意外中断续训才删除 `--resume_pretrain`，并把 `--resume_path` 指向该新 checkpoint；此时 `--max_steps` 表示最终全局步数。旧基线应直接评估原 checkpoint；`--no-separate_temporal_lora` 仅用于新组合注意力下的 shared-LoRA 消融，并不复刻旧并联数学。

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

兼容性方面，现有 `stage2/step_best` 已用当前服务端完成严格 state-dict 加载；自动选择 `legacy_additive`，42 个原 LoRA 模块可正常合并，K=6 推理输出为 `[14,24]` 且全部有限。它仍只用于复核旧基线，不应与新组合结构混合续训。


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

客户端正式命令不重复写默认参数：默认就是 `num_episodes=100`、`max_steps=250`、`memory_frames=6`、`memory_stride_steps=5`、`horizon=4`。不传 `--start-seed` 时每次运行随机生成 seed。客户端用 base64 JPEG 发送 `[K,V]` 图像并只发送真实相机；服务端仍兼容旧单帧 payload。K=1 严格走原单帧视觉路径。K>1 使用旧 checkpoint 只能证明结构可运行，不能代替 K=6 历史数据训练。
