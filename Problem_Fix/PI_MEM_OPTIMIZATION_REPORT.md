# π-MEM K=6 接入与优化修复报告

## 1. 原模型没有短期观测记忆

### 遇到的问题

原模型只编码当前图像和当前状态，无法利用 K 个历史时刻判断物体运动、夹爪接触和动作阶段。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
Evo-1/Evo_1/model/action_head/flow_matching.py
```

### 修改前

```text
当前图像 -> 单图 ViT -> 语言主干
当前状态 -> 动作头
```

### 修改后

```text
K=6 图像 -> 因果时空 ViT -> 当前帧视觉 token
K=6 状态 -> 共享状态编码 -> 6 个带掩码的状态 token
```

时间注意力只连接同相机、同空间块，固定时间编码满足 `e(0)=0`；`K=1` 直接调用原始单图路径。

### 验证结果

因果性、左填充隔离、K=1 精确路径和时间 QKV 反向传播测试均通过。

## 2. 历史窗口可能越过 episode 或与时间轴不一致

### 遇到的问题

按数组位置随意取历史帧会在 episode 开头越界；静态帧压缩后继续使用原时间戳，会使图像、状态和时间位置错位。

### 修改文件

```text
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
mujoco_pickplace/collect_data.py
mujoco_pickplace/episode_dataset.py
```

### 修改前

数据项只包含当前时刻，缺少统一的 K 帧时间选择和有效性标记。

### 修改后

- 优先按 parquet 时间戳选择 `K=6`、间隔 `1.0 s` 的历史。
- 时间戳不可用时退回 `memory_stride_steps=5`。
- episode 开头左填充第一帧，并令对应 `history_mask=False`。
- 默认保留完整 5 Hz 时间轴；显式压缩静态帧时同步重建视频与 parquet 时间戳。
- 缓存目录包含 horizon、K、步长和秒间隔，避免读取旧配置缓存。

### 验证结果

500 个 episode、64,185 帧及 64,185 个 K=6 Evo 窗口检查通过。

## 3. 论文要求在 ViT 上层丢弃过去时刻 token

### 遇到的问题

上一版虽然最后只输出当前帧，但 6 帧 token 仍经过全部 24 层 ViT，没有实现 MEM 图 4 的上层压缩，显存和计算量偏高。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
Evo-1/Evo_1/scripts/Evo1.py
Evo-1/Evo_1/scripts/train.py
```

### 修改前

```text
第 4/8/12/16/20/24 层：6 帧时间注意力
第 1–24 层：始终保留 6 帧 token
```

### 修改后

```text
第 4/8/12/16/20 层：6 帧时间注意力
第 20 层后：只保留已经融合历史的当前帧
第 21–24 层：原有单帧空间层
```

默认参数为：

```bash
--temporal_layer_interval 4
--temporal_drop_past_after_layer 20
```

### 验证结果

真实 K=6 前反向峰值已分配显存由 `4345.8 MiB` 降至 `3544.7 MiB`，减少 `801.1 MiB`，约 `18.4%`。

## 4. 视觉编码器保留全部层输出引用

### 遇到的问题

原实现用列表保存嵌入层和 24 个 ViT 层的全部输出，只为最后读取 `select_layer=-1`。这会延长中间张量生命周期，并削弱 gradient checkpointing 的显存收益。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py
```

### 修改前

```python
selected_states = [hidden_states]
selected_states.append(hidden_states)
hidden_states = selected_states[select_layer]
```

### 修改后

先把 `select_layer` 解析成目标层，只运行到所需层，不保存所有层输出；默认 `-1` 仍运行完整网络。

固定正弦时间位置编码也改为每次视觉前向只生成一次，由各时间层复用。

## 5. 填充相机造成无效视觉和语言计算

### 遇到的问题

MuJoCo 只有一路真实相机，但固定三相机 batch 会让 18 张图像进入 ViT，并让两个无效相机的图像占位 token 进入语言主干。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
```

### 修改前

```text
6 时刻 × 3 相机 = 18 张图像
融合序列约 805 token
```

### 修改后

- ViT 前删除无效相机，只编码 `6 × 1` 张真实图像。
- 语言主干前删除无效相机占位，而不是计算后只改 attention mask。
- 融合序列为 `[1,277,896]`，277 个 token 全部有效。

## 6. 图像预处理存在 CPU/PIL 往返

### 遇到的问题

训练张量被逐张转到 CPU、转 PIL、再转 Tensor，产生同步、Python 对象和 8 位量化误差。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
```

### 修改前

```text
Tensor -> CPU -> PIL -> Resize -> Tensor
```

### 修改后

K×V 图像整批在 Tensor 空间执行 bicubic resize 和 ImageNet 归一化；PIL 路径只保留给服务端图像输入。

## 7. 无效动作目标会影响有效动作预测

### 遇到的问题

动作损失虽然屏蔽了无效维度，但无效 `actions_gt` 仍参与 flow 中间状态并进入动作 token 自注意力，可能改变有效维度的预测。

### 修改文件

```text
Evo-1/Evo_1/model/action_head/flow_matching.py
Evo-1/Evo_1/scripts/train.py
```

### 修改前

只在损失或噪声上应用 `action_mask`。

### 修改后

```python
action_intermediate_seq = action_intermediate_seq * action_mask
squared_error = (pred_velocity - target_velocity).square()
loss = (squared_error * action_mask).sum() / action_mask.sum().clamp_min(1)
```

无效目标在进入动作投影前归零；每个样本无有效动作时直接报错。

### 验证结果

将无效目标维度增加 1000 后，有效预测保持逐元素完全一致。

## 8. 服务端拒绝单路真实相机

### 遇到的问题

客户端已只发送一路物理相机，但服务端仍要求相机数必须等于 3。旧烟测直接调用模型，绕过了服务校验，所以没有发现该错误。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1_server.py
tests/smoke_k6_server_protocol.py
```

### 修改前

```python
if num_views != expected_views:
    raise ValueError(...)
```

### 修改后

服务端允许 `1..max_views` 路物理相机；烟测改为调用 `infer_from_json_dict`，覆盖解码、校验、归一化、模型和动作反归一化完整路径。

### 验证结果

单路真实相机的 K=6 请求通过完整服务入口，输出 `[14,24]` 且全部有限。

## 9. 旧列表协议错误交换 RGB 通道

### 遇到的问题

MuJoCo Renderer 输出 RGB，旧服务端却按 BGR 转 RGB，导致红蓝通道交换。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1_server.py
tests/test_short_term_memory.py
```

### 修改后

列表图像直接按 RGB 构造 PIL，并验证输入必须是 `[H,W,3]`。纯红图像回归测试保持 `[255,0,0]`。

## 10. 在线评估开环执行时间过长

### 遇到的问题

模型一次预测 14 步后，客户端原样开环执行 14 步才重新观测。在 5 Hz 控制频率下约为 2.8 秒，误差无法及时修正。

### 修改文件

```text
mujoco_pickplace/eval_policy_client.py
```

### 修改前

```text
预测 14 步 -> 执行 14 步 -> 重新规划
```

### 修改后

```text
预测 14 步 -> 默认执行前 4 步 -> 重新观测和规划
```

`--horizon` 仍可调整，但不允许超过模型 horizon 14。该修改无需重新训练，可减少开环漂移。

## 11. 单步夹爪噪声会立即触发开合翻转

### 遇到的问题

原客户端直接用 `0.5` 阈值逐步二值化夹爪。一次预测噪声就可能在抬升或搬运时打开夹爪并掉落物体。

### 修改文件

```text
mujoco_pickplace/eval_policy_client.py
tests/test_short_term_memory.py
```

### 修改前

```python
action[6] = 1.0 if action[6] >= 0.5 else 0.0
```

### 修改后

- 小于等于 0.4 视为强关闭请求，大于等于 0.6 视为强打开请求。
- 0.4–0.6 区间保持上一命令。
- 默认连续 2 步同方向请求后才切换；`--gripper-debounce-steps 1` 可关闭去抖。
- 只读取模型命令，不读取 MuJoCo 的 `attached` 等隐藏真值。

### 验证结果

单步反向请求不会改变夹爪状态，连续两步请求会正常切换。

## 12. 数据损坏时静默换成随机样本

### 遇到的问题

缓存或视频读取失败时，Dataset 会递归返回一个随机样本。训练不会停止，但样本分布和复现性已经改变，错误也被隐藏。

### 修改文件

```text
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

```python
return self[random.randint(0, len(self.data) - 1)]
```

### 修改后

直接抛出包含缓存或视频路径的 `RuntimeError`，由训练入口报告真实数据问题。

## 13. BF16 时间层可能没有真实权重更新

### 遇到的问题

旧 checkpoint 的视觉权重为 BF16。普通 AdamW 在 `1e-5` 学习率下可能把小更新舍入为零，表面完成反向但时间层没有学习。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1.py
Evo-1/Evo_1/scripts/train.py
```

### 修改后

- 只将第 4/8/12/16/20 层的 attention、norm 和 layer scale 保持 FP32 主参数。
- LoRA 参数保持 FP32；独立验证脚本检查第 4 层 QKV LoRA-B 在 optimizer step 后真实变化。
- 其余 VLM 仍冻结并保持 BF16。

### 验证结果

第 4 层 QKV 最大绝对变化为 `1.0013580e-05`，五个时间层梯度均有限且非零。

## 14. episode 开头的无效历史仍进入 ViT

### 遇到的问题

`history_mask=False` 虽能阻止无效历史影响当前注意力，但对应图像仍会执行 resize、patch embedding 和 ViT。只有当前帧有效时，旧实现还会执行没有真实历史的时间分支。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py
```

### 修改前

固定用 K=6 图像进入 ViT，再依赖时间掩码隔离左填充。

### 修改后

- 只允许 `False...False, True...True` 的左填充掩码，拒绝中间空洞。
- 无效前缀在批量 resize 和 ViT 之前删除。
- 只剩当前帧有效时严格调用原 `extract_feature` 单图路径。
- 有效视觉时间位置仍与状态记忆的 `[-K+1, ..., 0]` 后缀一致。

### 验证结果

新增单图精确路径、左填充预处理压缩和非法空洞掩码测试，全部通过。

## 15. 语言主干保留全部层输出和推理 KV cache

### 遇到的问题

代码只需要语言主干最后一层，却请求 `output_hidden_states=True`；推理也没有显式关闭 KV cache。视觉 token 数不一致时还会捕获异常、截断后继续，可能隐藏 prompt 与图像错位。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
```

### 修改前

```python
output_hidden_states=True
fused_hidden = outputs.hidden_states[-1]
```

token 数量异常时取前 `n_token` 个视觉 embedding 继续运行。

### 修改后

```python
output_hidden_states=False
use_cache=False
fused_hidden = outputs.last_hidden_state
```

prompt 图像 token、`num_tiles_list` 和视觉 embedding 数量必须完全一致，否则立即报错。

### 验证结果

真实 K=6 模型严格加载、前向、反向和 optimizer 更新均通过，融合序列仍为 `[1,277,896]`。

## 16. flow 动作头重复生成无用注意力权重

### 遇到的问题

8 个动作 Transformer 层在每个 flow 积分步中调用 `MultiheadAttention` 时使用默认 `need_weights=True`，会生成注意力权重，但返回后立即丢弃。

### 修改文件

```text
Evo-1/Evo_1/model/action_head/flow_matching.py
```

### 修改前

每次 cross-attention 都计算输出和未使用的权重；每个 flow 步还重复切片、转换和复制相同时间编码表。

### 修改后

```python
self.attn(..., need_weights=False)
```

时间编码表在 flow 循环外准备一次，循环中只索引当前时间并用 `expand` 扩展 batch。

### 验证结果

该算子优化不改变积分步数。当前评估保持 Evo 原始的 50 步；真实 batch 8 推理输出为 `[8,14,24]` 且全部有限。

## 17. 非有限张量或梯度被静默跳过

### 遇到的问题

训练发现 NaN/Inf 时会 `continue`，step 不增加，可能无限跳过坏 batch；梯度范数非有限时仍可能进入 optimizer。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
```

### 修改前

记录日志后继续下一 batch，不保证训练结果有效。

### 修改后

- state、action、融合 token、预测或 loss 非有限时抛出带 step 和张量名的 `FloatingPointError`。
- 裁剪前后梯度范数非有限时禁止 optimizer step 并立即失败。

### 验证结果

真实数据训练入口完成一个 optimizer step，损失 `0.9492`，第 4 层 QKV 最大绝对变化 `1.0013580e-05`，峰值显存 `3621.2 MiB`。

## 18. FlashAttention 的 C++ ABI 失配

### 遇到的问题

旧环境中的 `flash_attn` 扩展要求 `CXXABI_1.3.15`，但运行时加载了较旧的 `libstdc++`，因此模型只能回退到标准 PyTorch attention。

### 修改范围

```text
Conda 环境 Evo1
PI_MEM_README.md
tests/PI_MEM_ENVIRONMENT_RESULT.json
```

### 修改前

只能证明 Python 包存在，不能证明 CUDA 扩展可以完成真实计算。

### 修改后

- `Evo1` 环境使用 `libstdcxx 15.2`，满足扩展的 ABI 需求。
- `flash_attn 2.8.3.post1` 可直接导入。
- 保留模型初始化时的扩展探测和安全回退；环境以后发生变化时不会导致整个模型初始化失败。

### 验证结果

在 RTX 5060 Laptop GPU 上执行 BF16、causal 模式的 FlashAttention，输入为 `[1,128,8,64]`。前向输出与 Q 梯度均全部有限，反向传播通过。随后真实 K=6 训练入口也在未触发回退的情况下完成。

## 19. DeepSpeed CPU optimizer offload 缺少链接库

### 遇到的问题

编译器和 Ninja 实际已经存在，但 CPUAdam 第一次 JIT 链接失败：DeepSpeed 只搜索 Conda 标准库目录，而 `libcurand`、`libcufile` 当时只位于 pip 包的私有目录。`libaio` 也缺失，导致异步 I/O 兼容性检查失败。

### 修改文件与环境

```text
Conda 环境 Evo1
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/ds_config_pi_mem.json
tests/PI_MEM_ENVIRONMENT_RESULT.json
```

### 修改前

```text
CPUAdam link: cannot find -lcurand
CUDA link probe: cannot find -lcufile
async_io: incompatible because libaio is missing
```

### 修改后

- 在 `Evo1` 环境内安装 `libcurand-dev 10.3.9.90`、`libcufile-dev 1.13.1.3` 和 `libaio 0.3.113`，版本与 CUDA 12.8 对齐。
- CPUAdam JIT 缓存固定到当前 Python 环境的 `var/torch_extensions`，不写用户全局缓存，也不设置系统级环境变量。
- 训练启动后记录 DeepSpeed 外层优化器、基础优化器和 ZeRO stage，防止配置未生效却被误判为已启用。
- DeepSpeed 自己管理模型前向的 BF16，不再给已由 DeepSpeed 接管的 forward 额外套一层 PyTorch autocast；视觉嵌入的直接方法调用仍保留所需 autocast。
- `resume_pretrain` 只加载模型权重时，日志改为准确显示 `model weights only`，不再错误声称加载了 optimizer state。
- README 的启动命令显式使用 `--dynamo_backend no`，避免 Accelerate 为未指定的默认值重复告警。

### 验证结果

- 独立 CPUAdam JIT 成功，参数从 `[1.0,-2.0]` 更新为约 `[0.9,-1.9]`。
- DeepSpeed 环境报告中 Ninja、CPUAdam 和 async I/O 均兼容。
- 真实 Accelerate 启动显示：`DeepSpeedZeroOptimizer`、`DeepSpeedCPUAdam`、ZeRO stage 2。
- 从 2.8GB checkpoint 严格恢复模型权重，复用 64,185 个 K=6 窗口并完成一次真实训练更新。
- 单步损失 `0.6562`；第 4 层 QKV 最大绝对变化 `1.52587891e-05`；峰值已分配显存 `3008.5 MiB`。

## 20. 默认 LoRA 微调

### 遇到的问题

原配置需要训练 92.24M 动作头参数和 21.01M 时间 ViT 参数，共 113.25M。即使使用 ZeRO-2 和 CPU optimizer offload，梯度、优化器通信与可训练权重仍会增加显存和训练开销。

### 修改文件

```text
Evo-1/Evo_1/model/lora.py
Evo-1/Evo_1/scripts/Evo1.py
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/scripts/Evo1_server.py
tests/test_lora.py
PI_MEM_README.md
```

### 修改后

- 训练 CLI 默认启用 `rank=8`、`alpha=16`、`dropout=0` 的 LoRA。
- 默认目标为 `vision,action`：五个 π-MEM 时间 ViT 层的 QKV/投影，以及动作头内的 MHA、FFN、状态/动作编码器和输出投影。
- 基础大矩阵冻结；训练约 1.49M 参数，其中动作头 1.21M、时间 ViT 约 0.28M；包含 LoRA 以及目标范围内的偏置、LayerNorm 和时间层缩放参数。
- LoRA-A 使用 Kaiming 初始化，LoRA-B 从零初始化，保证注入后首次前向与基础模型一致。
- LoRA 参数保持 FP32 且不施加 weight decay，避免小学习率下 BF16 更新消失。
- `--resume --resume_pretrain` 可以从不含 LoRA 键的旧阶段一 checkpoint 初始化；LoRA checkpoint 的正常续训仍严格加载模型和优化器状态。
- `--no-use_lora` 完整保留原来的全参数动作头、选择性时间 ViT 和 `--finetune_vlm` 路径。
- 推理服务在严格加载 LoRA checkpoint 后合并增量权重，不保留额外低秩前向算子。

### 验证

- LoRA Linear 和 MultiheadAttention 在 LoRA-B 为零时与原模块输出一致。
- MHA 覆盖跨注意力、padding mask、反向梯度与推理合并。
- 旧基础 state dict 只缺少预期的 `lora_*` 键；LoRA state dict 可严格重载。
- 动作头从 92.24M 全参数训练降为约 1.21M 个训练参数。

## 21. batch 8、FlashAttention 与阶段一恢复

### 问题

DeepSpeed 包装后直接读取不含 LoRA 的阶段一 checkpoint，会按新冻结参数表解释旧分片，并在动作头 MHA 的 `in_proj_weight` 上触发 `KeyError`。评估服务还曾把 Evo 原始 50 个流匹配求解步降到 32，可能降低动作质量。

### 修复

- `--resume_pretrain` 在 DeepSpeed 包装前载入阶段一模型权重；同一 π-MEM 任务续训仍在包装后恢复模型、优化器和调度器。
- 评估恢复 Evo 原始 `num_inference_timesteps=50`。
- 删除自动清理高步数 checkpoint 以及仅供临时验证的训练参数；训练不会擅自删除已有 checkpoint。
- 正式命令改为物理 `batch_size=8`、`max_steps=16000`、`warmup_steps=500`、`ckpt_interval=2000`。

### 结果

RTX 5060 Laptop 8GB 上，`torch 2.12.0.dev + CUDA 12.8 + flash-attn 2.8.3.post1 + DeepSpeed ZeRO-2 CPU offload` 使用真实阶段一 checkpoint、K=6、448×448、物理 batch 8 完成前向、反向和优化器更新；loss 为 `0.7695`，峰值已分配显存 `4918.1 MiB`。独立评估前向输出为 `[8,14,24]` 且全部有限。

## 22. LoRA 后取消 CPU optimizer offload，并批量化 VLM

### 问题

第 19 节的 CPUAdam 方案针对 113.25M 可训练参数有效；默认 LoRA 后只训练约 1.49M 参数，Adam 状态已经缩小到十几 MB，继续 offload 会保留 CPUAdam、同步和 GPU/CPU 传输开销。训练循环还把物理 batch 8 拆成 8 次 B=1 的视觉语言前向，没有利用大矩阵的批量吞吐。

### 修改

- `ds_config_pi_mem.json` 保留 BF16 与 ZeRO-2，删除 `offload_optimizer`。
- `train.py` 每个物理 batch 只调用一次 `get_vl_embeddings_batch()`。
- `internvl3_embedder.py` 新增 `[B,K,V,C,H,W]` 批量预处理、批量视觉 token 注入、批量 tokenizer padding 和一次语言主干前向。
- `temporal_vision_encoder.py` 为时间注意力增加独立 batch 轴；每个样本使用自己的左填充 mask，禁止跨样本注意力。
- 单样本服务和 K=1 原 Evo 路径保持不变；模型参数结构不变，旧 checkpoint 兼容。

### 论文约束

批量化保持 MEM 原文的 K=6、1 秒 stride、每 4 层同 patch 因果时间注意力、`e(0)=0` 固定正弦位置编码、上层只保留当前帧以及 K=1 与原图像编码器一致。MINT `evo1-flash` 只用于参考 batch-aware VLM 接口，不替代 π-MEM 时间机制。

### 验证

- 35 项 LoRA、短期记忆、批量时间 mask、完整 embedder 等价和视觉 token 注入测试通过。
- 批量 temporal 输出与逐样本输出在不同左填充长度下数值一致。
- 当前 Windows 图形负载占用约 6.4/8.15 GiB GPU 显存，真实 K=6、batch 8 的新路径性能验证暂未执行；不能沿用第 21 节旧串行路径的显存和速度数字作为本节结论。

## 23. batched K=6 单卡反向峰值与完整断点恢复

### 问题

第 22 节完成 batch-aware VLM 后，单张 8 GB GPU 使用无 CPU offload 的 DeepSpeed ZeRO-2 仍在反向阶段 OOM；改成普通 AdamW 后，整块 checkpoint 时间 ViT 层也会在重算 attention 与宽 MLP 时产生重叠峰值。另有两个断点语义问题：`step_final` 曾把字符串 `final` 作为进度写入，普通 checkpoint loader 也没有把已保存的 scheduler state 返回给训练循环。`resume_pretrain` 还错误继承阶段一的 `best_loss`，可能阻止阶段二生成自己的 `step_best`。

### 修改

- 单卡正式入口改为 Accelerate BF16 + CUDA fused AdamW；DeepSpeed 配置继续保留，但只用于以后多卡训练。
- LoRA 参数仍以 FP32 保存并由 AdamW 更新；适配器矩阵乘法使用 BF16 activation dtype，避免把 `[B,K,N,C]` 激活和 QKV 增量整体扩成 FP32。
- 时间 ViT 层把 attention residual 与 MLP residual 分成两个独立的 non-reentrant gradient checkpoint，数学顺序和 π-MEM 因果 mask 不变。
- `step_best`、`step_final` 的 tag 与数值 `global_step` 分离；普通 checkpoint 在同一个模型状态文件中保存 optimizer 和 scheduler。
- 普通续训恢复模型、optimizer、数值 step、scheduler；`resume_pretrain` 只取阶段一权重并将新任务的 `best_loss` 重置为正无穷。
- 训练日志增加 CUDA 当前/峰值 allocated 和 reserved 统计，便于长训练发现显存漂移。

### 验证

- 38 项 LoRA、π-MEM、批量/逐样本等价、跨样本 mask 隔离和 BF16 LoRA 梯度测试通过。
- RTX 5060 Laptop 8 GB 上，真实 64,185 个 K=6 窗口、InternVL3-1B、448×448、物理 batch 8、双塔 Flash-Attn 完成生产入口前向、反向、梯度裁剪和 fused AdamW 更新，未 OOM。
- 首次短步从阶段一 `step_best` 初始化，完成 step 0 并保存普通 `step_final`；文件含 `module`、199 个非空 optimizer state、数值 `step=1` 和 scheduler `last_epoch=1`。
- 随后从该 `step_final` 正常续训，日志明确显示 optimizer、step=1、scheduler 全部恢复，完成 step 1 后保存 `step=2`、scheduler `last_epoch=2`；两次训练均 exit code 0。
- `Evo1_server` 严格载入新 checkpoint，合并 42 个 LoRA 模块，确认 vision=`flash-attn`、language=`flash_attention_2`，以 K=6 和 50 个 flow steps 输出有限的 `[14,24]` 动作；exit code 0。

以上只证明当前训练、保存、恢复和推理路径可运行；完整 16000 步稳定性与 MuJoCo 闭环成功率仍需正式训练后判断。第 22 节的“35 项、GPU 待验证”是当时状态，当前以本节的 38 项和真实生产验证为准。

## 24. 分段 gradient checkpoint 数值等价补充

新增独立回归测试，将相同的 batched π-MEM 小模型分别运行“时间 attention 与 MLP 分段 checkpoint”和“不使用 checkpoint”两条路径。两者的输出、输入图像梯度，以及八个 ViT 层全部参数梯度均在容差内一致。最新完整回归为 `39 passed in 2.99s`；该结果补充第 23 节当时记录的 38 项，不改写已有历史内容。

## 25. 单卡生产路径显存实测补充

从数值 step 2 的普通 checkpoint 再续训一步，optimizer 与 scheduler 均恢复成功。K=6、448×448、物理 batch 8、双塔 Flash-Attn、BF16、fused AdamW 的 step 2 loss 为 `0.8446`；CUDA 峰值 allocated `6366.9 MiB`、reserved `6878.0 MiB`，训练与 `step_final` 覆盖保存均 exit code 0。最终临时 checkpoint 的数值 step 与 scheduler `last_epoch` 均为 3。该实测表明当前 8 GB GPU 有约 1.27 GiB 总显存余量，不需要恢复 CPU optimizer offload；长训练仍应观察日志中的峰值是否发生异常增长。

## 26. warmup 低损失阻止 step_best 落盘

旧逻辑从 step 0 起更新内存中的 `best_loss`，但只有 step 大于 1000 才允许保存。如果 warmup 内出现偶然低值且之后未被打破，训练可能没有任何 `step_best`。现改为只从 `max(1000, warmup_steps)` 开始参与 best 比较；第一个有效比较点可以正常建立并保存 `step_best`。checkpoint 保留策略未改变，仍不自动删除任何 tag。

## 27. 云端默认 Accelerate 配置导致单卡 LoRA 误启 DeepSpeed

### 问题

正式命令没有写 `--use_deepspeed`，但云端用户目录中的 Accelerate 默认配置可能仍把 `distributed_type` 设为 DeepSpeed。此时启动日志会出现 `DeepSpeedZeroOptimizer` 和 ZeRO stage 2，单卡无法获得参数分片收益，反而因包装器与通信缓冲在 batch 8 反向阶段 OOM。

### 修改

- 不新增 Accelerate YAML，也不修改只负责数据路径的 `dataset/config.yaml`。
- `train.py` 在构造 `Accelerator` 前检查启动环境：单 GPU、默认 LoRA 且检测到 DeepSpeed 时，自动清除 DeepSpeed 配置并回到 Accelerate + CUDA fused AdamW。
- 正式命令前显式写 `ACCELERATE_USE_DEEPSPEED=false` 表达单卡路径；脚本检查负责拦截云端默认 YAML 或旧命令重新注入的 DeepSpeed。
- 仅为有意诊断保留 `--allow_single_gpu_deepspeed`；正式单卡训练不得使用。

### 验证

- 使用旧式 `--use_deepspeed --deepspeed_config_file ds_config_pi_mem.json` 启动，日志先报告自动回退，随后确认 `AcceleratedOptimizer`、`AdamW`、`DeepSpeed ZeRO stage=None`。
- InternVL3-1B、双塔 FlashAttention、K=6、448×448、物理 batch 8 完成前向、反向、梯度裁剪、optimizer step 和 `step_final` 保存；峰值 `6355.6 MiB allocated / 6868.0 MiB reserved`，无 OOM。
- 从该 `step_final` 恢复 optimizer、scheduler 和 `step=1` 后再训练一步，峰值 `6366.9 MiB allocated / 6878.0 MiB reserved`，保存成功。
- 完整本地回归为 `39 passed`。当前修复无需恢复 CPU optimizer offload，也未降低 batch、K、图像分辨率或模型层数。

## 28. step_best 恢复 MINT 原始判定顺序

第 26 节曾把 `best_loss` 的比较也延迟到 `max(1000, warmup_steps)` 之后。这样在第一个允许保存的区间开始时阈值仍为正无穷，连续遇到更低的 batch loss 会反复覆盖 `step_best`。根据用户要求，现以本节取代第 26 节的当前方案，恢复 MINT 原始语义：所有全局 step 都参与单 batch `best_loss` 比较，只把实际保存限制在 `step > max(1000, warmup_steps)`。训练跨 epoch 时全局 `step` 和 `best_loss` 均不重置，所以只有整个训练开始时跳过一次保存区间，后续 epoch 开头不会再次排除 1000 步。best 指标仍是单 batch loss，没有改成平均 loss。

## 29. 云端 checkpoint 下载后无法启动评估

### 问题

阶段二 checkpoint 的 `config.json` 保存了训练机器上的绝对 VLM 路径 `/mnt/workspace/modelscope_cache/OpenGVLab/InternVL3-1B`。checkpoint 下载到本机后该目录不存在，Transformers 在模型初始化阶段直接报 `Incorrect path_or_model_id`。同时，评估服务原默认 checkpoint 仍指向阶段一，README 未显式指定阶段二目录。

### 修复

- 服务端默认 checkpoint 改为仓库内 `ckpt/evo1_mujoco_pickplace_stage2/step_best`，README 同时显式写出该路径。
- 服务端启动前检查 `config.json`、`norm_stats.json` 和模型状态文件是否完整。
- checkpoint 中的绝对路径存在时继续原样使用；若确认是已迁移的 `OpenGVLab/InternVL3-1B` 云缓存路径，则回退为可移植模型 ID。
- 新增 `--vlm-name`，用于显式指定其他本地模型目录或 Hub ID；未知的失效绝对路径会直接给出可操作错误，不静默猜测。

### 结果

- 阶段二 `step_final` 实际元数据为 `step=16000`、scheduler `last_epoch=16000`，含 199 个 optimizer state；`step_2000` 至 `step_16000`、`step_best` 和 `step_final` 均存在。
- 默认服务命令成功定位阶段二 `step_best`，严格加载 checkpoint，合并 42 个 LoRA 模块，并保持 CUDA 与双塔 FlashAttention。
- MuJoCo 客户端按 K=6、stride=5 发送一次真实 WebSocket 请求，服务端返回 finite 的 `[14,24]` 动作并由环境执行一步；客户端与服务端均正常退出验证流程。
- 完整轻量回归为 `42 passed`。以上确认评估链路可启动与通信，不代表正式多 episode 的任务成功率。
