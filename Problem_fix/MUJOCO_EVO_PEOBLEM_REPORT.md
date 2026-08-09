# MuJoCo + Evo-1 全流程问题修复报告

## 1. 没有 demonstration 数据采集入口

### 遇到的问题

Evo-1 训练需要 demonstration 数据。最初没有稳定的 MuJoCo 数据采集入口，无法生成后续转换和训练所需的 episode 数据

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
```

### 修改前

没有统一输出 raw episode 的脚本和目录约定

### 修改后

用 `scripted_expert(obs)` 采集轨迹：

```python
for ep in trange(NUM_EPISODES):
    obs = env.reset(seed=ep)
    images, states, actions = [], [], []

    for _ in range(MAX_STEPS):
        action = scripted_expert(obs)
        images.append(obs["image"])
        states.append(obs["state"])
        actions.append(action)
        obs, done = env.step(action)
        if done and len(actions) > 30:
            break
```

保存为 `.npz`：

```python
np.savez_compressed(
    RAW_DIR / f"episode_{ep:06d}.npz",
    images=np.asarray(images, dtype=np.uint8),
    states=np.asarray(states, dtype=np.float32),
    actions=np.asarray(actions, dtype=np.float32),
)
```

## 2. MuJoCo raw `.npz` 不能直接被 Evo-1 训练读取

### 遇到的问题

MuJoCo 采出来的是：

```text
images
states
actions
```

Evo-1 原训练 loader 需要 LeRobot/Evo-1 风格目录，包括 parquet、video 和 meta 文件。只生成 `.npz` 时，`Evo-1/Evo_1/scripts/train.py` 不能直接用这些数据

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/removed npz conversion script
```

### 修改前

只有：

```text
Mujoco_training_dataset/cache/mujoco_pickplace/data/chunk-000/episode_*.parquet
```

### 修改后

新增转换为 Evo-1 数据结构：

```python
df = pd.DataFrame({
    "index": np.arange(length, dtype=np.int64),
    "episode_index": np.full(length, ep_idx, dtype=np.int64),
    "frame_index": np.arange(length, dtype=np.int64),
    "timestamp": np.arange(length, dtype=np.float32) / FPS,
    "task_index": np.zeros(length, dtype=np.int64),
    "observation.state": [x.astype(np.float32).tolist() for x in states],
    "action": [x.astype(np.float32).tolist() for x in actions],
})

df.to_parquet(data_dir / f"{episode_name}.parquet")
```

写视频：

```python
imageio.mimsave(
    video_dir / f"{episode_name}.mp4",
    images,
    fps=FPS,
    macro_block_size=1,
)
```

写 meta：

```python
write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_rows)
```

## 3. MuJoCo state/action 维度和 Evo-1 固定维度不一致

### 遇到的问题

MuJoCo 原始维度：

```text
state: 10
  eef_xyz + cube_xyz + goal_xyz + gripper

action: 4
  dx + dy + dz + gripper
```

Evo-1 接口维度：

```text
state: 24
action: 24
action chunk: [50, 24]
```

dataset loader、训练 batch、server 推理和 MuJoCo 执行动作会 shape 或语义不一致

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
```

### 修改前

MuJoCo 只有 4 维动作，但 Evo-1 返回 24 维 padded 动作，没有明确只取哪些维度

### 修改后

配置保留 Evo-1 最大维度：

```yaml
max_action_dim: 24
max_state_dim: 24
```

推理 payload 告诉 Evo-1 只有前 4 维 action 有效：

```python
"action_mask": [1, 1, 1, 1] + [0] * 20
```

执行时只取前 4 维送给 MuJoCo：

```python
action = action_chunk[action_idx, :4].astype(np.float32)
obs, done = env.step(action)
```

## 4. Evo-1 DeepSpeed BF16 训练中 action head dtype 不一致

### 遇到的问题

使用 Evo-1 DeepSpeed/BF16 训练时，`flow_matching.py` 中部分张量只对齐了 device，没有对齐 dtype，触发 Float/BFloat16 混用错误，导致训练中断

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/model/action_head/flow_matching.py
```

### 修改前

位置编码类似：

```python
pos_enc = self.pos_encoding(H).to(out.device)
```

### 修改后

改成同时对齐 device 和 dtype：

```python
pos_enc = self.pos_encoding(H).to(device=out.device, dtype=out.dtype)
```

同类 dtype 对齐也用于 action head 内参与 BF16 计算的 time embedding / FFN 输入

## 5. Evo-1 server 期望 payload 和 MuJoCo observation 不一致

### 遇到的问题

Evo-1 server 期望的 JSON observation 包含多视角图像、mask、state、action mask、prompt；MuJoCo 环境天然只有一路 camera 图像和一个 10 维 state

如果直接把 MuJoCo obs 发过去，server 输入格式不对

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
```

### 修改后

把一路真实 MuJoCo 图像放到 image_1，另外两路补零：

```python
image = obs["image"].astype(np.uint8)
zero_image = np.zeros_like(image)
state = obs["state"].astype(np.float32)

return {
    "image": [
        image.tolist(),
        zero_image.tolist(),
        zero_image.tolist(),
    ],
    "image_mask": [1, 0, 0],
    "state": state.astype(float).tolist(),
    "action_mask": [1, 1, 1, 1] + [0] * 20,
    "prompt": PROMPT,
}
```

## 6. 视频保存强依赖 OpenCV

### 遇到的问题

本次发现 `eval_policy_client.py` 的视频保存强依赖库CV2而不是与原libero相同的imageio

LIBERO client 保存视频使用的是：

```python
import imageio
imageio.mimsave(filepath, frames, fps=fps)
```

而 MuJoCo client 之前使用的是 OpenCV：

```python
import cv2
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
```

这导致两个问题：

```text
保存视频也依赖 opencv-python，但 Evo-1 + LIBERO 保存视频并不需要 OpenCV
```

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
/home/user/mujoco+evo/README.md
/home/user/mujoco+evo/mujoco_pickplace/README.md
```

### 修改前

`eval_policy_client.py` 顶部直接 import OpenCV：

```python
import cv2
```

视频保存使用 OpenCV：

```python
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
for frame in frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()
```

### 修改后

保存视频改成和 LIBERO 类似的 `imageio`：

```python
def save_video(frames, path, fps=20):
    if not frames:
        return
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved: {path} ({len(frames)} frames)", flush=True)
```

OpenCV 只在实时显示时才需要：

```python
def maybe_show(frame, enabled):
    if not enabled:
        return False
    try:
        import cv2
        cv2.imshow("MuJoCo PickPlace", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return True
    except Exception as exc:
        print(f"render disabled: {exc}", flush=True)
        return False
```

## 7. 训练效果不够好

### 遇到的问题

进行测试的时候发现训练数据一直都不太正常，training_loss一直处于偏高状态

### 修改文件

```text
/home/user/mujoco+evo/Evo_1/Evo-1/train.py
```

### 修改后

修改dataset_config_path位置，提高step和dropout

```python
    parser.add_argument("--dataset_config_path", type=str, default="/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=2500)
    parser.add_argument("--save_dir", type=str, default="/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1")
```

## 8. 视频画面问题

### 问题

```text
视频画面不是单纯偏暗，而是经常只看到桌角，看不到 cube / goal / eef 等任务主体
```

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 修改内容

`pick_place_env.py` 中调整了 MuJoCo XML 的光照、floor 和相机，让任务主体进入画面：

```xml
<headlight diffuse="0.65 0.65 0.65" ambient="0.22 0.22 0.22" specular="0.12 0.12 0.12"/>
<light name="main_light" pos="0.1 -0.55 2.8" dir="0 0 -1" diffuse="0.7 0.7 0.7" ambient="0.18 0.18 0.18"/>
<light name="fill_light" pos="-0.9 0.8 1.5" dir="0.4 -0.2 -1" diffuse="0.22 0.22 0.22" ambient="0.04 0.04 0.04"/>
<camera name="front" pos="0.55 -0.95 0.95" xyaxes="0.94 0.32 0 -0.22 0.66 0.71" fovy="55"/>
```

## 9. 成功率过低的评测侧问题修正

### 问题

之前出现：

```text
Total Successful Episodes: 1/10
Average Steps: 273.40
success_rate=0.100
```

```text
1. 默认 horizon=15，一次连续执行 15 个旧动作，反馈太慢
2. MuJoCo render 返回 RGB，但 server 端用 OpenCV 按 BGR 转 RGB，直接发送会造成颜色通道错位
```

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
```

### 修改内容

```python
ACTION_HORIZON = 1
```

发送给 server 的图像：

```python
"image": [
    image[..., ::-1].tolist(),
    zero_image.tolist(),
    zero_image.tolist(),
]
```

## 10. 夹爪稳定性

### 遇到的问题

夹爪在搬运阶段会因为短时开指令抖动而误松，物体容易在中途掉落。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 修改前

夹爪处理也比较容易受单帧抖动影响，评测端只做了很轻的阈值滞回：

```python
if raw_gripper > 0.7:
    return 1.0
if raw_gripper < 0.3:
    return 0.0
```

环境端只要 `gripper > 0.8` 就会立刻解除抓取：

```python
if self.gripper > 0.8:
    self.attached = False
```

### 修改后

评测端把动作重规划频率提高到每步一次：

```python
ACTION_HORIZON = 1
```

夹爪改成带确认计数的非作弊滤波，避免搬运中被一帧开指令带偏：

```python
def update_gripper_state(raw_gripper, prev_gripper, open_count, close_count):
    ...
```

环境端增加 `release_counter`，只有连续 3 帧明确开才真正释放 `attached`，这样不会因为短时抖动把物体松掉：

```python
if self.attached:
    if self.gripper > 0.8:
        self.release_counter += 1
    else:
        self.release_counter = 0

    if self.release_counter >= 3:
        self.attached = False
```

## 11. 排查流程（按"先数据、再开环、后闭环"）

### 11.1 先确认训练数据采集没问题

在 WSL 里用实际环境（`mujoco` conda env, `MUJOCO_GL=egl`）逐条动作回放，量化轨迹：

- 原环境：单条命令位移为 0 时，eef 每步仍漂移 **0.014–0.070**，根本停不住；命令 +0.02 x，实际位移甚至可能为负，0.6s 子步内振幅达命令的 **2.4 倍**。
- 结论：控制环节欠阻尼，采集到的专家轨迹本身就抖，x 方向速度翻转很多。
- 修复后：速度更贴近命令，整条轨迹的反向翻转显著减少。

结论：**采集入口的第一个 bug 在控制层**，不修控制，采出来的数据就是抖的。

### 11.2 开环测试模型动作是否本身抖动

写 `eval_open_loop.py`：一次推理拿到 action chunk，不回传观测整段执行，并统计 action chunk 内部平滑度。

对旧模型（旧抖动数据 + 2000 步）：

```text
action_jitter(mean|dd|) = 0.053 ~ 0.128
dx 连续方向翻转 = 2~6 / 9
开环成功率 0/6
```

结论：**模型输出本身就在抖**，不是单纯执行问题。

## 12. 控制层修复（pick_place_env.py）

| 项 | 原值 | 新值 | 说明 |
|---|---|---|---|
| joint damping | 4.0 | **40.0** | 临界阻尼，消除极限环震荡 |
| nstep | 30 | **100** | 闭环更紧，动作到位更稳定 |

参数扫描结果表明，较高 damping 配合更细的步进可以明显压低漂移。

另新增 `PickPlaceEnv.step_video(action, frames_per_step)`：把物理推进分摊到多帧渲染，eval 视频从“跳帧式快放”变成接近实时、平滑。

## 13. 数据层（collect_data.py / removed npz conversion script）

- `collect_data.py`：加 argparse 和质量门，过滤掉失败、过短、过长、动作异常的数据。
- 重新采集了更干净的一批 episode，数据动作的 mean-abs-diff 明显下降。
- 去掉旧的 `.npz` 转换路径，统一到 Evo-1 直接可读的标准数据结构。
- 新数据集路径保持在 `Mujoco_training_dataset/cache/mujoco_pickplace`。

### 重要坑：LeRobot 窗口缓存

缓存路径 key 用的是 config 里的 dataset 名字，不是数据目录路径。换数据集后若不清缓存，loader 会静默复用旧窗口样本。重训前必须清理相关缓存。

## 14. 第二轮视觉 / 夹爪整改

针对后续反馈，继续重做了 `pick_place_env.py` 的几个关键点：

- 手指可视化：不再是一个模糊的大红球，而是更像真实夹爪的两指表示。
- 夹持接触：让 cube 与手指的空间关系更合理，不再看起来像悬浮。
- IK 朝向约束：让手腕姿态更符合抓取习惯，减少“看起来能抓但实际上夹不住”的情况。
- wrist 相机：改成更稳定的跟踪方式，避免穿模或视角丢失。
- 光照：提升可见性，方便判断是否真的抓住了物体。

这一轮的经验是：**模型看上去“差一点能抓住”，很多时候不是模型懂了，而是几何关系、抓取姿态和评测闭环还没对齐**。
