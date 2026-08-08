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

## 10. 误删 Evo-1 MuJoCo checkpoint 后的恢复

### 问题

`Evo1_server.py` 当前从以下路径加载 MuJoCo pick-and-place checkpoint：

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best
```

该目录下由训练生成的 checkpoint 文件被意外删除，导致启动 server 时出现以下错误：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best/config.json'
```

缺失的主要文件包括：

```text
config.json
norm_stats.json
checkpoint.json
mp_rank_00_model_states.pt
```

### 修改位置

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1
```

本次恢复没有修改 Evo-1 的源代码。缺失目录属于训练生成文件，并且被 `.gitignore` 排除，因此无法通过 git 历史直接恢复

### 修改后

使用当前 MuJoCo 数据集和 Evo-1 原有的 DeepSpeed 训练流程，重新启动正式的 Stage 1 训练：

```text
run_name=Evo1_mujoco_pickplace_stage1
max_steps=5000
batch_size=16
warmup_steps=1000
ckpt_interval=2500
horizon=50
finetune_action_head=true
```
## Current Canonical State (2026-08-06)

The current project no longer uses an npz intermediate dataset pipeline. The removed files include `mujoco_pickplace/collect_data_npz.py` and `mujoco_pickplace/convert_to_evo_lerobot.py`.

Current data flow:

```text
mujoco_pickplace/collect_data.py
-> /home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace
-> Evo-1/Evo_1/dataset/config.yaml
-> Evo-1/Evo_1/scripts/train.py
```

Canonical dataset layout:

```text
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/data/chunk-000/episode_*.parquet
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/videos/chunk-000/observation.images.front/episode_*.mp4
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/videos/chunk-000/observation.images.overhead/episode_*.mp4
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/videos/chunk-000/observation.images.wrist/episode_*.mp4
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/meta/dataset.json
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/meta/tasks.jsonl
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/meta/episodes.jsonl
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/meta/episodes_stats.jsonl
/home/user/mujoco+evo/Mujoco_training_dataset/cache/mujoco_pickplace/meta/stats.json
```

Current training command:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch --use_deepspeed --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py
```

Do not use `python scripts/train.py` for training unless `train.py` is changed back to a non-DeepSpeed save path. The current Evo-1-style checkpoint save calls `model_engine.save_checkpoint(...)`, so `--use_deepspeed` is required.

Current inference/evaluation commands:

```bash
conda activate Evo1
cd /home/user/mujoco+evo/Evo-1/Evo_1
python scripts/Evo1_server.py
```

```bash
conda activate mujoco
cd /home/user/mujoco+evo/mujoco_pickplace
python eval_policy_client.py
```

Default checkpoint used by the server:

```text
/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best
```

