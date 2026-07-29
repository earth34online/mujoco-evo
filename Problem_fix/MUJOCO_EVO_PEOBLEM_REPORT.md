# MuJoCo + Evo-1 全流程问题修复报告

## 1. 没有 demonstration 数据采集入口

### 遇到的问题

Evo-1 训练需要 demonstration 数据。最初没有稳定的 MuJoCo 数据采集入口，无法生成后续转换和训练所需的 episode 数据

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
```

### 修改前

没有统一输出 raw episode 的脚本和目录约定。

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
/home/user/mujoco+evo/mujoco_pickplace/convert_to_evo_lerobot.py
```

### 修改前

只有：

```text
Mujoco_training_dataset/raw_mujoco_pickplace/episode_*.npz
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

同类 dtype 对齐也用于 action head 内参与 BF16 计算的 time embedding / FFN 输入。

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

## 6. MuJoCo 有图像渲染，但默认评测不显示窗口

### 遇到的问题

MuJoCo 是可视化仿真环境，没有实时显示

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
```

### 修改前

MuJoCo 环境能 render，但评测默认只把图像发给 server，没有保存或显示 rollout 的明确方式。

### 修改后

增加/保留：

```text
--save-video
--render
```

保存视频：

```python
def save_video(frames, path, fps=20):
    ...
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
```

实时窗口：

```python
cv2.imshow("MuJoCo PickPlace", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
cv2.waitKey(1)
```

## 7. 视频保存强依赖 OpenCV

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
保存视频也依赖 opencv-python，但 Evo-1 + LIBERO 保存视频并不需要 OpenCV。
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
