# MuJoCo + Evo-1 项目问题修复记录

记录项目实际出现的问题、采用的修复方法与对应效果。历史条目中的旧Task3/Task4分别为当前Task2/Task3；旧实现仅说明修复演进，当前使用以README中的入口为准。

## 1. 没有 demonstration 数据采集入口

### 遇到的问题

Evo-1 训练需要 demonstration 数据。最初没有稳定的 MuJoCo 数据采集入口，无法生成后续转换和训练所需的 episode 数据

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
```

### 修改前

没有统一输出 raw episode 的脚本和目录约定

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

### 遇到的问题

```text
视频画面不是单纯偏暗，而是经常只看到桌角，看不到 cube / goal / eef 等任务主体
```

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 使用的解决方法和对应效果

`pick_place_env.py` 中调整了 MuJoCo XML 的光照、floor 和相机，让任务主体进入画面：

```xml
<headlight diffuse="0.65 0.65 0.65" ambient="0.22 0.22 0.22" specular="0.12 0.12 0.12"/>
<light name="main_light" pos="0.1 -0.55 2.8" dir="0 0 -1" diffuse="0.7 0.7 0.7" ambient="0.18 0.18 0.18"/>
<light name="fill_light" pos="-0.9 0.8 1.5" dir="0.4 -0.2 -1" diffuse="0.22 0.22 0.22" ambient="0.04 0.04 0.04"/>
<camera name="front" pos="0.55 -0.95 0.95" xyaxes="0.94 0.32 0 -0.22 0.66 0.71" fovy="55"/>
```

## 9. 成功率过低的评测侧问题修正

### 遇到的问题

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

### 使用的解决方法和对应效果

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

### 使用的解决方法和对应效果

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

## 11. 控制层修复（pick_place_env.py）

### 遇到的问题

早期控制层存在漂移和极限环振荡，评估录像也呈跳帧快放。

### 修改文件

```text
mujoco_pickplace/pick_place_env.py
```

### 修改前

damping为4.0，nstep为30，视频没有均匀覆盖控制周期。

### 使用的解决方法和对应效果

早期控制层将damping调至40.0、nstep调至100，并通过step_video把物理推进分摊到多帧渲染。

降低漂移并改善动作到位；录像接近实时且连续。此项保留的是原控制层修复，后续原生Panda的实际数值设置按对应环境实现执行。

## 12. 数据层（collect_data.py / removed npz conversion script）

### 遇到的问题

失败或异常示范进入数据池，旧转换链与缓存名称又可能使训练读取旧窗口。

### 修改文件

```text
mujoco_pickplace/collect_data.py
mujoco_pickplace/episode_dataset.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

数据入口和质量要求不统一，窗口缓存未充分关联数据源。

### 使用的解决方法和对应效果

采集增加参数与质量门，过滤失败、异常长度及异常动作；默认覆盖所选任务，显式append才追加。去掉旧npz转换路径，直接写训练可读的Parquet、视频和meta。缓存索引以源文件签名判断有效性。

数据保存在cache并由原dataset/config.yaml读取。重新采集后源签名变化会重建派生窗口，不会因为复用dataset名字继续训练旧样本。

## 13. 第二轮视觉 / 夹爪整改

### 遇到的问题

早期夹爪可视化、夹持几何、IK朝向和wrist相机没有与实际抓取对齐。

### 修改文件

```text
mujoco_pickplace/pick_place_env.py
```

### 修改前

夹爪像模糊红球，物体看似悬浮，相机可能穿模或丢失视角。

### 使用的解决方法和对应效果

改为明确两指表示，调整接触空间关系和手腕朝向约束，稳定wrist相机并改善光照。

抓取是否形成真实接触更易判断，减少画面上似乎抓住而物理上未夹持的情况。后续已接入真实Panda XML，本次归并不再次改变外观和场景。

## 14. 训练 loss mask 和有效动作维度

### 遇到的问题

后续真实闭环评测发现，模型并不是完全不会移动到物体附近，而是经常出现：

```text
1. 夹爪到了物体附近但没有稳定闭合
2. 训练数据里明明有 close 标签，模型输出仍然偏向开爪
3. open-loop chunk 里动作有效性弱，闭环第一集甚至会失败
```

旧 checkpoint 的快速诊断结果：

```text
open-loop learned chunk diagnostic: 0/8
closed-loop learned eval: 第一集 0/1 后停止
主要失败现象: gripper 大部分时间保持 open，没有稳定 latch close
```

数据标签排查结果表明问题不在于“数据里没有闭爪动作”：

```text
dataset: /home/user/mujoco+evo/.fix_tmp/dataset_7d_60
episodes: 60
frames: 4377
gripper close label ratio: 69.68%
horizon-14 chunks with at least one close action: 87.5%
```

主要成因不是 wrist 黑线、物体过小，也不是专家标签完全缺失，而是训练 loss 和动作语义没有对齐：

```text
1. flow-matching loss 只 mask 了 pred_velocity，没有 mask target_velocity
2. 无效 / padding 动作维度虽然不应该训练，但 target 端仍然进入 MSE
3. MuJoCo 当前任务实际只执行 dx, dy, dz, gripper
4. 7D 接口中的 droll, dpitch, dyaw 保留是为了接近 LIBERO / Evo-1 风格，但本环境没有真正执行这些旋转控制
5. 如果旋转维度和 padding 维度继续参与 loss，会把训练容量浪费到无意义目标上，并干扰 gripper 学习
```

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/Evo-1/Evo_1/model/action_head/flow_matching.py
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop.py
```

### 使用的解决方法和对应效果

训练时同时 mask `pred_velocity` 和 `target_velocity`：

```python
action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
pred_velocity_mask = pred_velocity * action_mask
target_velocity_mask = target_velocity * action_mask
loss = loss_fn(pred_velocity_mask, target_velocity_mask)
```

dataset loader 增加 `active_action_mask`，让 24D padded action 中只有当前任务真正有效的维度参与训练：

```text
active_action_mask: [1, 1, 1, 0, 0, 0, 1]
```

ps. 训练和评测只激活 dx, dy, dz, gripper 四个真正有效动作维度

## 15. MuJoCo 场景改为独立 XML，并接入真实 Panda 模型

### 遇到的问题

在前面完成控制层、视觉和夹爪的多轮修正后，继续把所有场景内容留在 `pick_place_env.py` 内会导致环境逻辑、机器人资产、灯光、相机和任务物体互相耦合，不方便继续检查真实 Panda 几何和碰撞关系。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
/home/user/mujoco+evo/mujoco_pickplace/pick_place_scene.xml
/home/user/mujoco+evo/mujoco_menagerie/franka_emika_panda/panda.xml
```

### 修改前

- MuJoCo XML 主要由 Python 内部拼接。
- 场景和机器人资产没有清晰边界。
- 修改相机、桌面、cube 或 Panda 几何时容易同时影响环境代码。

### 使用的解决方法和对应效果

- 新增独立的 `pick_place_scene.xml`。
- 场景通过 `<include>` 使用 MuJoCo Menagerie 的 Panda 模型。
- `_load_model()` 在 Panda 资产目录中生成临时 XML，让 mesh 相对路径可以正确解析。
- 模型加载完成后在 `finally` 中删除临时 XML。
- 灯光、floor、table、cube、goal 和 camera 统一放入场景文件，控制与 expert 状态机继续保留在 `pick_place_env.py`。

当前 Panda 模型还在本地增加了：

```xml
<body name="link0" childclass="panda" pos="0 0 -0.10">
...
<camera name="wrist" mode="fixed"
        pos="0.062 0 0.122"
        quat="0.9135 0 0.4067 0"
        fovy="68"/>
```

这一步让 Panda 基座高度和腕部相机更接近当前任务工作区。

## 16. 抓取从运动学跟随改为接触与受限夹持力

### 遇到的问题

前面的夹爪稳定性修改解决了短时开指令，但如果 `attached=True` 后直接把 cube 位姿绑定到 hand，物体仍然属于运动学拖拽，不是真实的 MuJoCo 接触。画面可能看似抓住，实际手指几何和接触却没有闭合。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
/home/user/mujoco+evo/mujoco_pickplace/pick_place_scene.xml
```

### 使用的解决方法和对应效果

- `attached` 由双侧手指接触或严格几何夹持条件触发。
- 几何夹持同时检查夹爪开度、XY 误差和 Z 误差。
- `_apply_grasp_force()` 使用弹簧-阻尼外力维持物体，不直接覆盖 cube pose。
- 夹持力增加重力补偿，并限制最大力：

```text
GRASP_HOLD_FORCE_KP = 55.0
GRASP_HOLD_FORCE_KD = 1.8
GRASP_HOLD_MAX_FORCE = 4.0
```

- 手指碰撞体提高 friction，并调整 `solref/solimp`。
- cube 仍由 MuJoCo contact 和 free joint 决定真实位置。
- 打开夹爪时通过 release counter 解除 attached，避免单帧抖动立即掉落。

成功不再只看 cube 是否短暂进入 goal，而是同时要求：

```text
1. cube XY 位于 goal 半径内
2. cube 回到桌面支撑高度
3. gripper 已打开且 attached=False
4. cube 速度低于 0.020
5. robot 没有与 table 发生不安全接触
6. 上述条件连续维持 8 step
```

## 17. 最终训练 state/action 和相机语义重新统一

### 遇到的问题

前面的历史阶段曾使用 10D 环境 state、4D action 或三路相机。后续为了避免 privileged state、统一 Evo-1 接口并减少不稳定视频，最终又调整为 8D robot state、7D 原始 action 和单路 front camera。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop.py
```

### 使用的解决方法和对应效果

环境内部仍保留 expert 所需的 task state：

```text
obs["state"]
= hand xyz + cube xyz + goal xyz + gripper
```

但写入数据集和发送给 policy 的是：

```text
obs["robot_state"]
= eef xyz(3) + eef axis-angle(3) + finger qpos(2)
= 8D
```

这样 cube/goal 世界坐标只供 scripted expert 和环境成功判定使用，不作为 policy 的 privileged input。

原始 action 最终统一为：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
= 7D
```

其中实际执行维度仍是：

```text
dx, dy, dz, gripper
```

夹爪语义保持：

```text
1 = open
0 = close
```

相机最终统一为：

```python
CAMERAS = ("front",)
```

Evo-1 仍接收 3-view tensor，但只有第一路是真实 front image：

```text
image_mask = [1, 0, 0]
```

## 18. action horizon 尾部多生成一个训练样本

### 遇到的问题

旧 loader 在 episode 尾部复制 `action_horizon` 个最后动作，再滑动窗口。对于“当前 observation 对应未来 horizon action”这个定义，只需要补 `horizon - 1` 行；多补一行会让每个 episode 多出一个窗口。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

```python
padding_rows = pd.concat([last_row] * action_horizon)
```

### 使用的解决方法和对应效果

```python
if action_horizon < 1:
    raise ValueError("action_horizon must be at least 1")

padding_count = action_horizon - 1
```

修正后，当前数据集 24,800 个原始 frame 正好生成 24,800 个 horizon-14 训练样本。

## 19. 训练和推理的 state 归一化顺序不一致

### 遇到的问题

训练 loader 的正确流程是先归一化原生 8D state，再补到 24D；旧 server 则可能先把原始 state 补到 24D，再用不完整统计量归一化。这样前 8 维之外的 padding 值与训练时不一致。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop.py
```

### 使用的解决方法和对应效果

训练路径：

```text
8D raw robot state
-> 使用原生 8D min/max 归一化
-> clamp 到 [-1, 1]
-> padding 16 个 normalized zero
-> 24D state
```

服务端路径：

```text
严格接收 8D raw robot state
-> 使用前 8 维 min/max 归一化
-> padding 16 个 normalized zero
-> 24D state
```

`open_loop.py` 增加 train/server state normalization equivalence 检查，要求最大差异小于 `1e-6`。

## 20. Flow action head 的 BF16 和动态建层问题继续修正

### 遇到的问题

相关问题条目先修了位置编码 dtype，但后续仍发现 category-specific linear、time embedding、context/action tokens 可能处于不同 dtype；旧代码还可能在第一次 forward 时动态创建 projection layer，使参数不进入 optimizer 或 checkpoint。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/model/action_head/flow_matching.py
```

### 使用的解决方法和对应效果

- category-specific linear 输入对齐权重 dtype。
- positional/time embedding 对齐 action head dtype。
- context token 和 action token 在 transformer 前统一 dtype。
- `single_action_proj` 和 `seq_pool_proj` 在 `__init__` 中创建，不在 forward 中动态创建。
- 多 embodiment 权重改用 Xavier 初始化，bias 置零。
- category id 数量与展平 batch 不一致时直接报错。
- 用 `expand` 代替部分 `repeat`，减少无意义内存复制。

## 21. Flow inference 的随机起点、mask shape 和积分步数不统一

### 遇到的问题

后续曾尝试用全零 action 作为 flow 起点来减小随机性，但这与训练时 `Uniform[-1, 1]` noise 不一致；同时旧 action mask reshape 对 batch/horizon shape 有隐含假设。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/model/action_head/flow_matching.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
```

### 使用的解决方法和对应效果

- 最终恢复与训练一致的随机均匀 flow 初始化，不使用全零初始化。
- `_expand_action_mask()` 支持：

```text
[B, D]
[B, H*D]
[B, H, D]
```

- shape 不匹配时直接报错。
- 每个 Euler integration step 前重新应用 mask，最终返回前再应用一次 mask。
- inference timesteps 统一为 32，并检查必须大于 0。
- server 支持可选 `flow_seed`，用于复现实验，但不改变随机起点的分布。

## 22. 图像通道和 resize 路径仍有 train/server 差异

### 遇到的问题

MuJoCo renderer 输出 RGB；client 为匹配 server 的 OpenCV 路径发送 BGR。如果颜色转换次数不一致，蓝色 cube 会被错误换色。旧 server 使用 OpenCV resize，也与训练 loader 的 PIL bicubic resize 不一致。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
/home/user/mujoco+evo/mujoco_pickplace/eval_policy_client.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop.py
```

### 使用的解决方法和对应效果

- client 使用 `[..., ::-1]` 把 RGB 转为 BGR 后发送。
- server 使用 `cv2.COLOR_BGR2RGB` 恢复 RGB。
- server 改用：

```python
transforms.Resize(
    (image_size, image_size),
    interpolation=transforms.InterpolationMode.BICUBIC,
)
```

- open-loop 复用与 server 相同的 image decode 路径。
- 单路 front image 与两个 blank image 统一使用 `image_mask=[1,0,0]`。

## 23. Expert 在抓住 cube 前仍可能提前向上移动

### 遇到的问题

后续专门检查抓取附近 action 时发现，不能只看最终 `20/20`，还必须证明：

```text
descend: dz 始终 <= 0
close 且 attached=False: dz 始终 <= 0
lift: 只有 attached=True 后才出现正 dz
```

旧 expert 在 phase 切换或位置误差下可能在尚未 attached 时出现正 `dz`，形成“靠近 cube 后又抬起”的错误 demonstration。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 使用的解决方法和对应效果

- 增加 `_move_before_attachment()`：

```python
action = self._move(target, gripper)
action[2] = min(float(action[2]), 0.0)
```

- `descend` 只有 XY/Z 都进入抓取容差后才切到 close。
- close 阶段在 `attached=False` 时继续向 grasp target 收敛。
- 距离 grasp depth 小于 20 mm 时，最大下降速度限制为约 `-0.003`。
- 只有 `env.attached=True` 后才进入 lift 并出现正 `dz`。
- close 超过 14 step 仍未抓住时回到 approach。

保存的对话记录给出的回归结果是：

```text
Expert success: 20/20
unsafe_contact=False
```

## 24. `transfer -> lower` 提前切换造成动作瞬间反向

### 遇到的问题

旧逻辑在末端距离 safe pose 仍有明显误差时就进入 lower，可能出现：

```text
上一帧 transfer: dz > 0
下一帧 lower:    dz 接近 -0.012
```

曾考虑插入一个零动作桥接帧，但这会让两个几乎相同 observation 分别对应“停住”和“快速下降”，增加 Evo-1 observation -> action chunk 的条件歧义。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
```

### 使用的解决方法和对应效果

- 不插入零动作桥接帧。
- 先完成 X 对齐：

```text
EXPERT_TRANSFER_X_ALIGN_TOL = 0.020
```

- 再收敛到完整 safe pose。
- 只有同时满足以下条件才进入 lower：

```text
EXPERT_LOWER_XY_TOL = 0.006
EXPERT_LOWER_Z_TOL = 0.010
```

- 数据质量门最终接受：

```text
MAX_ACTION_JUMP = 0.020
```

这一版通过当前 observation 的实际位置误差决定 phase，不依赖隐藏的“刚切 phase”状态。

## 25. 实时 `cube[2]` 使 expert 抓取目标随 cube 倾斜而上移

### 遇到的问题

随机 XY 环境下，失败视频主要表现为机械臂下探不足。进一步对 expert 轨迹做对照后发现，问题并不是 nominal grasp point 固定得太高，而是旧代码每一步都用实时 cube 质心高度计算抓取目标：

```python
grasp_z = cube[2] + self.env.GRASP_OFFSET
```

机械手在下降过程中如果擦碰 cube，使 cube 发生倾斜，cube 质心的实时 Z 会升高。随后 `grasp_z` 也会被一起抬高，导致 expert 的监督目标本身变浅。旧逻辑中首次 close 的最坏高度曾达到 nominal grasp height 上方约 `10.60 mm`。

任务当前只随机 cube/goal 的 XY，不随机 cube 的支撑高度，因此不应使用受接触和倾斜影响的实时 `cube[2]` 作为 expert 抓取高度参考。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 修改前

```python
grasp_z = cube[2] + self.env.GRASP_OFFSET
```

### 使用的解决方法和对应效果

```python
grasp_z = (
    self.env.CUBE_SUPPORT_Z
    + self.env.GRASP_OFFSET
)
```

这样 expert 的抓取目标由任务的固定支撑几何决定，不再跟随已经被擦歪的 cube 质心上移。

这次没有增加额外的负 Z bias，也没有修改 XY tolerance、approach、transfer、lift、release 或成功判定，避免在修复下探监督时破坏当前已经正常的其他阶段。

对修改前后的版本使用相同 100 个随机 seed 做 expert 对照：

```text
修改前：完整成功 98/100，进入 lift 100/100，unsafe 0
修改后：完整成功 98/100，进入 lift 100/100，unsafe 0
```

首次 close 相对 nominal grasp height 的最坏高度变化为：

```text
修改前：+10.60 mm
修改后：+3.99 mm
```

两个失败 seed 仍然是原有的 transfer 失败，没有新增抓取、放置或不安全接触回归。

## 26. `descend -> close` 使用对称 Z 误差会过早允许闭合

### 遇到的问题

旧 descend 判定使用：

```python
z_error = abs(float(hand[2] - target[2]))
```

但当前剩余问题具有明显方向性：危险情况是机械手仍在抓取目标上方时就开始 close，而不是已经略微低于目标。使用绝对值会把“目标上方”和“目标下方”作为同一种误差处理，不能直接表达“不允许在过高位置闭合”。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/pick_place_env.py
```

### 修改前

```python
z_error = abs(float(hand[2] - target[2]))

grasp_pose_ready = (
    xy_error <= self.env.GRASP_CLOSE_XY_TOL
    and
    z_error <= self.env.GRASP_CLOSE_Z_TOL
)
```

### 使用的解决方法和对应效果

```python
z_above_target = float(hand[2] - target[2])

grasp_pose_ready = (
    xy_error <= self.env.GRASP_CLOSE_XY_TOL
    and
    z_above_target <= self.env.GRASP_CLOSE_Z_TOL
)
```

最后 20 mm 的已验证减速逻辑保持不变，只复用同一个有符号高度差：

```python
if 0.0 < z_above_target < 0.020:
    action[2] = max(float(action[2]), -0.003)
```

close 阶段同样预先计算：

```python
z_above_grasp = float(hand[2] - grasp_target[2])
```

用于原有的近表面减速判断。没有加入 attached 后的额外深度 gate，也没有改变 attached 后立即进入 lift 的既有行为。

## 27. Flow replay 随机性会干扰 checkpoint 的下探误差比较

### 遇到的问题

Evo-1 flow matching inference 从随机 action noise 开始。同一 observation 如果使用不同随机起点，预测 action chunk 会变化，因此直接比较两个 checkpoint 的 `dz` 可能混入 flow noise，无法判断模型是否真的系统性少下探。

原来的 open-loop 汇总只有整体 position ADE 和 gripper accuracy，也不能单独回答：

```text
预测 dz 是否比 expert dz 更不负
descend/close 中少下探的比例是多少
少下探是否只集中在分布尾部
```

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/open_loop.py
```

### 使用的解决方法和对应效果

- `infer_chunk()` 增加可选 `flow_seed`。
- 多 sample 推理使用 `flow_seed + sample_index`，保证每个 sample 可复现且互不重复。
- replay 增加：

```text
--flow-seed-base
```

- 每个 replay frame 的 seed 由以下信息稳定生成：

```text
flow_seed_base
+ episode_index * 10,000,000
+ frame_index * 10,000
```

- 负 seed 会在参数解析或推理入口直接报错。
- 相同 observation、replay index 和 flow seed 可以用于公平比较不同 checkpoint。

同时对 `descend` 和 `close` phase 增加：

```text
dz_error = pred_dz - gt_dz
```

其中正值表示预测动作比 expert 更少向下。最终输出：

```text
mean(pred_dz - gt_dz)
median(pred_dz - gt_dz)
P10 / P50 / P90
under_descent ratio
```

`under_descent` 的计数条件为：expert 明确向下时，预测 `dz` 比 expert 高出超过 `0.001`。

```

```

## 28. Task4 抓取深度与抽屉侧壁安全间距冲突

### 遇到的问题

2026-10-19 评估后的复查确认：seed `1698215761` 的专家 IK 已收敛，但手掌与抽屉侧壁的安全距离不足，实际下探终点比专家目标高约 2.14 mm。原采集报告的 38 条拒收中，37 条集中在 `object_descend`，另 1 条未保持拉屉接触。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/assets/task4_scene.xml
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml
```

### 修改前

抽屉内腔约 0.208 m，部分物体位置下的深抓取目标与手掌、侧壁和 2 mm 安全间距不兼容。

### 使用的解决方法和对应效果

- 内腔调整为 0.248 m，每侧增加 20 mm；柜体和抽屉对应几何同步调整。
- 保留把手位置、柜体位置及朝向、抓取深度和原 2 mm 安全间距。
- 采集配置记录 `drawer_inner_width_m: 0.248`，禁止混入旧场景数据。

释放资格修正前，原 38 个拒收 seed 的完整专家物理检查全部通过；该组未渲染图像，不能算视觉历史验收通过。另有四目标各一条真实 448 图像采集通过完整严格验收。几何扫查的最小手掌—侧壁余量约 14.30 mm。

对应阶段证据为 `/home/user/mujoco+evo/tests/_diagnostics/review_20261019/final_report.json`，其中的源码指纹保留原样；不据此重新认证后续修改的代码。

## 29. Task3 释放后误报掉落、盒内数值边界与开爪时序差异

### 遇到的问题

物体合法释放后，远处空夹爪重新闭合可能被误报为运输掉落；物体边界越过硬编码上界约 0.3 微米也会被判未入盒。专家直接开爪，而 Task3 评估要求连续两次开爪请求，改变了释放后的执行时序。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml
```

### 修改前

脱手事实与夹爪命令混用，释放后重新闭空爪会重新启动掉落累计；盒内判定没有数值余量；Task1 和 Task3 共用两步开爪防抖。

### 使用的解决方法和对应效果

- 区分持物、合法释放和真实重新抓取，合法释放不清除此前已确认的掉落。
- 物理脱手与结束时夹爪打开分别判断，稳定放置按物理子步累计。
- 盒内 XY 边界从实际几何计算，使用 1 微米数值余量；真实越界仍拒绝。
- Task3 二值开闭立即生效，与专家一致；Task1 原防抖行为保留。
- 增加任务里程碑、长期无新里程碑和未抬起时的抓取尝试数，区分执行位移与任务进展。

对原 Task3 `0003`、`0007` 的同一组记录请求做物理时序对照：两步开爪分别在 120、118 步严格失败；一步开爪分别在 114、107 步严格通过。这是固定请求重放，不是新模型反馈闭环，也不能换算成模型成功率。

证据为 `/home/user/mujoco+evo/tests/_diagnostics/review_20261019/task3_release_timing_replay.json`。

## 30. 过高释放在下落途中被追认为合法释放

### 遇到的问题

物体最初在目标上方 120 mm 处失去双指保持，0.07 秒时下降到 96 mm，被重新认定为合法释放并清掉中断记录；到 0.11 秒累计下落 60 mm 仍未报掉落。这会影响评估和专家验收。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/tests/test_contact_events.py
/home/user/mujoco+evo/tests/test_open_loop_test.py
```

### 修改前

每个物理子步重新按当前位置授予释放资格，没有保留首次失去双指保持时的资格。

### 使用的解决方法和对应效果

- 首次双指保持中断时记录资格；同一次中断只能撤销资格，不能因后来进入合法区域而补授。
- 合法开爪允许两根手指先后脱离，完全脱手后才锁存合法释放。
- 过渡中闭爪、离开目标区或升至过高位置会撤销资格；恢复双指保持后才能开启新的判断区间。
- 保留此前真实掉落，合法释放后闭空爪不新增掉落；独立观察器补齐同类时序边界。
- 未修改抓取深度、释放高度上限、100 ms 确认时间、15 mm 下降阈值、场景或 Task1 行为。

Task3 两条严格通过；Task4 四目标物理检查通过。目标 3 在 64px 下缺少目标历史证据，该记录保留；同 seed 用正式 448 分辨率重新完整执行后严格通过，未放宽可见性规则。

本轮报告为 `/home/user/mujoco+evo/tests/_diagnostics/release_qualification_20261020/report.json`。上述为判定与小规模专家回归证据，不代表学习策略成功率已经恢复。

## 31. 单指仍承重时的合法释放误报与 Task1 动作追踪缺失

### 遇到的问题

最新 2026-10-25 best 的 Task3 episode_0003（seed 79385489）在闭爪下降期间失去双指接触，但右指仍持续承重，随后在原允许区域正常开爪。旧规则把首次双指中断直接当作释放资格唯一判定点，20.248 秒误报 transport_drop。另一方面，最新 best/task1/actions.jsonl 为零字节，Task1 失败回合缺少逐动作现场，无法仅凭摘要重建原因。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/tests/test_contact_events.py
/home/user/mujoco+evo/tests/test_open_loop_test.py
```

### 修改前

双指接触中断后，闭爪期间仍有真实单指保持也不能在随后正常开爪时取得合法释放资格；手随物体主动下降会被世界高度下降量误认成自由下落。独立开环观察器也冻结同一资格，且只看控制步边界，会漏过同一步内的实际开爪、承重与脱离顺序。Task1 未接入默认动作追踪分支。

### 使用的解决方法和对应效果

- 仅对该次中断开始时已闭爪且有实际接触载荷的原始保持，记录手端局部物体位置；局部漂移不超过已有 1 mm 跟踪尺度、卸载不超过一个物理子步时，保留保持证据。
- 必须在原合法 XY/高度范围内、仍承重时发生真实闭爪到开爪边沿，才能允许随后完整脱离。最初空夹、已开爪高释放、相对滑移、较长卸载、后续擦碰、脱离后才开爪均不能补领资格；已经确认的掉落不被清除。
- contact_instability_seen 继续保留。跟随手端的承重下降与自由下落分别处理；未改变 100 ms 确认时间、15 mm 下降阈值、释放高度、抓取深度、盒内边界、碰撞限值或场景。
- 独立观察器通过作用域内只读钩子采样 Task3/4 每个物理子步，独立读取原始几何、力和相对位置，不读取正式分类标志；异常或正常退出均恢复原回调。独立 sample() 调用仍不能认证未采样的间隔，Task1 保留原采样方式。
- Task1 默认 JSONL 增加原始/准备/裁剪后目标、执行前后状态、实际跟踪、物体与目标、双指接触及释放证据。读取证据不推进成功计数，不修改动作、物理、成功标准或终端形式；旧空日志无法补回。

本轮保留原正式视频、动作日志、数据、checkpoint 和旧来源哈希；未修改 README、Panda、抓取深度或现有模型参数。真实源文件修复会产生环境约定差异，未重写旧 checkpoint 或数据元信息来伪装兼容。以上单回合固定请求重放不等于重新测得模型成功率，更不代表 Task4 已解决。

证据目录：`/home/user/mujoco+evo/tests/_diagnostics/review_20261026/`，主要文件为 execution_findings.md、retention_fix.patch、retention_fix_late_replay.json、independent_retention_late_replay_summary.json、independent_retention_late_replay.log、combined_regression.json 和 combined_regression.log。

## 32. Task3 专家动作保存精度一致性与完整重放终点验收

### 遇到的问题

正式数据 Task3 episode0（seed 623777821）原质量记录成功，但完整动作重放只有 0.608 秒连续稳定，未达到 1 秒。正式 mujoco 环境与 Evo1 环境都复现，不能归咎于换运行环境。进一步对照发现：专家原精度执行与原精度动作重放均为 117 步、稳定 1.052 秒；只将动作转为数据实际存储的 float32 后，117 步重放稳定时间变成 0.608 秒。微小动作量化影响接触过程，机器人状态高度对齐不能保证物体支撑过程完全相同。

另外，正式开环此前只把机器人状态不对齐列为失败；即使完整重放的终点不满足严格成功，也没有使诊断失败。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/tests/test_contact_events.py
/home/user/mujoco+evo/tests/test_open_loop_test.py
```

### 修改前

Task3 专家以 float64 生成并执行笛卡尔动作，写数据时才转 float32；验收的是另一组实际执行过的数值。开环完整重放终点失败不会单独阻止诊断通过。

### 使用的解决方法和对应效果

Task3 专家在执行前就使用数据保存精度，保证实际被验收的动作与落盘动作一致。只修改专家命令构造，不修改策略 step()、Task1、Task4 动作路径、抓取深度、碰撞或稳定时间标准。

开环保留机器人前状态对齐结果，并另行输出 terminal_acceptance。完整执行才判断正式严格成功、独立抓取/抬起/无掉落/放置/1 秒稳定；前缀写 not_assessed_prefix。终点失败单独记录为 replay_terminal_failures，并使总诊断失败，不混记为动作时序错位。

- 正式 mujoco 环境原数据只读重放：Task3 终点稳定 0.608 秒、严格失败；Task4 episode0 全 302 动作执行，最大机器人位置误差约 0.0593 mm、终点稳定 2.478 秒、严格成功。均不是模型成功率测试。
- 修改后的 Task3 同 seed：专家、float64 重放、float32 重放均 114 步、稳定 1.038 秒、严格成功，三者最终物体和机器人状态一致。该物理精度对照使用 64px 渲染，不认证 RGB 或模型输入；前述正式运行环境重放使用 448px。
- 完整回放失败、完整回放成功、前缀未验收和 Task4 专家命令保持原精度的回归已补充；整组测试仍在运行，结果完成后追加。
- 没有重写旧数据质量标签、环境来源或 checkpoint；不以额外补帧或降低稳定标准掩盖旧记录差异。没有训练或参数更新。README 未改。

证据目录：`/home/user/mujoco+evo/tests/_diagnostics/review_20261026/`，文件为 current_replay_mujoco/report.json、task3_terminal_precision_probe.json、task3_terminal_precision_fixed.json 及对应日志。

- 最终补充：第二 seed 79385489 的专家和 float32 重放均 116 步、稳定 1.006 秒、严格成功。两个 seed 各对照的终点机器人与物体坐标逐值完全一致。
- 正式 open_loop_test.py 以 --audit-only、seed 27 抽中原 Task3 episode0，完整重放实际返回 diagnostic_checks_failed、退出码 1；replay_terminal_failures 精确列出 formal_strict_success 和 independent_stable_one_second。suspected_temporal_misalignment 为空、模型查询为 0。该退出码是反例正确被拒绝，不是测试程序异常；旧数据的真实未通过状态保留。
- 本节全部针对性验证现已完成。模型正确率仍未重新测量，没有训练新权重；不把两条专家成功当成策略验收通过。

## 33. Task4真实偏离的早期专家接管及原开环入口的上下文兼容

### 遇到的问题

真实策略在把手接近阶段会出现方向/姿态偏离，原成功数据不包含对应修正监督。若直到总距离明显变远才接管，部分轨迹已受阻或耗尽K6内的目标线索。另外，采集器增加Task4纠正之后，open_loop_test.py仍只允许Task3从frame0开始的空夹前缀，并假定correction_source总有recorded_empty_grasp_sim_time。这使新增Task4示范不能通过原正式开环入口。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

Task3不适用纠正来源会让整轮采集退出。Task4没有经过验证的真实偏离接管入口；旧开环把所有policy_context都当作Task3空夹，不能识别Task4展示后再执行的安全偏离前缀。

### 使用的解决方法和对应效果

- 保留真实请求前缀，仅在安全、未选屉、目标历史仍有效的早期偏离接管。首40步内分坐标把手误差相对此前最小值增加5mm，或实际姿态偏离HANDLE姿态2度时可接管；这是纠正采集时机，不是修改任务成功或碰撞标准。
- 每个前缀动作前后均与原日志比较状态，最大混合坐标误差不超过1e-4。专家通过原IK/安全控制恢复姿态，不改qpos，不给部署策略隐藏目标坐标/编号。失败动作只作历史，recover之后才进入监督。
- 数据验证区分Task3空夹响应与Task4展示、连续policy_context、recover的顺序，检查来源哈希/run_id、前缀长度、真实偏离、时间和纠正次数。晚接管不合格例保留，没有移除严格质量拒收。
- 原open_loop_test.py采用相同的上下文结构契约；实际重放时单独读取MuJoCo几何，核实Task4把手分量/姿态偏离、接管末态和有效目标历史。Task3仍独立确认空夹，不把metadata标志当作物理复现证据。没有新增替代正式开环的test入口。

Task3正式纠正采集3尝试、2不适用拒收、1严格保存，139帧中前57帧仅作历史，82个监督窗口全部来自专家接管之后；正式raw/Evo检查通过。

Task4底层seed79385490的真实纠正严格保存，float32全动作重放严格成功，前缀各动作前后与源日志最大误差0。较晚干预三目标对照只保存1/3，另两条分别因handle_contact_through_pull/selection_within_history及approach不到位拒收；未把物理完成但严格失败写成合格。

改用早期分坐标/姿态偏离接管后，seed79385488/79385493/79385495分别覆盖目标3/1/2，3尝试、3严格保存、0拒收，status=completed；选屉时有效历史槽分别[0]、[0,1,2]、[0,1]。正式check_dataset.py --require-evo退出0，3条原始数据和448/K6输入接口通过。这不等于模型已经会纠正，也不等于所有保存动作都可完整重放。

原open_loop_test.py --audit-only --replay-steps 10000随后完整检查这三条，模型查询0。前两条完成全程、上下文物理复现和独立/正式终点均通过；第三条在frame278出现旋转向量坐标误差0.012695rad，XYZ约0.0361mm，超过现有重放对齐阈值。整组返回diagnostic_checks_failed、退出1，未放宽阈值、未改写原数据质量。原因尚在用同seed专家请求精度/保存动作对照定位，不能先归因为几何不可行或报告全部通过。

证据均在mujoco_pickplace/outputs/repair_20261028：task3_correction_candidate、task4_correction_candidate、task4_correction_early_pose_candidate、saved_task4_correction_replay.json、task4_correction_early_pose_check.log、task4_early_official_open_loop/report.json、task4_precision_probe.log。原视频、正式数据、best/final和README没有因本条改写。

## 34. Task4专家请求在执行前使用保存精度，避免后段姿态重放漂移

### 遇到的问题

相关问题条目第三条纠正示范的完整开环重放未通过姿态对齐。Task3此前已在专家执行前转float32，但Task4仍使用float64请求、写数据时才转float32；接触和IK分支对细微命令差异敏感。不能凭“原专家成功”或“物体最后放好”掩盖所存观测—动作序列不能准确重现的问题。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

_step_pose仅对Task3专家请求提前转换数据保存精度，Task4不走此转换。

### 使用的解决方法和对应效果

两个原生接触任务的专家请求均在执行前转换float32。Task3实际路径与此前相同；只补Task4专家路径，包含旋转请求。未改策略step()、IK/碰撞限值、缩步、下探目标、Task1、放置/掉落质量标准，也未对旧数据进行改写。保持最小修复，没有因重放偏差改大检查容差。

同seed79385495的隔离对照已经退出0：原float64专家重新执行的354个观测状态与原数据逐值相同；其保存float32动作完整重放的最大混合状态坐标差0.0612938404。先转换专家请求的对照仍为354帧，保存动作完整重放最大混合状态坐标差3.370337e-5、没有超过1e-4的点。两组都有42步缩小执行，两组最终均物理完成，因此该结果证明重放一致性改善，不证明原抓取在物理上不可行或模型成功率提高。该最大值混有米和弧度，不能直接换算成毫米。

该隔离对照使用64px图像，仅认证命令/物理重放，目标可见性检查selection_within_history未通过；不能把64px对照称为448px目标历史验收。正式448px四目标重新采集和原开环完整重放正在独立目录验证，结果完成后追加。

诊断脚本首次在专家完成后因未初始化natural_corrections属性而退出1，没有生成有效对照；已按正式采集器的getattr默认方式修正，失败日志保留为task4_precision_probe.log。成功重跑为task4_precision_probe_retry.log与task4_precision_probe.json，未把首次失败记为通过。

此前最新best/final的Task4各10条、best的Task3四条，另完成每条8个时刻的人工抽帧检查。可见把手前停留、偏离后抬高及Task3重试行为，和既有日志一致；未从抽帧断言完整接触过程或新发现未经日志支持的根因。证据为video_review/manifest.json及五张时间标注接触表，原视频未变。

相关问题条目补充：修复后的正式448px四目标纠正采集已退出0，4尝试、4保存、0拒收、completed。seed79385488/79385490/79385493/79385495分别为目标3/0/1/2，帧数389/335/336/354；全部严格检查通过，有效目标历史槽分别[0]、[0,1]、[0,1,2]、[0,1]。正式check_dataset.py --require-evo退出0，4条raw和448/K6输入接口通过。独立目录为task4_precision_correction_candidate，没有替换旧数据或权重；保存动作完整重放另见相关条目，不能以采集通过代替重放。

## 35. 正式开环按采集分辨率重放，并复核阶段接触质量

### 遇到的问题

原CLI调用replay_alignment没有传入分辨率，默认64px；Task4目标展示可见性依赖实际像素，64px物理重放不能认证448px目标历史。另一个缺口是重放所有执行帧都标记为policy_execution，终点只检查正式success和独立放置/抓取/抬起/掉落，不能复核handle_contact_through_pull等按阶段定义的质量。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

CLI使用64px默认重放，丢失原阶段标签，最终物理完成可能掩盖阶段接触质量分歧。

### 使用的解决方法和对应效果

- 正式CLI从已验证的相机metadata读取接触任务的重放分辨率，本轮实际为448；直接调用API仍可显式选择64px作有限物理诊断，结果记录实际大小，不据此认证448px可见性。
- 重放只执行已保存的动作，不调用专家控制器；保留记录中的阶段标签供接触质量复核，policy_context仍仅为历史。碰撞和保持判断继续依据原始物理，不靠阶段标签豁免。
- 完整重放终点增加formal_trajectory_quality和replayed_expert_quality，检查现有阶段接触及目标历史规则。前缀仍明确not_assessed_prefix，不因缺少完整任务而假装验收通过。
- 正式控制、成功标准、数据和checkpoint未改；未新增另一套开环入口，未放宽原0.5mm/0.01rad/1mm的状态对齐容差。

配置不完整的precision_official_open_loop验证由本轮向所属单一进程发SIGINT中断，原报告保留KeyboardInterrupt/error；不是因耗时终止，也不记作通过。修正后的原正式CLI在precision_official_open_loop_448串行完整重放Task3一条、Task4四目标四条，尚在运行；完成后追加实际结论。

最终补充：上述448px正式CLI已正常退出0，status=data_audit_complete_model_not_tested。5条均whole_episode_executed=true、recorded_failure_context_verified=true、terminal_acceptance=passed，阶段质量failed_checks为空；suspected_temporal_misalignment与replay_terminal_failures均为空。Task3执行139动作、独立稳定2.004秒；Task4四目标执行345/291/292/310动作，独立稳定2.510/2.522/2.500/2.474秒。全5条最大XYZ对齐误差0.0001481mm、最大旋转向量误差1.7191e-6rad，未修改容差。模型查询0，不据此报告模型成功率。

另通过原load_episode/replay_alignment API验证7个负例：错误纠正类型、不实安全标志、错误前缀帧数/干预时间/重放误差、缺失纠正来源，以及结构合法但与实际几何姿态不符的伪证书，均被拒绝；最后一项在448px真实前缀重放中被拦住。验证进程退出0，未修改任何原数据、模型或统计；证据为correction_context_contract_verification.json及日志。

最终结果只认证所列5条保存动作和本轮入口修复。有限专家样本通过不代表已覆盖全部随机状态，更不代表400步候选或现有模型达到90%/80%。

## 36. 真实部分抬起后的保持监测与独立盒内字段比较

### 遇到的问题

最新隔离原型在Task3真实反馈中双指夹住筷子，抬高40.31mm后主动开爪，物体落回桌面。正式和独立保持检测均以50mm任务抬起里程碑为前提，漏记了这次真实掉落。策略主动开爪是另一项学习错误，不能靠修改标签解决。

另一条新Task3完整成功轨迹有7个contained比较分歧：物体悬在盒上方、XY已在盒内，独立contained为真，而正式placement.contained还要求原放置高度。此前汇报的“一个时刻”实际是一条回合中的7个控制时刻；不是最终放置失败。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

不足50mm的真实自由持物没有独立保持监测资格；Task3/4合法放置后的支撑中碰触还可能依历史抬起高度重新建立搬运状态。独立XY投影与正式包含高度的放置字段直接比较，产生并非物理状态矛盾的分歧。

### 使用的解决方法和对应效果

- Task3/4增加与成功抬起里程碑分离的保持监测：真实双指加载、物体高于初始1mm、外部向上支撑低于物重5%、闭合保持连续40ms后建立自由持物资格。原50mm成功要求不变；原已经超过50mm的首次双指保持仍立即监测。
- 外部支撑读取完整接触力及摩擦、转换到世界坐标并按geom1/geom2调整受力符号；各接触向上的分量单独累加，不能被向下载荷抵消。非法释放沿原时间段下降/分离证据确认，不能在下落途中补领释放资格；已确认掉落继续锁存。
- 合法释放后只允许新的持续自由加载重建保持。支撑中双指碰触、远离物体的空闭爪均不能建立新的搬运掉落。独立观察从原始物理另算资格；Task1原抬起保持路径及释放重置规则保留。
- 独立观察保留XY含义，并增加明确的contained_xy和contained_at_placement_height；后者从实际桌面和物体几何使用原Task3/4高度区间计算，分歧比较改用同义字段。正式成功、稳定时间、控制、几何、抓取深度与阈值不放宽。

隔离候选13项物理检查通过：两种geom顺序×水平/竖墙摩擦支撑4项、真实40mm掉落固定请求重放1项、Task3专家与Task4四目标专家及释放后空闭爪5项、真实支撑/自由双指加载3项。掉落重放机器人状态最大误差小于1e-6，原始失败日志保持；新增监测不把不足50mm的抬起算成功。64px检查仅认证物理，不认证448px历史可见性。

应用前同时校验生产原件哈希、候选哈希与两份通过报告，原件保存至single_frame_memory_20261030T010304Z/partial_lift_retention_candidate/production_before_apply。正式448px原开环入口对一条Task3与四目标Task4保存动作的完整重放仍在运行，结果完成后追加，不能提前记通过。Task1未做本轮20回合配对；这些物理检查均不是模型成功率。

## 37. 正确选屉后接触丢失纠正的实际重放结果与新反馈受阻证据

### 遇到的问题

正确首次选择后把手接触丢失的来源需要与普通未选择的接近偏离区分。第一次新纠正声明错误地以接管端点的抽屉位移代表前缀最大位移，独立重放已拒绝，不能将其收为合格监督。另一次真实反馈中IK允许全量动作而手端/指宽没有跟上，不能把允许目标、实际执行与正式双指选择混为一谈。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

纠正入口只接受首次选屉之前的接近偏离；新增声明生产者最初未正确累计前缀偏移，不能证明已观测整个前缀均满足未开屉条件。有关结构规则和初次拒绝保留于PI_MEM报告相关条目。

### 使用的解决方法和对应效果

- 新handle_contact_loss仅接受原首次选择正确、前缀真实曾有双指把手接触、接管时闭爪且已丢双指接触、整个前缀距闭合位置不超过5mm、目标历史仍有效且无危险/不可逆失败的情况。
- 累计整个前缀各抽屉离闭合位置的最大绝对偏移；旧错误数据保持quarantined、检查/训练拒绝。correct_handle_loss_correction_v2记录真实累计最大偏移0.002032522731m，原错误端点值为0.000118518801m。
- 原approach类型5mm/2度资格、错误选择与真实掉落锁存、抓取深度、物理碰撞和严格验收标准均未放宽；初始化v2仅忠实表达实际单目标，不向策略提供隐藏目标。

v2严格专家、正式raw检查和448px完整保存动作重放已全部通过，321个专家动作窗口可用；77帧展示/失败上下文仍排除动作监督。13个结构正负例和初始化12个正负例的历史通过结果保留。该结果只认证这一条真实纠正，不是模型成功率。

2026-10-31新原生图像读出反馈：Task3同适配seed106步、一次抓取、严格成功，contained/released/supported/drop分歧均0。Task4第一次尝试105步发生CUDA unknown中断，不计完整回合结果，原记录保留；新的独立CUDA小运算通过后仅补跑Task4，未重做Task3或训练。

Task4前50步的物理证据：第27步已首次接触正确把手3，但无双指把手接触和确认选择；第29至32步IK执行比例1、允许夹爪目标等于请求，实际指宽仍接近零，位置跟踪误差约8.475/5.548/5.375/4.224mm。因此这几步不是IK拒绝或保护主动把开度改零；允许的把手接触不在unexpected列表，不能用空列表认定没有接触阻碍，也未凭记录确定具体受力几何对。最早受阻时可见目标读出仍为3，第47步才首次无目标线索。完整补跑仍在进行，不提前报告成功或完整失败。

## 38. 控制边界支撑缓存刷新与真实策略接续的接近耗时

### 遇到的问题

最新Task3成功轨迹step99在mj_forward前后的支撑缓存不一致：正式supported为真、独立观察为假；角速度17.03rad/s，稳定及成功均为假，没有在该步假成功。另prefix28专家接续虽最终物理完成，首次选择时目标展示已移出K6，不能被接纳为合格轨迹。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/task_env.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/late_approach_path_diagnostic_20261031.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

integrate的物理子步推进placement后，末尾已有mj_forward刷新几何/接触，但记录前未同步placement缓存。慢接近的具体原因尚未查明，不能先归因于碰撞保护过严。

### 使用的解决方法和对应效果

在既有mj_forward后、记录前更新同时间戳placement，不增加积分/forward，不改变qpos/qvel或控制；同时间戳不补记稳定时间。修改前生产文件留在placement_boundary_candidate/production_before_apply。

对真实prefix28执行原专家接近/闭合阶段，逐步记录IK、请求/允许/实际手端进展。它只是把手前缀诊断，不是训练数据或完整任务结果。不放宽碰撞、抓取、稳定时间和历史要求。

原/候选及应用后生产重放106步真实成功、177步部分抬起真实掉落：各步qpos/qvel差0，成功轨迹support分歧清零，真实掉落保留，重复success查询不推进时间。正式448px原开环入口Task3一条/Task4四目标四条完整重放终点均通过，模型查询0。

prefix28接近41步、最终把手闭合成功；期间IK全量执行，无碰撞拒绝，实际手端在距预抓取目标约0.35mm处缓慢收敛。首次选择12.136s、最后展示4.1s，按原历史要求严格拒收。该证据不支持删安全检查或改变柜体几何；实际策略失败仍待解决。新正常初段拟合候选真实Task4仍600步失败，未部署。

相关问题条目实例候选验证补充：`outputs/repair_20261028/pregrasp_staging_candidate_20261031.py`仅在该实例专家的approach过渡点采用1mm到达精度，最终handle_approach仍要求原0.3mm、目标位置完全不变。过渡点4步后误差0.716mm，随后最终对准5步、误差0.158mm；首次选择从12.136s提前到8.336s，完整专家、原连续接触/碰撞/运输/稳定/历史质量均通过，独立终点分歧0，无优化更新。候选仅消除中间非抓取过渡点不必要的精确等待，没有放宽最终抓取精度或质量门槛。当前仍是实例候选、不是生产默认或训练数据；为保持冻结模型筛查条件不在途中应用生产改动。

## 39. 冻结候选独立初筛中的越沿受阻与运输接触

### 遇到的问题

Task4 seed194951009目标2已正确开屉约0.172米，首次前沿拒绝时手端高度约0.5725米，模型同时向内平移并下降1.33毫米。该初态专家跨沿高度约0.597米，缺口约25毫米。Task4目标3另有对向夹爪接触及后续桌面受阻，不能并入同一根因。Task3 seed194951009在第70步记录几何对72/91的接触，前后仍为双指夹持，最终稳定放置，严格失败保留。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/drawer_entry_recovery_probe_20261104.py
```

### 修改前

已有安全的把手接近纠正监督。该模型另一个独立失败发生在正确开屉后的越沿阶段，缺少这一状态下完成同一任务的可行路径验证。

### 使用的解决方法和对应效果

准备真实动作前缀重放，逐步比对机器人状态，要求正确选屉、开度及此前质量证据。专家诊断先垂直取得原越沿高度，再执行原俯抓与取放路径。诊断不更改策略、生产控制器或安全判定，不写入训练集。

冻结候选的目标2与目标3原回合均以600步结束。目标2的专家续段已完成，原176步机器人状态最大误差为0，介入前开度0.17169米，无质量失败。先抬升24.493毫米，再完成俯抓、搬运与释放，总计325步，严格成功，连续稳定支撑2.474秒，无新增安全事件。这验证可行纠正路径，不计入模型成功率。探针只输出物理结果，本次未生成或训练新监督数据。

续验：从同一真实前缀重新采样了42个越沿与抓取阶段诊断窗口。保存的149个专家动作在新模拟器完成原预算内重放，逐步机器人状态最大差0，严格成功，独立终点无分歧。只有接手后的动作是标签。Task3另一条真实空夹纠正走正式采集与严格raw入口，一条145帧，58帧仅作历史；原open_loop_test在Evo1环境完整重放，失败前缀复现和严格终点均通过，模型查询0。两项均未纳入正式训练。

## 40. 原模型没有短期观测记忆

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

### 使用的解决方法和对应效果

```text
K=6 图像 -> 因果时空 ViT -> 当前帧视觉 token
K=6 状态 -> 共享状态编码 -> 6 个带掩码的状态 token
```

时间注意力只连接同相机、同空间块，固定时间编码满足 `e(0)=0`；`K=1` 直接调用原始单图路径。

因果性、左填充隔离、K=1 精确路径和时间 QKV 反向传播测试均通过。

## 41. 历史窗口可能越过 episode 或与时间轴不一致

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

### 使用的解决方法和对应效果

- 优先按 parquet 时间戳选择 `K=6`、间隔 `1.0 s` 的历史。
- 时间戳不可用时退回 `memory_stride_steps=5`。
- episode 开头左填充第一帧，并令对应 `history_mask=False`。
- 默认保留完整 5 Hz 时间轴；显式压缩静态帧时同步重建视频与 parquet 时间戳。
- 缓存目录包含 horizon、K、步长和秒间隔，避免读取旧配置缓存。

500 个 episode、64,185 帧及 64,185 个 K=6 Evo 窗口检查通过。

## 42. 论文要求在 ViT 上层丢弃过去时刻 token

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

### 使用的解决方法和对应效果

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

真实 K=6 前反向峰值已分配显存由 `4345.8 MiB` 降至 `3544.7 MiB`，减少 `801.1 MiB`，约 `18.4%`。

## 43. 视觉编码器保留全部层输出引用

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

### 使用的解决方法和对应效果

先把 `select_layer` 解析成目标层，只运行到所需层，不保存所有层输出；默认 `-1` 仍运行完整网络。

固定正弦时间位置编码也改为每次视觉前向只生成一次，由各时间层复用。

## 44. 填充相机造成无效视觉和语言计算

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

### 使用的解决方法和对应效果

- ViT 前删除无效相机，只编码 `6 × 1` 张真实图像。
- 语言主干前删除无效相机占位，而不是计算后只改 attention mask。
- 融合序列为 `[1,277,896]`，277 个 token 全部有效。

## 45. 图像预处理存在 CPU/PIL 往返

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

### 使用的解决方法和对应效果

K×V 图像整批在 Tensor 空间执行 bicubic resize 和 ImageNet 归一化；PIL 路径只保留给服务端图像输入。

## 46. 无效动作目标会影响有效动作预测

### 遇到的问题

动作损失虽然屏蔽了无效维度，但无效 `actions_gt` 仍参与 flow 中间状态并进入动作 token 自注意力，可能改变有效维度的预测。

### 修改文件

```text
Evo-1/Evo_1/model/action_head/flow_matching.py
Evo-1/Evo_1/scripts/train.py
```

### 修改前

只在损失或噪声上应用 `action_mask`。

### 使用的解决方法和对应效果

```python
action_intermediate_seq = action_intermediate_seq * action_mask
squared_error = (pred_velocity - target_velocity).square()
loss = (squared_error * action_mask).sum() / action_mask.sum().clamp_min(1)
```

无效目标在进入动作投影前归零；每个样本无有效动作时直接报错。

将无效目标维度增加 1000 后，有效预测保持逐元素完全一致。

## 47. 服务端拒绝单路真实相机

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

### 使用的解决方法和对应效果

服务端允许 `1..max_views` 路物理相机；烟测改为调用 `infer_from_json_dict`，覆盖解码、校验、归一化、模型和动作反归一化完整路径。

单路真实相机的 K=6 请求通过完整服务入口，输出 `[14,24]` 且全部有限。

## 48. 旧列表协议错误交换 RGB 通道

### 遇到的问题

MuJoCo Renderer 输出 RGB，旧服务端却按 BGR 转 RGB，导致红蓝通道交换。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1_server.py
tests/test_short_term_memory.py
```

### 使用的解决方法和对应效果

列表图像直接按 RGB 构造 PIL，并验证输入必须是 `[H,W,3]`。纯红图像回归测试保持 `[255,0,0]`。

## 49. 在线评估开环执行时间过长

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

### 使用的解决方法和对应效果

```text
预测 14 步 -> 默认执行前 4 步 -> 重新观测和规划
```

`--horizon` 仍可调整，但不允许超过模型 horizon 14。该修改无需重新训练，可减少开环漂移。

## 50. 单步夹爪噪声会立即触发开合翻转

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

### 使用的解决方法和对应效果

- 小于等于 0.4 视为强关闭请求，大于等于 0.6 视为强打开请求。
- 0.4–0.6 区间保持上一命令。
- 默认连续 2 步同方向请求后才切换；`--gripper-debounce-steps 1` 可关闭去抖。
- 只读取模型命令，不读取 MuJoCo 的 `attached` 等隐藏真值。

单步反向请求不会改变夹爪状态，连续两步请求会正常切换。

## 51. 数据损坏时静默换成随机样本

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

### 使用的解决方法和对应效果

直接抛出包含缓存或视频路径的 `RuntimeError`，由训练入口报告真实数据问题。

## 52. BF16 时间层可能没有真实权重更新

### 遇到的问题

旧 checkpoint 的视觉权重为 BF16。普通 AdamW 在 `1e-5` 学习率下可能把小更新舍入为零，表面完成反向但时间层没有学习。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1.py
Evo-1/Evo_1/scripts/train.py
```

### 使用的解决方法和对应效果

- 只将第 4/8/12/16/20 层的 attention、norm 和 layer scale 保持 FP32 主参数。
- LoRA 参数保持 FP32；独立验证脚本检查第 4 层 QKV LoRA-B 在 optimizer step 后真实变化。
- 其余 VLM 仍冻结并保持 BF16。

第 4 层 QKV 最大绝对变化为 `1.0013580e-05`，五个时间层梯度均有限且非零。

## 53. episode 开头的无效历史仍进入 ViT

### 遇到的问题

`history_mask=False` 虽能阻止无效历史影响当前注意力，但对应图像仍会执行 resize、patch embedding 和 ViT。只有当前帧有效时，旧实现还会执行没有真实历史的时间分支。

### 修改文件

```text
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
Evo-1/Evo_1/model/internvl3/temporal_vision_encoder.py
```

### 修改前

固定用 K=6 图像进入 ViT，再依赖时间掩码隔离左填充。

### 使用的解决方法和对应效果

- 只允许 `False...False, True...True` 的左填充掩码，拒绝中间空洞。
- 无效前缀在批量 resize 和 ViT 之前删除。
- 只剩当前帧有效时严格调用原 `extract_feature` 单图路径。
- 有效视觉时间位置仍与状态记忆的 `[-K+1, ..., 0]` 后缀一致。

新增单图精确路径、左填充预处理压缩和非法空洞掩码测试，全部通过。

## 54. 语言主干保留全部层输出和推理 KV cache

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

### 使用的解决方法和对应效果

```python
output_hidden_states=False
use_cache=False
fused_hidden = outputs.last_hidden_state
```

prompt 图像 token、`num_tiles_list` 和视觉 embedding 数量必须完全一致，否则立即报错。

真实 K=6 模型严格加载、前向、反向和 optimizer 更新均通过，融合序列仍为 `[1,277,896]`。

## 55. flow 动作头重复生成无用注意力权重

### 遇到的问题

8 个动作 Transformer 层在每个 flow 积分步中调用 `MultiheadAttention` 时使用默认 `need_weights=True`，会生成注意力权重，但返回后立即丢弃。

### 修改文件

```text
Evo-1/Evo_1/model/action_head/flow_matching.py
```

### 修改前

每次 cross-attention 都计算输出和未使用的权重；每个 flow 步还重复切片、转换和复制相同时间编码表。

### 使用的解决方法和对应效果

```python
self.attn(..., need_weights=False)
```

时间编码表在 flow 循环外准备一次，循环中只索引当前时间并用 `expand` 扩展 batch。

该算子优化不改变积分步数。当前评估保持 Evo 原始的 50 步；真实 batch 8 推理输出为 `[8,14,24]` 且全部有限。

## 56. 非有限张量或梯度被静默跳过

### 遇到的问题

训练发现 NaN/Inf 时会 `continue`，step 不增加，可能无限跳过坏 batch；梯度范数非有限时仍可能进入 optimizer。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
```

### 修改前

记录日志后继续下一 batch，不保证训练结果有效。

### 使用的解决方法和对应效果

- state、action、融合 token、预测或 loss 非有限时抛出带 step 和张量名的 `FloatingPointError`。
- 裁剪前后梯度范数非有限时禁止 optimizer step 并立即失败。

真实数据训练入口完成一个 optimizer step，损失 `0.9492`，第 4 层 QKV 最大绝对变化 `1.0013580e-05`，峰值显存 `3621.2 MiB`。

## 57. FlashAttention 的 C++ ABI 失配

### 遇到的问题

旧环境中的 `flash_attn` 扩展要求 `CXXABI_1.3.15`，但运行时加载了较旧的 `libstdc++`，因此模型只能回退到标准 PyTorch attention。

### 修改文件

```text
Conda 环境 Evo1
PI_MEM_README.md
tests/PI_MEM_ENVIRONMENT_RESULT.json
```

### 修改前

只能证明 Python 包存在，不能证明 CUDA 扩展可以完成真实计算。

### 使用的解决方法和对应效果

- `Evo1` 环境使用 `libstdcxx 15.2`，满足扩展的 ABI 需求。
- `flash_attn 2.8.3.post1` 可直接导入。
- 保留模型初始化时的扩展探测和安全回退；环境以后发生变化时不会导致整个模型初始化失败。

在 RTX 5060 Laptop GPU 上执行 BF16、causal 模式的 FlashAttention，输入为 `[1,128,8,64]`。前向输出与 Q 梯度均全部有限，反向传播通过。随后真实 K=6 训练入口也在未触发回退的情况下完成。

## 58. DeepSpeed CPU optimizer offload 缺少链接库

### 遇到的问题

编译器和 Ninja 实际已经存在，但 CPUAdam 第一次 JIT 链接失败：DeepSpeed 只搜索 Conda 标准库目录，而 `libcurand`、`libcufile` 当时只位于 pip 包的私有目录。`libaio` 也缺失，导致异步 I/O 兼容性检查失败。

### 修改文件

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

### 使用的解决方法和对应效果

- 在 `Evo1` 环境内安装 `libcurand-dev 10.3.9.90`、`libcufile-dev 1.13.1.3` 和 `libaio 0.3.113`，版本与 CUDA 12.8 对齐。
- CPUAdam JIT 缓存固定到当前 Python 环境的 `var/torch_extensions`，不写用户全局缓存，也不设置系统级环境变量。
- 训练启动后记录 DeepSpeed 外层优化器、基础优化器和 ZeRO stage，防止配置未生效却被误判为已启用。
- DeepSpeed 自己管理模型前向的 BF16，不再给已由 DeepSpeed 接管的 forward 额外套一层 PyTorch autocast；视觉嵌入的直接方法调用仍保留所需 autocast。
- `resume_pretrain` 只加载模型权重时，日志改为准确显示 `model weights only`，不再错误声称加载了 optimizer state。
- README 的启动命令显式使用 `--dynamo_backend no`，避免 Accelerate 为未指定的默认值重复告警。

- 独立 CPUAdam JIT 成功，参数从 `[1.0,-2.0]` 更新为约 `[0.9,-1.9]`。
- DeepSpeed 环境报告中 Ninja、CPUAdam 和 async I/O 均兼容。
- 真实 Accelerate 启动显示：`DeepSpeedZeroOptimizer`、`DeepSpeedCPUAdam`、ZeRO stage 2。
- 从 2.8GB checkpoint 严格恢复模型权重，复用 64,185 个 K=6 窗口并完成一次真实训练更新。
- 单步损失 `0.6562`；第 4 层 QKV 最大绝对变化 `1.52587891e-05`；峰值已分配显存 `3008.5 MiB`。

## 59. 默认 LoRA 微调

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

### 使用的解决方法和对应效果

- 训练 CLI 默认启用 `rank=8`、`alpha=16`、`dropout=0` 的 LoRA。
- 默认目标为 `vision,action`：五个 π-MEM 时间 ViT 层的 QKV/投影，以及动作头内的 MHA、FFN、状态/动作编码器和输出投影。
- 基础大矩阵冻结；训练约 1.49M 参数，其中动作头 1.21M、时间 ViT 约 0.28M；包含 LoRA 以及目标范围内的偏置、LayerNorm 和时间层缩放参数。
- LoRA-A 使用 Kaiming 初始化，LoRA-B 从零初始化，保证注入后首次前向与基础模型一致。
- LoRA 参数保持 FP32 且不施加 weight decay，避免小学习率下 BF16 更新消失。
- `--resume --resume_pretrain` 可以从不含 LoRA 键的旧阶段一 checkpoint 初始化；LoRA checkpoint 的正常续训仍严格加载模型和优化器状态。
- `--no-use_lora` 完整保留原来的全参数动作头、选择性时间 ViT 和 `--finetune_vlm` 路径。
- 推理服务在严格加载 LoRA checkpoint 后合并增量权重，不保留额外低秩前向算子。

- LoRA Linear 和 MultiheadAttention 在 LoRA-B 为零时与原模块输出一致。
- MHA 覆盖跨注意力、padding mask、反向梯度与推理合并。
- 旧基础 state dict 只缺少预期的 `lora_*` 键；LoRA state dict 可严格重载。
- 动作头从 92.24M 全参数训练降为约 1.21M 个训练参数。

## 60. batch 8、FlashAttention 与阶段一恢复

### 遇到的问题

DeepSpeed 包装后直接读取不含 LoRA 的阶段一 checkpoint，会按新冻结参数表解释旧分片，并在动作头 MHA 的 `in_proj_weight` 上触发 `KeyError`。评估服务还曾把 Evo 原始 50 个流匹配求解步降到 32，可能降低动作质量。

### 使用的解决方法和对应效果

- `--resume_pretrain` 在 DeepSpeed 包装前载入阶段一模型权重；同一 π-MEM 任务续训仍在包装后恢复模型、优化器和调度器。
- 评估恢复 Evo 原始 `num_inference_timesteps=50`。
- 删除自动清理高步数 checkpoint 以及仅供临时验证的训练参数；训练不会擅自删除已有 checkpoint。
- 正式命令改为物理 `batch_size=8`、`max_steps=16000`、`warmup_steps=500`、`ckpt_interval=2000`。

RTX 5060 Laptop 8GB 上，`torch 2.12.0.dev + CUDA 12.8 + flash-attn 2.8.3.post1 + DeepSpeed ZeRO-2 CPU offload` 使用真实阶段一 checkpoint、K=6、448×448、物理 batch 8 完成前向、反向和优化器更新；loss 为 `0.7695`，峰值已分配显存 `4918.1 MiB`。独立评估前向输出为 `[8,14,24]` 且全部有限。

## 61. LoRA 后取消 CPU optimizer offload，并批量化 VLM

### 遇到的问题

相关条目的 CPUAdam 方案针对 113.25M 可训练参数有效；默认 LoRA 后只训练约 1.49M 参数，Adam 状态已经缩小到十几 MB，继续 offload 会保留 CPUAdam、同步和 GPU/CPU 传输开销。训练循环还把物理 batch 8 拆成 8 次 B=1 的视觉语言前向，没有利用大矩阵的批量吞吐。

### 使用的解决方法和对应效果

- `ds_config_pi_mem.json` 保留 BF16 与 ZeRO-2，删除 `offload_optimizer`。
- `train.py` 每个物理 batch 只调用一次 `get_vl_embeddings_batch()`。
- `internvl3_embedder.py` 新增 `[B,K,V,C,H,W]` 批量预处理、批量视觉 token 注入、批量 tokenizer padding 和一次语言主干前向。
- `temporal_vision_encoder.py` 为时间注意力增加独立 batch 轴；每个样本使用自己的左填充 mask，禁止跨样本注意力。
- 单样本服务和 K=1 原 Evo 路径保持不变；模型参数结构不变，旧 checkpoint 兼容。

批量化保持 MEM 原文的 K=6、1 秒 stride、每 4 层同 patch 因果时间注意力、`e(0)=0` 固定正弦位置编码、上层只保留当前帧以及 K=1 与原图像编码器一致。MINT `evo1-flash` 只用于参考 batch-aware VLM 接口，不替代 π-MEM 时间机制。

- 35 项 LoRA、短期记忆、批量时间 mask、完整 embedder 等价和视觉 token 注入测试通过。
- 批量 temporal 输出与逐样本输出在不同左填充长度下数值一致。
- 当前 Windows 图形负载占用约 6.4/8.15 GiB GPU 显存，真实 K=6、batch 8 的新路径性能验证暂未执行；不能沿用相关问题条目旧串行路径的显存和速度数字作为本节结论。

## 62. batched K=6 单卡反向峰值与完整断点恢复

### 遇到的问题

相关问题条目完成 batch-aware VLM 后，单张 8 GB GPU 使用无 CPU offload 的 DeepSpeed ZeRO-2 仍在反向阶段 OOM；改成普通 AdamW 后，整块 checkpoint 时间 ViT 层也会在重算 attention 与宽 MLP 时产生重叠峰值。另有两个断点语义问题：`step_final` 曾把字符串 `final` 作为进度写入，普通 checkpoint loader 也没有把已保存的 scheduler state 返回给训练循环。`resume_pretrain` 还错误继承阶段一的 `best_loss`，可能阻止阶段二生成自己的 `step_best`。

### 使用的解决方法和对应效果

- 单卡正式入口改为 Accelerate BF16 + CUDA fused AdamW；DeepSpeed 配置继续保留，但只用于以后多卡训练。
- LoRA 参数仍以 FP32 保存并由 AdamW 更新；适配器矩阵乘法使用 BF16 activation dtype，避免把 `[B,K,N,C]` 激活和 QKV 增量整体扩成 FP32。
- 时间 ViT 层把 attention residual 与 MLP residual 分成两个独立的 non-reentrant gradient checkpoint，数学顺序和 π-MEM 因果 mask 不变。
- `step_best`、`step_final` 的 tag 与数值 `global_step` 分离；普通 checkpoint 在同一个模型状态文件中保存 optimizer 和 scheduler。
- 普通续训恢复模型、optimizer、数值 step、scheduler；`resume_pretrain` 只取阶段一权重并将新任务的 `best_loss` 重置为正无穷。
- 训练日志增加 CUDA 当前/峰值 allocated 和 reserved 统计，便于长训练发现显存漂移。

- 38 项 LoRA、π-MEM、批量/逐样本等价、跨样本 mask 隔离和 BF16 LoRA 梯度测试通过。
- RTX 5060 Laptop 8 GB 上，真实 64,185 个 K=6 窗口、InternVL3-1B、448×448、物理 batch 8、双塔 Flash-Attn 完成生产入口前向、反向、梯度裁剪和 fused AdamW 更新，未 OOM。
- 首次短步从阶段一 `step_best` 初始化，完成 step 0 并保存普通 `step_final`；文件含 `module`、199 个非空 optimizer state、数值 `step=1` 和 scheduler `last_epoch=1`。
- 随后从该 `step_final` 正常续训，日志明确显示 optimizer、step=1、scheduler 全部恢复，完成 step 1 后保存 `step=2`、scheduler `last_epoch=2`；两次训练均 exit code 0。
- `Evo1_server` 严格载入新 checkpoint，合并 42 个 LoRA 模块，确认 vision=`flash-attn`、language=`flash_attention_2`，以 K=6 和 50 个 flow steps 输出有限的 `[14,24]` 动作；exit code 0。

以上只证明当前训练、保存、恢复和推理路径可运行；完整 16000 步稳定性与 MuJoCo 闭环成功率仍需正式训练后判断。相关条目的“35 项、GPU 待验证”是当时状态，当前以本节的 38 项和真实生产验证为准。


采用分段gradient checkpoint与完整训练状态恢复，降低K=6反向峰值并使优化器、调度器和采样游标可以接续；单卡8GB配置曾将峰值allocated控制在约6367MiB，不需要CPU optimizer offload。

## 63. warmup 低损失阻止 step_best 落盘

### 遇到的问题

warmup阶段偶然极低的单批loss可能使后续step_best长期不落盘。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
```

### 修改前

最初将所有step更新best_loss与warmup后才保存分开，可能看不到best文件。

### 使用的解决方法和对应效果

早期曾将best比较也延迟；随后按MINT语义恢复所有step参与比较、只对实际保存施加warmup下限，不重复在每个epoch重置。

当前保留MINT单batch best判定，final与周期checkpoint独立保存，best不被当作闭环最优的证明。早期延迟比较方案已被后续修正取代。

## 64. 云端默认 Accelerate 配置导致单卡 LoRA 误启 DeepSpeed

### 遇到的问题

正式命令没有写 `--use_deepspeed`，但云端用户目录中的 Accelerate 默认配置可能仍把 `distributed_type` 设为 DeepSpeed。此时启动日志会出现 `DeepSpeedZeroOptimizer` 和 ZeRO stage 2，单卡无法获得参数分片收益，反而因包装器与通信缓冲在 batch 8 反向阶段 OOM。

### 使用的解决方法和对应效果

- 不新增 Accelerate YAML，也不修改只负责数据路径的 `dataset/config.yaml`。
- `train.py` 在构造 `Accelerator` 前检查启动环境：单 GPU、默认 LoRA 且检测到 DeepSpeed 时，自动清除 DeepSpeed 配置并回到 Accelerate + CUDA fused AdamW。
- 正式命令前显式写 `ACCELERATE_USE_DEEPSPEED=false` 表达单卡路径；脚本检查负责拦截云端默认 YAML 或旧命令重新注入的 DeepSpeed。
- 仅为有意诊断保留 `--allow_single_gpu_deepspeed`；正式单卡训练不得使用。

## 65. step_best 恢复 MINT 原始判定顺序

### 遇到的问题

延迟best_loss比较使首次允许保存时阈值仍为无穷，连续较低batch可能反复覆盖best。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
```

### 修改前

早期修正同时推迟了比较与保存，偏离MINT原判定顺序。

### 使用的解决方法和对应效果

恢复所有全局step参与best_loss比较，仅将实际保存限定为step大于max(1000,warmup_steps)；跨epoch不重置step和best_loss。

warmup保存限制只作用于训练开始的一次区间，保留原单batch指标及checkpoint机制，未用loss取代任务效果。

## 66. 云端 checkpoint 下载后无法启动评估

### 遇到的问题

阶段二 checkpoint 的 `config.json` 保存了训练机器上的绝对 VLM 路径 `/mnt/workspace/modelscope_cache/OpenGVLab/InternVL3-1B`。checkpoint 下载到本机后该目录不存在，Transformers 在模型初始化阶段直接报 `Incorrect path_or_model_id`。同时，评估服务原默认 checkpoint 仍指向阶段一，README 未显式指定阶段二目录。

### 使用的解决方法和对应效果

- 服务端默认 checkpoint 改为仓库内 `ckpt/evo1_mujoco_pickplace_stage2/step_best`，README 同时显式写出该路径。
- 服务端启动前检查 `config.json`、`norm_stats.json` 和模型状态文件是否完整。
- checkpoint 中的绝对路径存在时继续原样使用；若确认是已迁移的 `OpenGVLab/InternVL3-1B` 云缓存路径，则回退为可移植模型 ID。
- 新增 `--vlm-name`，用于显式指定其他本地模型目录或 Hub ID；未知的失效绝对路径会直接给出可操作错误，不静默猜测。

## 67. 初次夹取专家轨迹严格门禁

### 遇到的问题

本轮问题发生在开始夹取，不是放置。旧专家控制器的 80-seed sweep 虽有 76 次最终成功，但其中 34 次在首次附着时 cube 倾角超过 10°，附着倾角 p95 为 86.98°。逐步回放显示，张开的指尖在最后下降阶段先撞到 cube，控制器继续追逐被推出去的 cube；旧采集门禁只检查阶段入口目标与最终任务成功，所以这类“最终放进目标但初抓已把 cube 推歪”的视频可以通过。

### 使用的解决方法和对应效果

- 将专家 XY 补偿从 6 mm 收敛到 3 mm，只修正初次夹取的系统偏差。
- 只对张指状态的最后下降段施加每控制步 6 mm 的最大 Z 位移，避免低控制频率下跨步撞击；运输、放置和 action 定义均不变。
- `precision-grasp-stable-v3` 在实际首次附着时要求双指垫真实接触至少连续 2 步，并检查预抓取 cube 位移/倾角、附着倾角/角速度、持物最大倾角与释放前持续附着。
- 每条合格 episode 立即接受，不规定失败恢复 episode 的比例，也不为等待某类轨迹拖慢采集。
- v3 数据不能追加进旧 v2 目录；检查器提供 `--require-strict-grasp-quality`，防止训练时把旧数据误称为严格数据。旧数据没有删除或覆盖。

修改后的 80-seed sweep 成功 78 次，首次附着倾角超过 10°为 0，附着倾角 p95/最大为 0.377°/0.398°，预抓取 cube 位移 p95/最大为 0.850/1.709 mm，持物倾角 p95/最大为 6.21°/6.55°，未发生持物后掉落。`tests/_smoke/strict_dataset` 的 3 个小规模 episode 全部通过严格采集与 Evo K=6 数据接口检查，共 410 帧。另有反例测试确认两个不连续的瞬时双垫接触不能凑够门禁要求；最终完整 `unittest` 为 42 项通过。这些数据证明专家轨迹门禁与采集入口可用，不替代正式规模数据采集。

## 68. π-MEM 可见失败历史与纠错微调

### 遇到的问题

π-MEM 的短期记忆不只是把 K 帧送进时间注意力；其纠错实验保留策略失败尝试作为短期历史，并用后续人类修正动作继续微调，使模型能在同一段可见历史中改变策略。旧工程已经有 K=6 因果时间视觉融合，但训练窗口一律均匀采样，没有识别“模型输入中实际包含失败且当前正在修正”的监督片段。稀少的自然恢复窗口因此容易被大量 routine 窗口淹没；仅有时序结构不能保证论文所说的错误后修正能力被训练出来。

### 使用的解决方法和对应效果

- 缓存索引升至版本 3，并对模型实际采样到的 K 个历史索引分类。只有历史索引中出现 `recover`，且当前为恢复后的 `recover/approach/descend/close`，才标记 `post_failure_correction`。
- 不读取未来 phase，不把“仅发生在 stride 采样间隙、模型没有看到”的恢复误标为纠错，避免标签泄漏或虚假短期记忆。
- 训练默认按当前数据的真实事件频次使用逆平方根权重构造 `WeightedRandomSampler`；它提高稀有纠错窗口的训练可见度，但不规定采集比例、不复制固定数量的失败 episode，也不删除普通任务/首次抓取对齐数据。
- 可用 `--no-memory_event_sampling` 做消融；日志输出事件计数和权重，便于审计短期记忆是否真的进入训练。

旧 v2 数据中可形成 39,819 个有效窗口：`grasp_alignment=15,786`、`routine=22,395`、`post_failure_correction=1,638`。原始纠错窗口约占 4.1%，自适应采样后的期望占比约 12.8%；12.8% 是数据分布计算结果，不是硬编码目标或采集配额。本轮没有增加与 π-MEM 短期记忆无关的其他论文结构。

旧 `stage2/step_best` 尚未用上述事件采样与严格 v3 数据训练，因此旧视频中的定位失败或夹错后停滞不会因代码修改自动消失。验收客户端新增逐决策/逐动作 JSONL：同时记录 raw 模型动作、实际 applied action、末端和 cube 状态、双垫接触、附着、推理时延与连续停滞步数。停滞检测只做观测和证据定位，不把 MuJoCo 真值输入策略，也不硬编码恢复动作。只有新 checkpoint 在同 seed 闭环评估中更快改变 raw action 并重新接近/夹取，才能声称复现论文的纠错效果。

## 69. 单卡 LoRA 直接 Python 启动更正

### 遇到的问题

普通Accelerate启动不一定等于脱离DeepSpeed，历史显式launcher和继承环境仍可能触发其分布式路径。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
```

### 修改前

将回到普通Accelerate误当作彻底移除DeepSpeed launcher。

### 使用的解决方法和对应效果

单卡直接Python启动，在分布式初始化之前处理不适合本机的DeepSpeed环境；pi-mem入口明确使用单卡设置，真正分布式训练仍保留其原路径。

单卡路径为DistributedType.NO且单进程，不会因旧launcher含义或继承变量误走CPU offload；不更改正常训练算法。

## 70. 真实空夹纠正示范、历史前缀与训练快照

### 遇到的问题

正常成功示范没有覆盖已观察到的空夹纠正状态。直接监督失败前缀会把错误动作作为专家目标；中断采集仍标记 `running` 时，也不能称作已完成的训练快照。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/collect_data.py
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
```

### 修改前

缺少从真实策略失败日志复现空夹并接入专家纠正的采集入口，无法验证失败经历是否进入有效历史。

### 使用的解决方法和对应效果

- `--recovery-actions` 是可选补采：复现记录请求、确认安全空夹、保留约 1 秒真实失败响应，再接入专家纠正。
- `policy_context` 前缀只保留为历史，不作为专家动作目标；阶段与来源记录不输入策略。
- `--append` 保留已有 episode；示例中的两条来自当时两条可复现日志，不是固定采集比例。
- 写入、原始检查、加载器和开环检查共同验证前缀来源及时间；缓存索引更新为 8。
- 中断及时记录尝试状态；未结束的 `running` 快照拒绝作为已完成数据使用。checkpoint 保存实际数据快照和元数据指纹。
- 当前默认按 MINT 将全部有效窗口均匀 shuffle；事件加权是显式可选项。本条更正 `README.md` 相关问题条目关于历史版本默认加权的描述，旧记录保留。

两条临时纠正示范的历史前缀为 61、68 帧；前缀均未进入专家动作目标，实际 K=6 历史含失败后的观测，生成 9 个纠正窗口。现有开环工具重放核验了真实空夹和动作前状态。

三任务各一次真实图像前后向通过：448、K=6、无 LoRA、默认时间视觉参数更新，梯度有限且非零；优化器更新和 checkpoint 写入均为 0。未启动正式批量采集或完整训练，不能据此声称模型已学会恢复或成功率已提高。

证据为 `/home/user/mujoco+evo/tests/_diagnostics/review_20261019/final_integrity.json`、`training_startup.json` 和 `final_correction_replay/report.json`。

## 71. 开环检查的诊断选项和验收口径

### 遇到的问题

数据重放、专家观测下的动作预测和反馈闭环能验证的内容不同。将修复原因与试验细节持续加入 README，会干扰正常命令使用。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/README.md
/home/user/mujoco+evo/Problem_Fix/README.md
```

### 修改前

README 混入较长的开环诊断与验收说明；可选纠正补采命令缺少明确用途标注。

### 使用的解决方法和对应效果

- 继续使用已有 `open_loop_test.py`，不另建正式测试入口。README 保留原有命令和新增参数的简短用法，修复原因、实现和验收记录追加在本目录。
- `--audit-only` 只审计数据，不加载策略；新数据与旧权重环境不匹配时，纯数据审计记录差异，实际模型评估仍拒绝不匹配。
- `--replay-steps 600` 最多重放 600 个记录动作，核对动作前状态；Task1 使用数据记录的抓取参考偏置。
- 默认比较完整历史、仅当前观测和仅当前图像；`--no-memory-ablation` 跳过该消融。输出变化只能说明对历史敏感，不能证明选择正确。
- `--drawer-selection-probe --closed-loop-max-steps 40` 追加四目标、完整历史与仅当前观测的短段接触选择检查。目标只用于初始化与评分，不输入策略。
- 单独报告独立物理观察与正式判定的分歧；开环动作误差不推算闭环成功率。

前轮最终 8 条临时数据通过数据审计，模型查询数为 0；Task3 三条轨迹的全部动作前状态重放对齐，其中两条纠正轨迹的空夹得到独立核验。终端动作后状态及图像重放对齐不在该结果的认证范围内。

README 文档整理单独核对全部 Bash 命令块，保留用户当前命令、参数、路径及已有问题记录；不执行文档中的采集、训练或清理命令。

## 72. 最新模型关键动作未学准、纠正历史资格与串行资源限制

### 遇到的问题

2026-10-25 新 best（step 70830）的 Task1/3/4 分别为 9/10、8/10、0/10；新 final（step 80000）此次只新评估 Task4，结果为 0/10。不能把 final 目录内早期 Task1/3 残留视频混入当前结果。前次 Task4 1/10 与本次 0/10 不能当成已通过验收，也尚不能在未控制 seed、权重和环境的情况下认定某一改动是唯一退化原因。

在正确专家观测上，当前模型也存在闭合、释放与安全抬高阶段预测错误。另有两条真实失败前缀物理上能够开始纠正，但是否仍有可见目标历史尚未验证。多个重型 Python 工作流并行及单模型加载期间的长 I/O 等待影响用户机器可用性，用户明确要求限制并发与内存。

### 修改文件

```text
/home/user/mujoco+evo/AGENT.md
/home/user/mujoco+evo/Problem_Fix/README.md
/home/user/mujoco+evo/tests/_diagnostics/review_20261026/
```

### 修改前

缺少单独的要求与约束基线；阶段性检查容易被误当成模型验收。未核验真实纠正时刻实际 K6 采样是否仍含目标。多个诊断进程容易形成资源争抢，未完成的模型加载不能当成完成的数值比较。

### 使用的解决方法和对应效果

- 按用户指定的单数文件名建立根目录 AGENT.md，记录目标、限制、文档归属、证据层次、未通过项、当前授权边界和每轮更新规则；不另建第二份要求文档。
- 最新权重和视频身份分别核对。保留原视频与正式数据，不修改模型架构、统计、采样默认或参数来掩盖误差。
- 对照 MINT 官方提交 5fd14b015013c4fd0aacf5f8f48f868ca9b870a2 的两阶段训练与本地 π-MEM 论文。当前训练初始化、batch 与可训练模块并不等同官方配方；这属于待验证学习因素，不据此直接认定欠拟合或保证再训练有效。不能直接照搬论文的 action-expert detach，因为此适配没有对应 FAST 监督分支。
- 两条真实 Task4 失败前缀先做无渲染物理时序探测，再单独用正式 448 分辨率重跑展示分割可见性，按现有 history_indices 计算关键时刻 K6 采样。未调整 K、时间、相机或目标真值输入。
- 重任务串行，不重复加载模型，额外子代理已结束。确认本次大型模型进程停止后才运行单一小规模可见性核验；不结束用户其他程序，不重启环境。
- 未启动新的 optimizer 更新，未写新权重。有限训练方案记录在诊断目录，未把方案当作已执行；此前只允许少量前后向的要求及后续模型交付目标之间的训练范围仍需明确。

证据为 review_20261026 下的 action_audit.json、best_open_loop_baseline/report.json、learning_audit.md、video_decode_alignment_audit.json、task4_recovery_physics_probe.json、task4_recovery_cue_gate.json、combined_regression.json 和 bounded_training_plan.md。README 哈希保持本轮开始时的 003d8446493870131533dcefe3f32029f14fac530c8b29db9e0c10f9dc7551c4。

当前结论：已定位实际动作学习与纠正覆盖问题，但没有生成改进权重，Task4 尚未通过验收。不得将本条的审计、要求文档或资源治理称作成功率已经提升。

用户明确允许等待后，冻结原环境的唯一模型进程已正常完成并以 0 退出。7 个真实专家窗口中，上下文和归一化动作实际均为 FP32；同噪声 Euler 精度候选仅改变平均约 0.003～0.007 mm，误差有升有降。因此未采用该候选，不再将其作为 Task4 的主要解释。

四个目标各做完整历史与仅当前观测对照，共 320 次请求；当前图像/状态历史相同、四种目标展示历史不同的受控条件通过核验。两组均在各 40 步内未确认选中把手，即各 4 条未决，不报告成完整任务成功率或“记忆没有作用”。证据为 paired_euler_real.json、best_selection_isolated.json、isolated_policy_comparisons_serial.log 和 isolated_policy_resources.jsonl。

## 73. 缓存重建隐式启动最多 16 个进程

### 遇到的问题

用户已明确限制内存与重任务并发，但 LeRobotDataset 重建派生窗口仍硬编码 min(16, episode_count) 个进程，独立于训练 DataLoader 的 --num_workers。仅降低训练侧并发不能限制这条路径；即使只希望串行，原路径也会创建进程池。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/tests/test_short_term_memory.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

缓存重建最多启动 16 个进程，调用方没有单独控制项；训练的 --num_workers 不影响它。

### 使用的解决方法和对应效果

- cache_num_workers 默认 1，直接在当前进程 map 处理，不创建子进程；只有显式大于 1 才创建对应进程池，并限制到实际 episode 数量。
- 正式入口新增可选 --cache_num_workers，传给加载器；正整数以外的配置拒绝。原 --num_workers、batch、K、448、时间间隔、采样和损失不变。
- 保留既有终端输出结构、缓存格式、排序与事件配对逻辑；并发数量不参与模型输入指纹，未使旧缓存自动失效或修改正式数据。
- 用户现有命令不用修改；README 未改。默认串行已写入要求基线。

首次新增断言把缓存字符串路径当作 Path 而失败，属于本轮测试实现错误，已用 Path 转换修正；原日志 cache_worker_regression.log 保留，修正后完整结果为 cache_worker_regression_fixed.log。未以此改动宣称模型正确率提高。

## 74. 现有 best 与真实训练窗口的无更新前后向验证

### 遇到的问题

之前随机图像、随机动作或新初始化动作头的梯度检查，不能代替当前训练所得权重在实际数据上的学习路径验证。模型关键动作仍错，需要先核实真实输入、掩码、损失和梯度，再决定参数更新实验，避免盲改网络。

### 修改文件

```text
/home/user/mujoco+evo/tests/_diagnostics/review_20261026/trained_checkpoint_backward_pilot.py
/home/user/mujoco+evo/tests/_diagnostics/review_20261026/bounded_training_adapter.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

真实权重的关键窗口前后向证据不足；直接调用 train(max_steps=0) 仍会构建训练状态并写最终 checkpoint，不符合只做少量前后向的限制。

### 使用的解决方法和对应效果

- 复用正式 prepare_dataset、custom_collate_fn、set_finetune_flags 和 FP32 masked_velocity_loss，严格载入当前 best（step 70830），每任务一条真实缓存窗口，448、K6、1 秒间隔、原增强、无 LoRA。
- 缓存只读：无有效索引就拒绝，禁止重建/删除或写索引；环境约定、数据快照和归一化边界与 checkpoint 对齐后才前后向。无 DataLoader 子进程，不构建优化器或调用 checkpoint writer。
- 参数哈希按 1 MiB 块传到 CPU，避免整份模型副本占用内存；完成后再次逐参数哈希并检查正式缓存索引身份。
- 单独准备有限适配划分与配置，默认准备模式不导入 Torch、不更新参数。实际训练路径仍调用正式 native.train；仅显式训练选项可进入，并且目前尚未获准执行。适配排除集与真正未见泛化严格区分。

进程退出码 0。Task1 episode0/frame45、Task3 episode12/frame49、Task4 episode79/frame72 的实际 flow loss 分别约 0.03321、0.06427、0.04228；仅为单窗口、单噪声结果，不当作平均训练损失。五个时间视觉层 QKV 及动作头梯度均有限非零，冻结语言主干和首个视觉层无参数梯度。

全部参数前后 SHA256 都为 40faeeb707a078ee4f74207d84276fe6bfb8c803b6ac18574235239c96b9507b，正式缓存索引未变，参数更新 0、checkpoint 写入 0。峰值 CUDA allocated 约 3143 MiB，进程峰值 RSS 3081712 KiB，保持单模型串行。首次诊断脚本未将 scripts 加入启动路径导致导入失败，已修正调用路径；原失败日志保留，没有修改正式模块导入语义来迁就脚本。

准备模式退出码 0：每任务适配 90/排除 10 个回合，共适配 50453、排除 5630 个窗口；Task4 排除四目标计数为 3/1/3/3。原权重已经见过这些数据，不能称为源训练未见集。实际更新尚未执行，不能声称获得新模型或正确率提高。

证据为 trained_checkpoint_backward_pilot.json、trained_checkpoint_backward_pilot_retry.log、bounded_training_prepare.log、adaptation_split.json、bounded_training_ready_config.json。当前确认的是这三个真实窗口的学习通路可用，未证明模型已学准动作或泛化成功。

## 75. 单条不可用的真实失败前缀阻断整轮纠正采集

### 遇到的问题

真实失败记录不一定能提供合格的纠正前缀。当前 best Task3 的三个候选中，seed 79385487 在前缀第 51 步发生危险接触，79385489 在第 58 步已经抬起物体，79385490 则能复现安全空夹并由专家纠正。旧入口对前两种情况抛普通 ValueError，collection_main 按异常退出整轮，因此正常命令无法继续处理第三条可用记录。这直接阻断所需的真实纠正监督，并非要取消质量拒收。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

无法安全复现空夹，或干预前已自行恢复，会让纠正采集全局退出。合格的后续来源不能被尝试。

### 使用的解决方法和对应效果

只把这两类物理不适用情况作为 RecoveryContextRejected；采集器持久化 seed、原因与 safe_empty_grasp_context 拒收结果，继续下一条真实记录。损坏来源、编码错误及程序异常仍传播并中止，不改专家接触、成功或安全规则。不修改 README、原数据或已有权重。

正式 collect_data.py 入口在独立 outputs/repair_20261028/task3_correction_candidate 目录完成：3 次尝试、2 次拒收、1 条保存，status=completed，进程退出码 0。保存的是原 best episode4/seed79385490 的真实前缀及专家后续，严格 failed_checks 为空。原日志路径、SHA256、run_id 和 episode 均记录，未伪造来源。

正式 check_dataset.py --require-evo 入口退出码 0：139 帧的原始数据、视频和环境约定检查通过，448/K6/horizon14 的 Evo 输入接口通过。另用生产 LeRobotDataset 构建全部窗口：57 帧失败前缀仅作历史，82 个合法监督起点全部位于专家接管之后；第一个纠正窗口仍含真实失败历史，失败动作未作为标签。

核验过程中第一次调用误用不存在的 Task.create_env，实际未执行物理验证；已改为正式 make_env，原调用错误单独保留在 task3_real_correction_invocation_error.json，不计为通过。修复前直接复现的物理结果见 task3_real_correction_feasibility.json，完整监督审计见 correction_supervision_audit.json。

本条只证明一条真实纠正监督可用、不可用前缀不再阻断整轮；没有模型加载、参数更新或新模型成功率结果，不能称为达到 80%。MINT 官方训练源码已再次核对；未发现支持修改 flow 符号、替换注意力或统一加深的确凿证据，因此没有同时叠加这些改动。

## 76. 三组有上限的真实适配对照均未形成可采用的改善

### 遇到的问题

用户指出此前提出的小训练并未执行，却结束工作，且项目进度不透明。仅真实纠正示范通过不能说明模型学会。本轮用实际参数更新和固定动作对照判断候选是否有进展，不把100步当成收敛或闭环成功率证明。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/bounded_correction_pilot.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/parameter_ablation.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

尚无本轮新参数更新的效果结果；旧 Task4 best/final 合计0/20，不能宣称修好。原始训练阶段有共享视觉更新与随机动作侧同时学习的差异，但还不能证明它是根因。

### 使用的解决方法和对应效果

- 诊断适配器调用生产 train.train、生产数据加载/增强/损失/AdamW/调度/保存路径，448/K6/composed/无LoRA；一份模型串行运行，物理batch为1、无数据子进程。正式命令及默认采样不改。
- 使用独立的192窗口排练集合，每任务64个，Task3含32个真实纠正窗口。此为有限诊断抽样，不是将正式采样改成任务等概率。动作归一化保持源checkpoint统计；纠正样本没有替换它。
- 对照19个观测窗口、每窗3个固定flow种子，Task4包含四目标的接近动作。窗口排除本次适配，但旧权重见过原数据，不能称为未见泛化；57查询不等于57独立回合。
- 先进行原部分视觉联合更新100步，再独立恢复源权重做冻结视觉100步；另核对源实际学习率后做同样的冻结视觉低学习率100步。每组有100步上限，合计训练计时上限900秒，不是每组900秒；三组实际合计300次更新、264.49秒训练。构建模型、动作查询、存盘时间另计。
- 第二、三组逐字节复用第一组窗口划分。第二组训练前原始预测与第一组最大绝对差0，确保初始权重、观测和种子一致。

三组进程均以0退出，全部候选只在独立outputs目录保存，没有替换best/final。Task4固定动作查询的平移误差如下，误差按正式执行前的逐轴±12mm限幅计算，不含后续IK和物理跟踪：

| 状态 | 平均误差 | 反方向查询 |
|---|---:|---:|
| 原权重 | 4.78mm | 0/33 |
| 部分视觉联合更新100步 | 8.74mm | 6/33 |
| 冻结视觉更新100步 | 7.76mm | 2/33 |
| 冻结视觉、沿源学习率量级100步 | 4.80mm | 0/33 |

源checkpoint step70830的实际学习率为3.287806e-7，前两组新适配的峰值为1e-5，约为30倍。两组是重置优化器的局部适配，不是原优化器状态的正常续训，不能据此断言正常续训100步必然退化。低学习率组基本保持基线，但没有明确改善；不采用为交付权重。

零更新参数互换也未挽救候选：原视觉+更新动作头的Task4误差7.68mm，更新视觉+原动作头8.24mm。它只说明这次局部适配的两侧变化都可能造成副作用，不证明从头正确分阶段训练无效。论文的随机动作专家对齐阶段并未被这次旧权重100步冻结实验完整复现。

第一组Task1/3误差计算曾错误使用向量长度限幅，已根据正式代码改为逐轴限幅并用保留的原始输出重算：Task1约3.96→3.97mm，Task3约2.69→2.81mm。原文件保留，更正记录为corrected_action_metrics.json；不隐去原错误。夹爪指标只是分数与专家差小于0.25，不是物理抓取成功。

第一次参数互换因关闭增强改变缓存指纹而被只读保护拒绝，未改正式缓存；保持原构造配置、仅查询时关闭增强后完成。适配日志初次重定向误用Linux路径作为PowerShell输出路径，没有启动训练，已改为工作区路径。上述失败不计为通过。

证据目录为bounded_pilot、bounded_pilot_frozen_visual、bounded_pilot_frozen_visual_source_lr、parameter_ablation_retry；原统计和环境来源不伪装成统一的新部署约定。所有候选标记diagnostic_only，closed_loop=not_run，passed_80_percent=false。当前模型仍未验收。

## 77. 对齐官方LeRobot Evo-1：屏蔽回合末尾人为补齐的动作监督

### 遇到的问题

本地loader将最后一行动作重复到horizon长度，却把超出回合的重复动作全部算进loss。对照Hugging Face官方Evo-1的modeling_evo1.py，官方通过action_is_pad排除这些补齐位置。当前三任务缓存各有9100个此类未来时间槽（不是9100个真实数据帧）；它们是张量补齐，不是真实专家决策。192窗口小对照集合中只有67个补齐时间槽，因此也不能把它当成上条全部退化的解释。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

末尾重复行参与有效监督；直接改变缓存源指纹又可能导致原派生窗口重建，且旧权重完整续训不能暗中改标签语义。

### 使用的解决方法和对应效果

- 加载manifest时记录按arm、dataset和episode区分的真实回合长度；读取样本后，用start+h<length与原维度mask相交。最后一个真实动作保留，之后人为补齐部分不贡献flow输入噪声、loss或梯度。图像、状态、动作数值及原统计不改。
- 原缓存身份cache_source_signature保持旧规则；有效训练身份source_signature额外记录新掩码契约。原有有效缓存可直接读取，不重写pkl或索引，完整续训也能辨识监督规则变化。
- 新训练、weights-only初始化默认启用；完整续训默认从原checkpoint配置继承，旧配置缺字段时沿用旧规则。显式改变规则的完整续训由已有sampler契约拒绝，不偷偷沿用旧sample cursor。
- CLI和直接train(config)共用解析，配置记录实际布尔值；旧命令不需为了本修复改写。README未修改，不影响旧checkpoint推理，也不重采原始视频。

生产loader读取原56083个缓存窗口，三任务各检查首个有效窗口、倒数5步和最后一步共9个实际样本；Task4首个有效监督窗口是frame44，人展示阶段没有被错误纳入监督。图像、状态、动作张量与旧语义一致，原索引和抽查pkl的SHA256不变。有效未来动作及有效维度保留，末尾补齐关闭。

生产masked_velocity_loss验证有效位置仍产生梯度，给补齐位置注入12345的错误目标不产生loss或梯度。新/旧训练指纹不同而缓存身份相同；旧语义完整恢复通过，新语义不能静默恢复旧采样契约。新训练、weights-only、旧/新checkpoint完整续训和非法配置解析均按预期。正式train.py --help退出码0，未加载模型或训练。

初次诊断错误假定Task4存在frame0监督窗口，导致样本数断言失败；已改为首个实际可训练窗口frame44，原始失败不算生产缺陷。修正后运行退出码0，证据为terminal_action_padding_verification.json、terminal_padding_audit.log、train_help_after_padding_fix.log。另一个gpt-6.1-sol/medium代理纯文本复核未发现确认问题，没有额外运行Python或读取测试。

本条仅修正监督语义，尚未在修复后重新训练出改善权重，不能宣称成功率提升。来源为官方代码：

- https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/evo1/modeling_evo1.py ，_prepare_actions的action_is_pad处理。
- https://huggingface.co/docs/lerobot/en/evo1 ，官方分阶段与数据接口说明。此集成当前默认单观测，采用native-HF InternVL权重布局，不可直接替换本项目K6或把旧checkpoint当作兼容。
- https://arxiv.org/html/2511.04555v1 ，3.3节的动作专家对齐与全模型微调。
- https://www.pi.website/download/Mem.pdf ，IV-B及附录B的真实误抓后干预和失败历史保留。当前纠正链沿该原则，不能以K6结构替代纠正训练。
- https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py ，时间集成针对目标关节位置；当前世界位姿增量不能无条件照搬。
- https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py ，EMA/验证/回合执行组织可供对照；不为复制EMA再驻留第二份VLM，不把他项目成功率当作本项目证据。

## 78. 完全冻结VLM时固定特征模式；四目标纠正小对照未通过效果门槛

### 遇到的问题

冻结参数并不关闭随机层。本地InternViT配置drop_path_rate=0.1，在完全冻结VLM、只训练动作专家时，model.train()仍让视觉特征随机变化。Hugging Face官方Evo-1的_keep_frozen_embedder_eval会在此阶段保持embedder.eval()。这是明确的训练模式差异，不等于已经证明它是Task4失败的唯一原因。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

只设置requires_grad=False，冻结视觉仍保留随机深度；旧配置也没有声明是否固定冻结特征。

### 使用的解决方法和对应效果

- 完全冻结VLM且frozen_embedder_eval=true时保持embedder.eval，动作头仍按训练模式运行；部分视觉或其他VLM参数可训练时仍保持train，不关闭其正常训练路径。
- 新训练和weights-only初始化默认启用；完整续训继承原checkpoint的值，旧checkpoint缺字段时保留false。显式改变完整续训的模式会报错，要求使用新适配，不暗改原训练条件。
- set_finetune_flags支持同实例重新开启明确要求微调的模块；不新增state_dict权重，不改变server.eval下的输入输出。
- 继续遵守单模型、448/K6/1秒/composed/无LoRA；未修改README、正式采样、成功判定或抓取深度。

真实模型、相同观测、零optimizer更新的对照：冻结VLM仍train时，两次特征最大差32.296875；固定eval后为0。临时启用一个VLM参数后会恢复train，动作头保持可训练。证据为bounded_pilot_real_corrections/frozen_mode_verification.json。

随后综合候选通过正式train.train完成400次有上限更新，退出码0，实际训练285.55秒；与此前三组累计700次更新、约550.04秒，未进行长训练。该组同时含四目标真实纠正、尾部mask及模式修复，所以不能把综合结果单独归因于其中一个改动。

| 固定动作查询子集 | 原权重平移误差 | 候选平移误差 | 其他证据 |
|---|---:|---:|---|
| Task1原窗口，12查询 | 3.96mm | 3.78mm | 夹爪分数接近11/12不变，反向1次不变 |
| Task3原窗口，12查询 | 2.69mm | 2.82mm | 夹爪12/12，未见改善 |
| Task4原窗口，33查询 | 4.78mm | 5.02mm | 夹爪30/33、反向0次均不变 |
| Task4纠正轨迹内排除帧，24查询 | 12.12mm | 11.96mm | 反向仍4次，夹爪24/24 |

这组没有通过关键动作门槛，拒绝部署，未替换正式best/final。纠正验证帧与适配帧来自同轨迹，只是轨迹内插值检查；原窗口虽然排除本次适配，但旧权重见过原数据。81次预测不是81个独立回合，不换算任务成功率。证据为bounded_pilot_real_corrections/result.json及before_actions.json、after_actions.json；没有新闭环成功率。

最新用户将短训练候选的观测严格闭环放行目标提高到Task3、Task4各90%，保护Task1，Task4覆盖四目标；正式最终目标至少各80%。短训练的训练集拟合/开环动作一致率达到90%也不能保证长训练80%，未达到的候选继续记未通过。

本轮再次联网核对的来源及边界：

- https://github.com/huggingface/lerobot/blob/cc2fbf08976c6edd14edb001616ae79f3121de96/src/lerobot/policies/evo1/modeling_evo1.py ：冻结embedder模式及action_is_pad处理；只做适配，不复制整套native-HF权重布局。
- https://arxiv.org/html/2511.04555v1 ，3.3及4.4.2；https://github.com/MINT-SJTU/Evo-1 ：先对齐动作专家再联合微调。旧联合训练权重上冻结400步并不等于从随机动作头完整复现两阶段流程。
- https://huggingface.co/zuoxingdong/evo1_libero ：社区发布的两阶段LIBERO模型可作训练与验证组织参考。模型卡报告的是另一基准、两H100/global batch128、单seed评估，不能把其成功率移植为本项目保证，也未下载替换本地权重。
- https://www.pi.website/research/knowledge_insulation ：方法同时训练FAST离散动作、连续动作专家并控制梯度，且混合更广数据。只做detach而没有表示学习监督会切断本地唯一动作学习信号，因此未盲加梯度阻断或声称复现KI。
- https://compliant-residual-dagger.github.io/ 及 https://github.com/yifan-hou/cr-dagger ：接触任务的纠正分布、微调不稳定与残差学习证据。其力反馈/力控制和残差动作接口不等于本地世界位姿增量接口，未直接引入另一策略来掩盖基础模型错误。

以上文献支持检查具体实现与实验条件，不支持保证本地短训练后长训练必达某个成功率。当前仍未交付达到80%或90%的Task4模型。

## 79. 实际训练窗口的闭爪拟合、固定flow时间误差拆分及诊断推理mask

### 遇到的问题

此前小适配主要报告排除窗口，没有充分区分“训练窗口本身没学会”和“仅排除窗口退化”。旧小训练诊断evaluate又将训练batch的action_mask直接传给推理；启用末尾padding mask后，这会引入部署时不存在的回合剩余长度信息。正式服务和open_loop本来使用公开任务维度掩码，没有发现同样的生产推理错误。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/bounded_correction_pilot.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/learning_fit_audit.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/summarize_learning_fit.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

原诊断推理复用含回合末尾padding的训练掩码；增强重建按817假设，还未读回候选runtime核对；没有把首步夹爪、首步XYZ与后续夹爪的flow训练误差拆开。

### 使用的解决方法和对应效果

- 小训练诊断evaluate按正式TASK_CONTRACTS生成公开维度掩码，14个未来步都不使用回合剩余长度；训练loss仍保留正确的末尾mask，不取消监督修复。旧结果保留，没有重标或覆盖。
- 从实际training_state_rank_00.pt核实sampler seed817、epoch2、cursor16，重建400步中每个受检样本最后一次出现的增强seed。增强概率仍按实际采样执行，不将同一输入重复出现冒称多样性。
- 单实例、零更新比较源与400步动作头；固定flow seed31/47/83，分开实际training和excluded_from_adaptation。固定t=.1/.5/.9拆分原MINT目标，检查各项加权后能还原生产masked_velocity_loss，绝对差小于1e-6。
- 后一诊断将12个上下文的冻结视觉特征保存到独立诊断目录，供后续仅动作头的无更新检查复用，避免再加载第二份VLM。正式入口不依赖这些诊断脚本。

第一轮learning_fit_audit_20261028T172201Z退出0：28个窗口、38个干净/实际增强上下文，两套头各三个flow seed，共228次生成。第二轮learning_fit_audit_20261028T172936Z退出0：12个干净上下文、72次公开掩码生成、216次固定t训练前向和6次尾部掩码对照。两轮optimizer更新均0，没有写新checkpoint。

| 实际训练子集 | 源头 | 400步头 | 解释 |
|---|---:|---:|---|
| Task4 object_close，4窗口×3seed错误开爪 | 12/12 | 12/12 | 训练关键窗口本身仍未拟合；这些4窗口只含3个抽屉目标，不冒称四目标完整抓物验收 |
| Task4 handle_close，四目标4窗口，XYZ均值 | 9.16mm | 9.30mm | 闭爪请求正确不等于把手位置正确 |
| Task4 object_close，t=.1首步夹爪MSE | 3.8148 | 3.3849 | 归一化velocity误差，不能当毫米或成功率 |
| 同上，t=.1后续夹爪MSE | 0.1301 | 0.0914 | 后续动作比首步转换更容易，整体loss下降不能放行 |
| Task3 close，1训练窗口×3seed错误开爪 | 3/3 | 2/3 | 仍未通过；增强上下文400步头仍3/3错误开爪 |

受检Task3训练尾部release窗口（episode19/frame99）的公开/尾部掩码对照，XYZ差约0.145–0.205mm，开闭判断都仍为打开。它证明诊断口径需统一，但解释不了本批厘米级运动偏差；不能把这个训练窗口的结果泛化到旧报告所有排除窗口。

再次联网读取MINT官方flow_matching.py与train.py、HF固定commit的modeling_evo1.py：训练目标/正向积分与当前一致；HF同样将末尾action_is_pad用于训练，将公开动作维度用于推理。总体有效项平均损失与MINT一致，首步夹爪只占Task4完整窗口98项中的一项；这不是直接更换loss或加权的依据。本轮轻量GPT6.1-sol/medium只读审查未找到新的确定生产学习bug，没有启动并行Python或读取tests。

结论限定为这些真实窗口的拟合失败，不是新闭环成功率。400步候选继续拒绝采用，正式best/final、数据、视频、README不改；Task3/4各90%候选门槛与最终各80%目标仍未达到。后续32次更新方案只准备，不因用户催促扩大训练授权。

补充验证：`bounded_head_fit_probe.py --verify-only`仅加载生产动作头和已保存的真实冻结视觉特征，6个闭爪窗口预测与完整模型差值逐值为0。固定t=.1时，所查共享MLP第一层中Task4首步夹爪对正式loss的梯度范数约0.68–1.07，其余有效项约0.68–1.24，方向余弦约-0.012至-0.002；未发现首步梯度消失或在该层严重抵消，不能据总体平均的分母就认定必须重新加权。Task1/3对应路径同样有有限非零梯度。未创建optimizer，0次更新，全部头参数哈希前后同为dc2b92ddc649828a379b7f2b085adea578a5ffd6d06615a516575ab824f8dff9。结果`head_fit_gradient_20261028T174304Z/result.json`，退出0；首次将requires_grad张量转scalar有一条PyTorch提示，记录值和梯度检查有效，随后改为detach读取，未重跑来美化原日志。

该脚本默认只生成可审查方案，已实际执行默认准备入口；只有--execute才会进入待授权的最多32次更新、最多新增60秒训练。原累计700更新/550.041秒不会清零。新方案不驻留VLM、不写checkpoint，不代替正式训练或最终闭环验收；本轮没有执行--execute。

## 80. 闭爪定点改善伴随释放退化：补齐小对照的完整阶段排练并限制更新范围

### 遇到的问题

用户在32步具体方案之后要求继续修改、不得因未达标停止。本续轮继续原累计900秒训练预算内的有限验证。32步能够修正受检闭爪，却损伤释放；旧小对照的Task4起始窗口池又只列8个阶段，漏选松开把手、撤离、转姿态、运输和释放等11个阶段。Task1/3也漏选部分运输和结束阶段。这是诊断选样缺漏，不是正式全窗口加载器没有这些数据；horizon14邻近窗口可能偶然监督后续阶段，不能说遗漏阶段完全未参与loss。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/bounded_head_fit_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/phase_complete_head_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/cached_decoder_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/summarize_phase_alignment.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

定点11个训练窗口偏重闭爪，未能保护整体阶段；旧Task4小对照只保证四个目标的approach，后续阶段主要评估单个目标。高峰值学习率的全头更新容易连同运动和释放一起改变。

### 使用的解决方法和对应效果

- 完整阶段诊断选择256训练窗口、172个适配排除窗口，Task1正常8阶段、Task3正常9阶段、Task4四目标各19阶段均覆盖。失败前缀只作历史；已有真实recover单独保留，不制造固定失败比例。
- 专家阶段首、末作为当前观测单独验收，三任务分别报告XYZ、旋转及夹爪；训练后核对实际执行更新清单，而不只核对候选池。
- 正式全窗口均匀采样、原flow损失和step_best逻辑保持。诊断子集为阶段覆盖选样，不伪称与MINT全数据采样分布完全相同。
- 冻结视觉特征只计算一次并保存，后续单头工作流复用，不重复驻留VLM。对全头、最后线性层和完整两层输出头分别做有限更新；新候选独立保存，原best/final不动。训练计时累计原全部组，不重置900秒上限。

| 组 | 更新/实际训练秒 | 关键结果 | 处理 |
|---|---|---|---|
| 11窗口定点 | 32/13.97 | Task4抓物误开12/12→0/12，把手XYZ9.30→2.84mm；Task3释放1.35→6.64mm、2/3误闭 | 拒绝采用，未保存新权重 |
| 完整阶段、全头 | 256/9.37 | 新四目标面板抓物误开10/12→0/12，但Task1平均XYZ4.89→6.22mm、Task4接近5.96→8.11mm | 拒绝采用 |
| 只训练最后线性层 | 512/12.96 | 保住大部分原动作，但抓物仍10/12误开 | 未通过 |
| 只训练两层输出头 | 512/5.45 | 抓物仍10/12误开；Task4接近5.96→6.47mm | 未通过 |

这些组都退出0，固定面板每窗口用seed31/47/83。两个输出头组在更新前的全部预测与原完整阶段before逐值核对，冻结的其余参数SHA256前后相同。三组完整面板每组1032次生成，但不是1032个回合。源checkpoint见过原数据，适配排除不等于未见泛化；纠正数据也只涉及少量轨迹，不冒称恢复能力通过。

累计至两层输出头组为2012次有限更新、约591.79秒optimizer训练。VLM编码和无更新评估另计，不伪装成训练耗时；没有启动长训练。实际阶段更新覆盖包含全部正常阶段和真实recover，仍没有可采用候选或新闭环成功率。

完整阶段结果目录phase_complete_alignment_20261029T003844Z，两个输出头组为decoder_alignment_20261029T010547Z与readout_alignment_20261029T011519Z。当前驻留单头继续做零更新模块反事实，不能把“输出头不够”直接升级为网络结构错误。

参考MINT官方两阶段/原损失，以及 https://github.com/AnanyaKumar/transfer_learning 的冻结表示/LP-FT实验组织。后者是一般迁移学习研究，不是本项目机器人成功率保证；其读出限制对照未通过，因此没有改为正式默认训练策略。

## 81. weights-only冻结视觉时保留源FP32存储，避免先BF16复制后无法恢复

### 遇到的问题

原_prepare_temporal_parameter_precision同时依赖finetune_temporal_vision与fp32_temporal_parameters。源checkpoint的时间视觉参数已经为FP32时，新训练明确冻结视觉会让构造阶段跳过提升，随后load_state_dict按目标BF16存储复制，提前舍入源值。之后再.float无法恢复。已跑的诊断用覆盖冻结标志、保持源FP32配置，未走该路径；本问题不能被反用为此前所有策略失败的解释。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
```

### 修改前

存储精度与梯度开关绑定；weights-only初始化不记录源FP32参数，新的冻结checkpoint即使训练时补提升，未来服务重建也可能再次舍入。

### 使用的解决方法和对应效果

- 仅在非LoRA的weights-only初始化、仍要求FP32时间参数时，从已读取的实际state dtype收集时间视觉norm1/attn/ls1参数；copy前提升目标存储。参数名使用模型实际named parameters白名单，不接受任意norm、buffer或未知名称。
- 新run配置保存checkpoint_fp32_temporal_parameter_names；EVO1共享构造消费显式声明，独立于当前是否训练视觉。requires_grad不被改变，optimizer仍在加载后创建。
- 完整续训在构造前继承声明并拒绝擅改；服务/正式开环核对config.json与checkpoint client_state的声明一致。
- 旧缺字段构造仍执行原有逻辑，不全局删除finetune gate，不修改旧checkpoint。没有增加CLI参数或更改终端输出格式，README不动。

checkpoint_precision_20261029T010545Z退出0。使用真实源checkpoint的layer3 norm1.weight/bias/ls1值，在小型同名模块中复现原BF16复制最大差0.0070033073，修后最大差0。提取并调用实际生产加载、保存和配置继承函数，保存后构造/加载保持源值，前向逐值相同；完整恢复的声明继承及不一致拒绝通过。训练/冻结与FP32开关四种旧构造组合的参数和值保持一致；4种非法声明拒绝。

本项0次optimizer更新、0个完整VLM实例。它验证源值保留和存储契约，不是完整模型闭环或成功率验证；当前训练诊断的源FP32路径本来就保留，不能据此重标其失败结果。修改前Evo1.py/train.py另存precision_baseline_20261029，原生产基线与记录保留。

## 82. 受限注意力更新与真实四目标验证：拒绝单阶段改善掩盖任务退化

### 遇到的问题

只更新Q/K能修正部分抓物闭合，但仍改变接近运动；完整600步真实反馈又出现错误开屉。前轮工作摘要错误地把该回合写成“没有选中”，还把Task4连续开度.25错当提前闭爪。原始JSON显示目标3、实际选中/打开2，选择时历史仍覆盖展示。专家接近把手本来就要求.25，这个请求不属于已证明的模型错误；不能据错误标签增加开爪监督。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/cached_qk_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/reconstruct_feedback_start.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/qk_interpolation_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/paired_prefix_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/noise_time_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/controlled_initial_actions.py
```

### 修改前

此前只验证专家和有限阶段误差，未完成该候选的真实全过程反馈；诊断又可能把连续夹爪目标当二值类别，造成错误归因。单纯冻结V/输出参数也不能保证几何动作不变。

### 使用的解决方法和对应效果

- Q/K packed参数只训练前2/3行，V和其余参数逐值/哈希保护；所有候选独立保存，不替换best/final。
- 正式server加载完整候选后复用原open_loop_test反馈入口、完整600步预算和原成功标准。相同首请求的服务/本地推理差为0；源FP32参数逐一保留。混合数据环境来源仍明确未验证，不伪造兼容性通过。
- 用全任务统一alpha=.25/.5/.75做零更新插值，参照WiSE-FT的参数插值方法，不按阶段或目标切换权重。该通用迁移方法不意味着机器人效果保证。
- 原损失与执行前缀权重4做相同初始权重/样本顺序/RNG/QK范围的256+256配对对照，权重只进入FP32损失，不改变forward的二值动作mask、噪声或插值输入。此权重不是MINT官方策略，正式默认未变。
- 参照openpi pi0.py的时间约定，仅做反向换元后的Beta(1,1.5)时间先验对照，保留本地U(-1,1)、原clamp、原损失及推理积分。实际GPU噪声序列与控制组哈希相同，未改变正式时间分布。
- 四目标独立物理重建相同当前RGB及整段状态历史，只改变真实目标展示历史；首动作标签取自正式run_task4_expert实际发出的第一请求，目标只用于分析标签，不进入策略请求。

QK512新增训练8.8447秒，抓物错误开爪10/12→0/12，但Task1 XYZ4.89→5.82mm、Task4接近5.96→7.52mm；未采用。实际候选600步为Task4 0/1，错误选择2而目标为3，没有新成功率达标证据。已人工查看该次实际观测第0/30/100步图像；这是时序截图，不是完整视频观看。

三个统一插值比例均未保护完整阶段，未采用。前缀配对原损失256步5.8232秒、加权256步4.1376秒，Task4接近XYZ6.79→7.15mm、抓物错误开爪均3/12，未采用。时间先验256步4.6406秒使抓物错误开爪降为1/12、最终释放0/24误闭，但Task3下探与降低阶段、Task1和Task4接近仍退化，未采用。

四目标首请求共每组12次查询：source400错误上下方向4/12，QK512、prefix4、noise_time均5/12。它是对首动作方向的受控验证，不是完整选屉率。证据controlled_initial_20261029T030217Z/result.json。累计至此3292次更新、615.23782352秒optimizer训练，原累计900秒上限未重置；VLM始终只有一个实例，无长训练、无部署。上述候选全部拒绝，不能把局部闭爪改善当作80%/90%验收通过。

## 83. 开环任务进展对连续夹爪开度的诊断口径修正

### 遇到的问题

TaskProgressObserver按.5阈值把Task4的正常.25预抓取开度算作闭合尝试，开度波动又可能反复增加空夹计数；进展列表只看物体抓取/抬起/放置，漏记把手接触。诊断因此夸大空夹、漏记已经发生的进展。这不是物理成功判定漏洞，也不是修改后即可提高模型成功率。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
```

### 修改前

同一个实际失败回合被报告58次闭合尝试、57次未观测到双指接触后重新打开，进展列表为空。

### 使用的解决方法和对应效果

Task4连续开度单独使用接近闭合端点的.1/.2诊断迟滞，.25不计作闭合；迟滞区内抖动不新增尝试。增加独立观测到的把手双指接触、抽屉接触里程碑；Task1/3二值行为保持。该观察器仍不控制动作、终止或严格成功。

在同一驻留进程重新加载实际正式模块，验证.25预抓取不计闭合、.1/.2迟滞避免重复、Task1/3原二值转换计数，以及把手/抽屉接触里程碑。逐行复算原600步实际JSONL后为33次闭合、31次未观测到双指接触后的重新打开，并记录把手与抽屉接触；仍存在真实重复动作和约55.7秒无新里程碑。原始JSONL和失败结果保留，证据progress_aperture_semantics_20261029.json。该验证不是新的物理回合，Task4严格失败不变。

## 84. 针对真实四目标方向混淆补配对专家监督，按起点隔离适配与验证

### 遇到的问题

相同当前RGB及完整状态历史的四目标首请求对照仍出现上下方向错误；旧少量轨迹的起始扰动又与目标选择共同变化。已有证据支持增加直接可核对的目标历史到动作监督，不能继续只改变学习率、时间分布或闭爪权重。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/paired_target_collection.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/paired_target_replay.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/paired_target_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
```

### 修改前

原诊断监督不保证同一起点可对应四种真实展示历史。直接按seed重建新配对数据又会丢掉显式target override，重放成seed默认目标；正常旧数据不能自动补目标或改种子来冒充匹配。

### 使用的解决方法和对应效果

- 独立诊断数据集采用两个nuisance seed，各初始化四个真实目标。79385488组用于后续有限适配，79385489组排除该适配；目标只在初始化和质量分析中使用，不进入策略请求。
- 原正式collect_episode、严格质量和EpisodeDatasetWriter执行八条完整专家，不复制图像、不伪造失败恢复、不覆盖正式三任务数据。相同起点四目标只改变展示历史及相应目标物理初始化，保留真实演示后的状态演化。
- 新数据显式记录paired_target_initialization版本1。原open_loop_test重放仅在有该声明时按记录的真实target初始化；普通旧数据缺字段时仍按seed构造。未知版本、错误任务、未声明seed和非法target拒绝。
- 后续新监督对照已准备：同一驻留VLM复用特征，原归一化、MINT flow损失、Beta(2,2)、QK范围与512更新/新增最多30秒；保护Task1/3全阶段，Task4四目标全阶段及定位路径，按nuisance seed隔离。不改变正式默认与原best/final。

paired_target_collection_20261029T093920Z已完成8次尝试、8条保存、0拒收，正式raw检查8条/2808帧通过。五个非法初始化反例拒绝、缺声明旧路径保持。保存动作正式448px完整重放正在串行进行：相关问题条目287个执行动作、331个状态全部检查，XYZ最大差约0.00002397mm；原严格成功、全程质量、独立抓取/抬起/无掉落/放置及连续稳定1秒全部通过。其余七条尚在运行，不能记8/8重放通过。

后续参数更新仍须等待八条重放全部通过；截至本记录没有执行新数据适配，没有新的模型成功率，也没有采用或部署新候选。所有重任务在session31803串行，驻留完整模型只有一份，原累计900秒optimizer预算不重置。README保持不动。

## 85. 配对专家完整重放收尾、目标可读性与失败原型撤回

### 遇到的问题

上一节保留的是当时第1/8条完成的进行中记录，后续完整结果和若干失败实验尚未追加。旧session31803退出后，新worker的恢复脚本又保留已拒绝FiLM的构造字段和过期累计计时，容易造成接续状态误读。

### 修改文件

```text
/home/user/mujoco+evo/AGENT.md
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/current_repair_status.json
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/resume_condition_worker.py
```

### 修改前

状态文件仍指向31803及完整VLM驻留；文档只记录一条重放通过。恢复脚本把664.660秒作为固定旧计数，不能反映后续实验。失败FiLM原型的构造参数还留在诊断脚本中。

### 使用的解决方法和对应效果

- 接续实查唯一PID1677/session54699，只驻留原动作头，完整VLM实例0，RAM available约8474MiB、swap使用0。恢复脚本删除废弃FiLM字段，严格加载源动作头，并读取现有累计计时而非回退到固定旧数值。
- 确认配对专家8次尝试、8保存、0拒收，正式raw检查及8条完整448px保存动作重放全部通过；formal_replay.json逐条同时记录正式严格质量与独立抓起、抬起、无掉落、支撑放置及连续稳定1秒。
- 补核后续QK、完整头学习率、FiLM、端点训练和配对课程均未通过。FiLM已从生产撤回，原flow源码及FiLM前Evo1.py恢复一致，保留更早的FP32源值加载修复。正式默认采样、损失、积分求解器未改。
- 恢复后真实PNG源400动作12次查询与此前完整VLM输出最大差0。另逐项核对原best的516项视觉参数与缓存来源完全相同，源归一化统计也相同，允许复用缓存而非再加载完整VLM。

history_decodability报告显示：4个训练历史上的线性诊断在另一起点四目标为4/4；去除历史时四个输入特征逐值相同，只能1/4。这证明有限场景中目标信息保留，不是策略选屉率。直接L1小读出虽然held首动作方向符号错误0/4，但上层仅输出0.458/3.142mm而专家12mm，并损害pull/lift及夹爪时机，拒绝采用。

截至上述实验累计7644次更新、693.041166667秒，原900秒预算未重置，未交付高正确率模型。diagnostic固定旧统计时发现纠正旋转标签约0.029824rad被裁剪；这是固定旧统计对照的局限，未证明正式本数据统计有同样问题。后续监督直接读取raw专家动作再用原统计归一化、不裁剪目标，不替换旧checkpoint统计。原数据、best/final、视频、README及历史报告均保留。

## 86. 零初始化残差保留基线与排除起点验证：拒绝训练拟合冒充实际提升

### 遇到的问题

直接替换小回归头损伤原运动能力，因此检验冻结原best、只学习动作修正量是否更稳妥。此前诊断动作头来源为400步失败候选，不能据其输出宣称保护了实际best技能；只看方向符号也会漏掉位移不足。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/residual_action_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/fit_residual_action_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/contrast_residual_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/action_conditioned_residual_probe.py
```

### 修改前

没有初始化逐值保持原best的独立残差对照。平均池化读出不能等同OpenVLA-OFT的action-token隐藏表征读出；前者失败不能用来否定后者。

### 使用的解决方法和对应效果

- 原best作为冻结基线，缓存每个真实窗口及不同噪声的完整14步原预测；只使用公开动作维度mask，不以terminal padding泄露未来。相同噪声的批处理与B1差异实测并约束，真实PNG基线按B1保留。
- 小残差末层零初始化，实际验证初始修正量逐值为0。输入仅可见图文特征、状态历史、history mask和原模型自身动作；Task1目标为源输出蒸馏，Task3/4目标为未裁剪raw专家动作。目标编号、物体坐标、phase与专家计时不进入策略输入。
- 两个LR的完整阶段、三噪声及排除起点比较后，继续只变首动作损失权重14的单变量对照；另以四种真实目标历史的无监督rank3对比方向代替原高维平均视觉输入，检验起点扰动影响。均是独立诊断，不改正式默认或部署路径。
- 一个gpt-6.1-sol/medium代理仅文本复核实现及官方MINT/OFT源码，没有另起Python、训练或物理工作流。按复核建议准备补入原动作头完整条件注意力后的最终x_pooled，以明确平均池化/动作压缩的信息瓶颈；该补充对照在本记录时仍进行中。

原残差两组各1024更新，新增训练合计4.312010628秒。Task1最大平移输出漂移0.362/0.321mm，但Task4把手闭合XYZ从原5.613mm退化至9.346/9.496mm，抓物仍10/12错误开爪；训练起点几乎精确拟合而排除起点6/12上下方向错误，均拒绝。首动作权重14再1024更新、2.133262309秒仍6/12反向，拒绝；rank3对照1024更新、2.101580990秒为4/12反向，仍拒绝，不能按局部数字下降放行。

对763个受检完整输入做哈希，未发现FP32缓存图文特征和BF16状态/history完全相同而夹爪标签相反的冲突；这个有限检查不证明不存在近似混淆。至此累计11740更新、701.588020594秒，原900秒上限保留。没有新闭环成功率，没有候选采用；所有原安全与成功条件保持，README不变。

## 87. 复用native动作注意力定位历史信息丢失与动作阶段误差

### 遇到的问题

只用平均视觉向量的残差没有解决目标方向。补入原完整注意力后的隐藏表征也未改善，因此不能继续以相似的小残差排列学习率。另一方面，原flow旧权重未证明其短期历史能可靠决定选屉方向，需要分清视觉信息与动作侧表征。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/action_conditioned_residual_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/native_deterministic_alignment.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/additional_paired_initials.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/verify_loaded_feature_parity.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/native_memory_alignment.py
```

### 修改前

平均视觉读出与官方OFT的action-token条件不同，直接替换后损害运动。原完整动作注意力在相同真实目标历史上保留了多少可用信息尚未单独核对；受控配对只有一组训练nuisance起点。

### 使用的解决方法和对应效果

- 只读捕获原最后积分步的x_pooled896，按每个噪声与原动作对齐。真实PNG条件B4与原B1比较出现最大0.0104543归一化差并被断言阻止训练；恢复已完成非raw缓存，真实PNG按B1重新捕获，不放宽断言糊过去。保存中断记录及逐section进度，避免重算全部缓存。
- 该补充残差1024更新、2.048590779秒后仍held6/12反向，Task1最大输出漂移1.274mm，拒绝并停止残差路线。没有并入正式模型。
- 独立native确定性回归原型保留原MINT完整action encoder、8层cross-attention、state encoder及MLP，仅以固定零动作查询直接监督真实动作。原Beta时间clamp仍保留，因此实际使用t=.02、时间表index20；不是给旧flow模型偷偷改求解步数。正式flow/训练默认未改。
- 为核对起点覆盖，新增三个当前正式Task4/纠正/配对记录中未出现的nuisance seed：194951001/2用于训练，194951003排除训练。每个生成四个真实展示历史及原专家首请求，验证相同当前RGB/状态、历史不同。仅是首动作诊断条件，不冒称完整专家轨迹通过，不伪造后续14步标签。

native_deterministic_20261029T163321Z完成1024更新、51.157972381秒。原Task4抓物闭合4个窗口误开0/4、XYZ1.278mm，paired4窗口0/4、XYZ0.775mm；但两组object_descend各4/8提前闭爪，原Task1部分阶段和新起点两个上层目标仍失败，因此拒绝采用。原/paired把手闭合XYZ3.517/5.143mm，不能据整体XYZ下降宣称接触可靠。

四训练/四held真实PNG历史的native池化特征线性诊断只对1/4，而此前raw VLM平均特征为4/4；支持检查动作侧目标信息保持，不足以证明唯一根因。首次诊断solve右端Float/左端Double类型不一致，修正后实际运行结果保留，不把第一次异常写通过。累计13788更新、754.794583754秒，剩余145.205416246秒，不重置900秒预算。

新增12个真实上下文已生成、同组四目标当前图像和状态一致性全部通过；当前正在原worker内加载唯一完整VLM，旧动作头CUDA实例已释放。后续新旧特征功能一致性通过才允许control/训练期目标历史辅助监督对照，两组合计最多90秒，均使用相同可见输入与动作标签，目标编号只作训练CE标签、不作推理输入。此对照尚未运行完，没有新的模型成功率或部署，README和正式默认不变。

## 88. 移动历史读出失真、真实反馈种子更正与压缩前线索核验

### 遇到的问题

上一轮新组合策略仍未完成Task4。落盘的temporal_token_readout_20261029T174504Z/actual_feedback为真实seed83、目标3、600步、未选中把手、严格失败；诊断脚本原计划seed194951005被缓存对照循环的变量覆盖，不能把该结果冒称预定排除seed的验收。该回合第1–8步的学习读出为3，第9步变为2，后续仍有类别漂移；不少错误动作实际完整执行，不能统一归因于碰撞保护。

### 修改文件

```text
/home/user/mujoco+evo/AGENT.md
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/current_repair_status.json
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/moving_memory_continuation.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/moving_memory_runtime.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/one_frame_memory_probe.py
```

### 修改前

只在少量固定初始历史上读出正确，尚不能覆盖移动、历史mask变化和展示线索向窗口较旧位置移动后的读出。接续摘要又仍称上轮反馈进行中；实际文件已记录该回合结束。AGENT原验收层次尚未把用户90%候选、80%最终目标转成明确样本与四目标门槛。

### 使用的解决方法和对应效果

- 读回真实seed、完整动作和结束报告，不覆盖原失败，不给未执行的194951005补造结果。前轮seed循环已改为独立parity_seed/parity_target；新增诊断反馈实现另加请求seed与逐请求flow_seed断言，只有门槛通过才会运行。
- 复用已完成的paired_moving_histories_20261029T180813Z：4起点×4目标×41请求=656真实特征，同组当前RGB及整段本体状态历史一致；训练492、排除起点164。失败动作只作真实观测上下文，不成为专家动作标签。
- 固定ridge及配对nuisance投影均是独立诊断读出，不并入正式训练/服务；门槛未通过就不加载完整模型去跑这个失败候选。
- 继续用原冻结VLM的448px单帧路径核验压缩前的历史图像，监督四种展示/无可见展示五分类；推理规则仅使用实际K6有效帧，目标和可见slot只作训练/审计标签。参考Octo保留历史观测token及独立readout的接口思路，明确这是本地隔离诊断，不宣称MINT/MEM完整复现。
- AGENT相关问题条目新增可核对门槛：候选至少20条Task3和20条Task4、各18/20严格成功，Task4每目标至少5条且至少4/5；Task1至少20条同seed配对不得低于基线；最终独立验证各80%，记录样本/重叠/置信区间。不得把短训练90%推成未来总体80%保证。

已完成两个零optimizer更新对照：原固定ridge训练492/492，排除起点70/164；配对nuisance投影为64/164，均拒绝采用。没有因此新增模型闭环、没有覆盖best/final。真实反馈seed83的600步失败是前轮已结束结果，本轮只核对并追加说明。

压缩前单帧线索核验目前进行中，尚无通过结论。唯一重任务为exec session2235/PID29973，无第二份VLM或并行Torch工作流。参数训练累计仍16860更新/815.722739949秒，原900秒上限不重置；本轮新增optimizer更新0。README、正式场景/控制/flow/安全和成功标准不动；整体模型仍未达到90%或80%门槛。

## 89. 证实历史线索压缩前可读，并拦住阶段动作退化的候选

### 遇到的问题

移动历史上的线性读出无法泛化，不能直接把它当成图像中不存在目标线索。另一方面，旧残差即使改善了初始方向，仍会在把手释放、物体下探和闭合时给出错误夹爪动作；只检查方向或全局平均误差会漏掉这些退化。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/one_frame_memory_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/one_frame_memory_feedback.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/prepare_visible_phase_conditions.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/fit_visible_phase_residual.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/local_visible_phase_residual.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/dense_transition_conditions.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/fit_residual_action_probe.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

只有压缩后特征的目标读出。原残差没有当前视觉阶段条件，部分把手释放/下探动作被错误闭爪；历史读出是否正确与动作阶段是否正确尚未分成独立放行条件。

### 使用的解决方法和对应效果

- 采用同一冻结VLM的原单帧路径，448px真实输入、四展示类别及无展示类别监督；从请求中实际有效K6帧读出类别，不延长窗口、缓存过期线索或送入真实目标编号。只保留当前帧的对照输入完全相同，不伪造本体状态。
- 用原正式开环读取、时间索引及视频解码接口，为原/纠正/配对专家的30回合635窗口计算真实图片条件。没有新增替代正式开环入口，也没有把失败动作变成专家目标。
- 独立有限残差增加当前视觉和可见读出；另以训练观测拟合32维视觉投影、当前本体及因果1秒运动做容量/输入对照。Task1原完整策略按实际推理路径精确保留；不是只用蒸馏loss保证不退化。
- 放行代码同时要求历史读出与关键动作阶段通过。旧残差已知存在退化时，不因分类164/164就自动启动真实反馈或部署。正式MINT采样、flow损失/求解器、step_best规则及README保持不变。

单帧读出完成464张去重真实图片，训练351/351、排除起点113/113；完整真实历史为492/492训练、164/164排除起点，只用相同当前帧强制选目标为123/492与41/164。分类数据无跨split原图SHA交叉和矛盾标签。这是有限受控线索可读性，不能称为物理选屉率或总体泛化保证。

完整阶段条件已完成30回合635窗口、3016次新增单帧编码。有限残差4096更新/10.112627675秒，关键转换训练误差多为亚毫米，但排除回合仍有29条夹爪转换错误与11组平移超门槛，拒绝。128宽局部条件对照实际9668更新/20.001766316秒，仍有27条夹爪转换错误与10组平移超门槛，拒绝。两组均未进入正式闭环、未替换原权重；不能把接近方向正确或训练拟合换算成成功率。

累计为30624次更新、845.837133940秒，原900秒上限剩余54.162866060秒。当前在同一worker串行读取25条训练回合中转换前后各4个真实邻近观测，潜在457个新窗口；整条验证回合排除、prefix失败动作排除。此准备仍进行中、更新0，不写成已完成或通过。原best/final、正式数据、原视频、安全标准和README保留，模型尚未通过用户成功率门槛。

## 90. 对失败反馈补全持物与转姿态监督，按完整轨迹复核候选

### 遇到的问题

上轮隔离确定性读出在真实反馈中仍失败。Task3 seed194951005第51步形成双指接触，第56步物体比初始高40.31mm，第57步模型请求开度约0.73，物体落回桌面；177步危险接触终止。Task4同seed目标3，已选对并拉开，后续转姿态及取物未完成，600步用尽。不能再将这次Task3失败说成下探不足，也不能用分类正确替代Task4完成。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_conditions.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_continuation.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/partial_lift_retention_validation.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

先前诊断覆盖所有阶段，但主要在阶段中选少量点及转换邻近帧，尚不足以检查夹住后整个运输区间与转姿态连续过程。原型仍未通过动作精度门槛，已保留失败记录、未部署。正式完整窗口训练加载器没有因此被证明缺帧。

### 使用的解决方法和对应效果

- 通过既有open_loop_test的读取、视频解码和历史索引编码38条训练回合、9条排除适配回合的全部执行帧；真实失败前缀只作历史，不作专家标签。448/K6/1秒、源统计、原专家动作和正式质量要求保持。
- 增加更新前/后全帧诊断，逐阶段记录XYZ、实际限幅旋转和错误开闭；不再把阶段端点误差作为持物/转姿态全过程通过。验证回合用过的结果不能再冒称独立最终验收。
- 只排入一组累计900秒原预算内、最多新增18秒的小读出对照。公开任务条件仅加旋转动作mask的一位，来源于正常请求，非目标/阶段/几何真值；原型的Task1路径仍精确保留。正式MINT训练/flow/采样默认不修改。
- 再次查阅MINT官方README/train.py及论文2511.04555，和RoboFlamingo官方数值/夹爪分支及论文2311.01378。后者仅支持本地隔离读出对照，不称为MINT或MEM正式复现，不直接替换生产策略。
- 另准备不足50mm真实持物掉落的隔离候选物理复核，要求固定动作重放状态一致并保持50mm成功门槛；候选尚未应用生产，不能用它代替修正模型主动开爪。

本轮接续实际核实唯一worker为session2235/PID29973；没有第二份模型或并行重Python。两个新诊断脚本编译退出0。编码仍在进行，尚未获得本次全帧对照和物理候选结论，不写通过。Task3/4原型反馈均0/1，不能换算为正式权重成功率。原best/final、原视频、正式数据、README与正式终端输出保留。参数更新累计仍42592次/880.840795099秒；新组尚未执行，原900秒上限不重置。

## 91. 全帧监督与真实纠正更新后的实际结果，以及正确选屉后接触丢失的覆盖缺口

### 遇到的问题

只看少量专家阶段点不足以验证持物与转姿态。后续完整监督虽使Task3同seed真正成功，Task4仍会在正确目标附近空闭爪、抬高手部并开始拉动，失去接触。首次真实接近纠正也未覆盖下一版策略的新状态；不能把开环平均误差下降写成反馈任务成功。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_conditions.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_continuation.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_feedback.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/direct_visible_decoder.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/current_handle_correction.py
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

静态读出只有少量阶段点与邻近帧。正式纠正入口在任何first_selected_drawer非空时都停止筛选/拒绝接管，无法生成“选对但未夹稳、抽屉未打开”的纠正监督。旧四目标配对初始化协议也要求四目标全集，不能诚实表达单目标的实际target override。

### 使用的解决方法和对应效果

- 完整编码38条训练回合8573帧、9条整回合排除验证2562帧；保留原数据和实际专家标签。全部真实监督包含持物、连续转姿态和释放，失败前缀只作上下文。
- 在原900秒累计预算内完成7579次小读出更新18.002214119秒。随后用户对具体新增单组最多60秒/10000更新明确回复“完全可以，这并没有任何影响”，累计上限改为960秒，原计时不重置。该授权组实际10000更新22.224774768秒；当前累计60171更新/921.067783986秒。
- 同一冻结VLM、同一共享Task3/4静态读出，448/K6/1秒/原统计，Task1源路径不改。没有把新读出并入生产MINT默认，没有覆盖原best/final。另一次固定alpha=.5的零更新参数中点旋转4组失败，拒绝，不继续网格搜索。
- 从第一版实际失败第33步接管，首条纠正398帧，77帧展示/失败上下文排除动作监督，321个正确动作窗口；严格专家、正式raw及448完整保存动作重放通过。当前模型对该新纠正首个recover位置误差12.69mm，有限更新后0.40mm；只能称该观测拟合改善。
- 为真实“正确选中后失去接触”的下一次失败增加独立handle_contact_loss类型。要求实际双指接触曾发生、原首次选择正确、当前闭爪且已失去双指接触、整个已观测前缀抽屉离闭合位置不超过5mm、目标历史仍有效、无安全/不可逆失败。原错误选择、危险接触和掉落不清除；旧approach类型原5mm/2度门槛不变。正式raw与物理重放都复核新类型。
- 初始化协议保留v1四目标要求；新增v2可表达真实非空唯一目标子集，不谎称未采集的目标对。未知版本、bool/越界/重复目标、未声明seed/目标拒绝；初始化信息不进入策略输入。
- 原完整上下文重放现在在接管时刻立即拒绝不实证明，避免明知证明失败仍把后续专家全跑完。该规则只用于保存动作审计，不改变真实策略默认运行预算或普通质量失败的继续执行。

原完整监督候选：2562验证帧错误开闭27→3，XYZ均值3.867→0.796mm，38组旋转均通过，但仍8项动作门槛失败。真实同seed194951005反馈：Task3 107步一次抓取严格成功；Task4 600步无有效把手抓取/打开，失败。候选未采用。

用户授权的10000次纠正更新后：验证XYZ均值约0.71mm，错误开闭仍3，38组旋转通过、9项精度门槛仍失败。实际同seed回归Task3 106步一次抓取严格成功，独立contained/released/supported/drop分歧均0；Task4在目标历史有效时确认选择正确目标3，但仅两个控制时刻出现双指把手接触，未夹稳、未拉开，600步失败，后续还出现质量失败。仍拒绝联合候选。该seed已用于纠正适配，不能冒称独立泛化，更不能报Task3总体100%或Task4达标。

输入邻域审计显示第32–33步实际失败状态与已有纠正状态手端相差约15mm；正常成功拉动的指宽约5.7mm，本次接触后约0.2–0.9mm。轻量只读审查支持先检查新纠正覆盖/首步误差，不足以支持加历史或换网络。参考MINT官方两阶段策略、RoboFlamingo明确动作/夹爪分支，以及DAgger原论文https://proceedings.mlr.press/v15/ross11a.html；不将理论保证套用于本轮有限实验。

初始化协议12个正负例、纠正证明13个结构正负例通过；Task1原20回合保护与两任务90%/80%实际验收均未完成。新的接触丢失纠正首次重放发现“接管时位移误记为前缀最大位移”，被独立证明拒绝；原失败数据已quarantined，训练入口拒绝，没有进入特征融合或参数更新。已修正累计绝对闭合位置偏移并重新采集correct_handle_loss_correction_v2，严格专家/raw通过，448完整保存动作重放尚在运行。未把此项提前写通过，未自行启动下一组更新。

## 92. 同状态原生RGB与视频编码敏感性、有限纠正续轮及保存读出恢复

### 遇到的问题

正确选屉后失去把手的纠正v2最终通过正式448px完整动作重放，但同静态头追加5000更新后，Task4仍未有效打开。不能把纠正专家成功或开环均值下降当成策略完成。进一步受控检查发现，只改变同一状态图片的原生RGB/MP4来源，也会改变当前诊断读出的动作和连续夹爪开度。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/codec_pair_probe.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/native_rgb_augmentation.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_conditions.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/current_handle_correction.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/resume_native_feedback_20261031.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

隔离读出主要用视频解码图像拟合，正式反馈输入为原生RGB；先前同PNG预处理一致性检查没有排除这个域差异。原型仍未通过整体动作门槛，旧正式MINT默认未被替换。

### 使用的解决方法和对应效果

- 同状态、同本体K6/掩码/指令的三个窗口中，原生重放与实际原始图片逐像素最大差0，同输入重复输出最大差0；原生RGB换成原MP4解码后，首步XYZ差0.773/1.664/1.098mm，连续夹爪差0.106/0.0557/0.1342，预测目标类别未翻转。只证明该读出有行为相关域敏感性，不宣称是全部正式失败原因。
- 保留原始paired_input_checks.json，并新增paired_input_checks_corrected_roles.json：前两个窗口为失败policy_context，只比较输入域，不当专家动作标签；第三个recover窗口才有有效专家标签。没有把错误标签用于训练。
- 串行恢复两条真实纠正全轨迹及四目标配对专家的训练/验证轨迹，共10条。逐个动作前核对状态与时间；846个原生训练窗口、204个整seed排除验证窗口与原MP4样本并存，合并10061/2766窗口，源标签、原统计和原正式数据不改。失败前缀仅作历史。
- 用户已授权同类小验证自主执行、固定60秒基准时间增幅不足50%不逐次询问。iteration1为5000更新11.786441487秒，iteration2为5000更新11.106652747秒；连最初批准10000更新22.224774768秒，本次新增共45.117869002秒。累计70171更新943.960878220秒，计数不重置，未启动长训练。
- 2026-10-31核实旧worker已不存在。新恢复脚本只还原已保存读出，核对39个跨任务/图像域输出最大差2.98e-8后，通过原生产loader加载一份源VLM，继续原open_loop_test实际反馈；不重做特征、不重训，不修改正式terminal输出或README。
- 再次联网核对MINT官方README的两阶段训练与官方loader的Resize/随机裁剪/旋转/颜色增强。此次原生RGB配对是基于已测差异的本地隔离对照，不能称为MINT官方codec方案，也未盲搬旋转/裁剪去改变动作坐标语义。

iteration1同适配seed的Task3严格成功，Task4仍600步失败。iteration2的2766验证帧XYZ均值0.7600→0.7295mm，但错误开闭仍4项、关键动作门槛11项失败、旋转2组失败；联合原型未通过、不部署。当前正在恢复源模型并执行Task3/Task4实际反馈，结果尚未收取。原best/final、源统计、原视频、正式数据和严格安全/成功规则保持，Task1配对及90%/80%独立成功率验收尚未完成。

相关问题条目结果补充：iteration2真实反馈Task3 106步一次抓取严格成功；Task4首个尝试在105步CUDA unknown中断，保留记录不作完整结论。原进程退出后新CUDA小运算通过，仅补跑Task4，完整600步失败、17次无双指后的重新开爪。源模型/统计/读出SHA相同，没有重做更新或Task3，也没有重置WSL或驱动。五帧native缓存到实际LocalPredictor数值对照通过：条件、视觉、本体和XYZ差0，夹爪仅约1.03e-8浮点差。

## 93. 已监督反向动作的有限拟合控制、真实反馈结果与缺失正常初段监督

### 遇到的问题

native_handle_loss的79/80帧已是真实专家监督，专家dx为-3.995/-1.480mm，iteration2却预测+2.815/+4.401mm；第82帧要求重新前进12mm，预测仅4.668mm。完整路径对照差0后，这些是实际拟合错误，不能继续归咎于MP4编码或接口。邻域有正反标签的相近状态，但输入有可观测差异，未发现完全相同输入的矛盾证明。

此外，两条seed194951005纠正的前77帧都是展示和失败历史，不作专家动作监督；原正常Task4数据seed为623777821..623777922，配对训练seed79385488，均没有这个实际失败初态的正常专家接近。实际早期输入与所查最近训练状态手端仍差11–20mm，恢复帧拟合改善不能证明正常接近已经学会。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/native_cached_pipeline_parity_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/native_failure_neighborhood_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/direct_visible_decoder.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/training_residual_priority_probe_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/recover_priority_accounting_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/continue_priority_optimizer_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/priority_counterexample_feedback_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/normal_start_supervision_20261031.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/full_trajectory_conditions.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

当前诊断小读出均匀抽取全部有效窗口，对少数已经标注的反向恢复仍拟合错误。诊断训练循环在累计时间检查通过后才保存结果，一次更新若因不可预测延迟超过时间，会使已完成更新和计时尚未落盘；这次实际触发了该问题。正式训练保存路径没有因此更改。

### 使用的解决方法和对应效果

- 仅做一次隔离对照：根据初始训练首动作残差设置有限采样优先级[1,20]，保留同一共享静态读出、损失、实际监督、448/K6/原统计和Task1源路径；不使用使用的解决方法和对应效果或阶段作为优先级。连续.25标签的BCE存在非零本征熵，先扣除该熵下限才衡量夹爪误差，避免把正确.25本身当困难样本。正式MINT默认uniform不改，也不称为MINT官方策略。
- 计划12秒的首更新实际耗时17.307858441秒，触发保存前断言；保留异常，从原异常帧恢复同一已完成模型/计时，未重新训练该更新。诊断循环改成先保存实际训练/模型/累计计时，再报告预算超出，防止丢计数。首更新耗时原因尚未确定，不擅自归因于CUDA冷启动。
- 在用户“同类验证增幅不足50%无需再问”的固定60秒参考下，设本轮自主上限85秒，不滚动放大。随后原优化器、scheduler、采样生成器精确接续4999更新11.116416499秒。累计75171更新/972.385153160秒，本次新增73.542143942秒，比60秒增加22.57%，无长训练、无原权重覆盖。
- 方向拟合改善后，继续用原open_loop_test真实反馈反例验证，原开环失败保留，安全/严格质量不放宽。失败后转为补该实际初态的正常完整专家；直接保留collect_episode同一动作前sample的原生RGB、状态和时间，不先压缩再恢复，不把截断轨迹伪装成功。完整严格专家、raw、448保存动作重放均通过后才编码前半段与阶段代表窗口。

两帧反向预测已改正确，XYZ误差7.429/6.412→1.442/0.994mm；第82帧重新接近误差7.843→2.549mm。但2766验证帧平移均值0.729→0.846mm，错误开闭4→3、旋转仍2组失败；候选未通过。真实反馈Task3再次严格成功，Task4仍跑满600步失败、15次无双指后的重新开爪，未采用，不将同seed重复诊断累计成独立成功率。

另确认该未采用视觉读出在后段无目标展示的K6窗口中会出现弱假目标提示：当前实际输入图中抽屉均关闭、无目标展示，约150–200步仍输出类别1，最高相对无提示分数仅0.05279；最早接近时目标3分数约1.075，且早期偏差先于该假提示。该新风险保留为未解决的诊断读出验收项，不以它解释全部早期失败，也未部署这个读出或新增隐藏时间/目标门控。

相同初态的正常专家采集/严格重放和原生特征准备正在串行进行，尚未收取结果、未新增更新。Task1 20回合配对与Task3/4独立90%/80%门槛仍未完成，整体任务未完成。

## 94. 正常接近监督的实际结果与安全策略状态接续

### 遇到的问题

相关条目的正常专家准备已完成，114新窗口使训练条件10175、验证2766。正确专家观测上应抬升6.703mm却只预测2.638mm，预抓取停止帧XYZ误差6.019mm；闭环尚未稳定抓住把手。仅有失败之后的纠正，不能替代真实正常接近和安全已访问状态的监督。

### 修改文件

```text
/home/user/mujoco+evo/mujoco_pickplace/tasks.py
/home/user/mujoco+evo/mujoco_pickplace/episode_dataset.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/direct_visible_feedback.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/two_action_feedback_20261031.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

接近纠正要求真实5mm/2度偏离，正确地拒绝尚未达到失败资格的prefix30/34；但安全策略状态的专家接续与失败纠正没有独立语义。不能把未失败状态伪造为失败来收数据。重复首动作是否加重阶段停滞也没有执行对照。

### 使用的解决方法和对应效果

- 已进行一次3000更新/6.841695875秒同静态读出有限拟合，未改正式采样/结构/权重。累计78171更新/979.226849035008秒，原固定60秒参考下实际新增80.383839817秒，自主85秒上限未重置。
- 新增handle_policy_state中性来源，逐子步核对真实前后状态、从未选择/四把手双指/四屉开度等；记录policy_state_continuations，不增natural_corrections。旧失败证明与旧分类保持。loader将其recover历史标policy_state_continuation，前缀不作动作标签。
- 原生RGB保留同一个动作前sample，严格专家、raw、完整448px重放均通过后才编码。prefix24/30通过，增加118窗口，总10293/2766，尚未据此更新模型；prefix28因选择时历史过期被拒，未训练。
- 再次核对MINT官方LIBERO客户端horizon=14及依次执行预测动作的循环。仅隔离固定2动作零更新对照，不直接照搬14步、不改正式默认；日志保留原NN请求时间与动作索引，所有物理子步保护不变。自定义suite明确标真实反馈，不能沿用teacher-forced标签。

3000更新后受检抬升预测5.656mm、停止帧误差0.604mm，验证XYZ均值0.846→0.761mm，但仍3项错误开闭、3组旋转失败。真实反馈Task3在同适配seed106步一次抓取严格成功；Task4完整600步失败，始终未确认双指选屉/打开，候选未通过、不采用。局部拟合改善不等于正确率改善。

新来源17项结构、旧来源兼容与事件分类检查以及两条生产真实请求重放通过。正式formal_boundary_open_loop_448/report.json中5条完整专家重放逐条终点通过，模型查询0。两动作真实反馈仍运行中，不能写成功；Task1配对和独立90%/80%模型验收仍未完成。README、原正式数据/视频/best/final保留。

相关问题条目续轮结果：固定两动作方案完整600步失败，300次真实模型查询、16次无双指后的重新开爪，未选中/打开，拒绝采用。合格新接续中的approach、handle_close等真实监督误差仍有8–14mm，沿原预算仅再做800更新/1.974170515秒，累计78971更新/981.201019550009秒，本次相对固定起点新增82.358010332秒；没有扩大85秒自主上限。prefix24/30的approach均值8.12/11.84→2.53/3.90mm，但完整验证XYZ0.76148→0.81354mm、错误开闭3→4、旋转仍3组失败，不能放行。实际Task3同适配seed108步一次抓取严格成功；Task4已形成真实双指接触并打开约0.175m，仍在运行，不提前报告任务成功。第一次双指接触控制边界为9.3s，其请求采样历史始于4.2s、晚于最后展示4.1s，保留时序证据，不冒称K6目标选择验收已通过。

相关问题条目完整反馈及更正：Task4最终356步严格成功，物理完成、稳定放置1.058秒、无质量失败，contained/released/supported/drop逐控制边界分歧0。Task3配对为108步一次抓取严格成功。此前把step47的控制边界双指接触当成首次选择依据不准确：首次选择已在前一控制步的物理子步确认，实际触发NN查询9.1s，历史第一帧4.1s为展示，完整result.json记录target_available=true。旧中间判断保留为误判并由本段更正，没有修改窗口或验收规则。这是已适配seed194951005上的首次完整Task4正面结果，不是独立成功率。冻结候选后排队新seed194951007..010的Task4四目标和Task3小筛查，参数/阈值不变，尚无筛查结果，不部署到正式best/final。

## 95. 共享观察动作读出接入正式模型与双推理入口

### 遇到的问题

已取得适配起点Task3/Task4完整成功，以及新seed目标0/1的Task4成功，但实现仍依赖诊断回调，正式服务和独立开环不能直接加载同一读出。用户明确授权边测试边接入正式代码。新seed目标2/3随后均选对并打开，但进入取物阶段失败，实际Task4新seed为2/4；不能把接入正式入口包装为高成功率验收通过。

### 修改文件

```text
/home/user/mujoco+evo/Evo-1/Evo_1/model/action_head/observation_decoder.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/Evo1_server.py
/home/user/mujoco+evo/Evo-1/Evo_1/scripts/train.py
/home/user/mujoco+evo/mujoco_pickplace/open_loop_test.py
/home/user/mujoco+evo/mujoco_pickplace/outputs/repair_20261028/production_observation_decoder_20261031.py
/home/user/mujoco+evo/AGENT.md
```

### 修改前

小共享读出通过诊断回调替换动作头，正式Evo1仍执行原flow头。只改server loader还会漏掉独立LocalPredictor loader；直接把新runtime约定写成旧骨干训练约定会丢失真实来源。Task1与Task3 mask相同，不能只靠mask分任务。

### 使用的解决方法和对应效果

- 仅显式声明附件的候选启用共享观察读出；无此配置的旧checkpoint仍走原路径。Task3/4共用冻结MLP，输入为真实可见K6线索、masked fused视觉投影、FP32当前本体与前一有效1秒状态差、公开旋转mask；Task1返回原flow路径，不加额外视觉分类推理。
- FP64单帧线索读出、FP32视觉投影/动作MLP及真实连续夹爪分别保留；最终由原normalizer反归一化，公开padding/mask不变，不使用目标编号、对象坐标、接触真值或专家动作作为策略输入。
- 正式server与独立open_loop loader共用附件校验/attach；checkpoint复制原骨干参数与原统计，新增manifest锁住三个附件和source配置，来源环境与当前读出runtime分别校验。source构造/推理字段默认全部冻结，两个入口都要求原非legacy50步Task1 solver。
- 请求边界核对task_key、指令与mask，归一化前拒绝非有限原始状态；cache只保留当前有效K6图像的只读分数，每请求重建线索。新增后端源码进入开环代码身份。
- 候选明确为冻结推理bundle。训练入口、load helper、train(True)及forward防止把候选静默当原flow续训而丢掉已学读出；原MINT训练默认、旧checkpoint续训/采样逻辑不改。未因此启动训练。
- 源文件备份在production_observation_decoder_20261031/production_before_apply。独立候选路径为ckpt/evo1_mujoco_pickplace_observation_candidate_20261031/step_candidate；原best/final不替换，README不改。

原训练在收到停止信号时也写入step_final，可能替换完整训练的最终权重。现在按是否达到max_steps区分step_final和step_interrupted，后者包含优化器、调度器和采样游标。使用实际保存函数在临时目录验证中断24840步保留旧final并保存步数及游标。此前CUDA unknown error可能来自更早的异步操作，此项修复没有证明CUDA故障已消除。 [PyTorch CUDA执行说明](https://docs.pytorch.org/docs/stable/notes/cuda.html)

6个生产/导出Python文件AST通过。轻量sol/medium文本审查提出并已修正两loader solver差异、source字段漏冻结、Inf被归一化截断及后端源码身份遗漏；这不是运行验证。现有单VLM完成四目标筛查后将串行导出与逐值比较成功轨迹的正式LocalPredictor/server输出，独立新服务启动尚未运行，不先写通过。新seedTask4实际2/4，Task3筛查进行中，独立90%/80%及Task1保护门槛均未完成。

相关问题条目正式接入复核（2026-11-04）：冻结候选初筛完整结束，Task3为3/4、Task4为2/4。目标2正确开屉后卡在前沿，目标3选对时展示已过期，随后出现夹爪接触和桌面受阻。Task3未通过的回合发生手掌与筷子接触，后来稳定放置。20回合候选门槛未启动。

当前正式best 22036步通过原服务的环境校验，原客户端完整复测Task3、Task4各一回合。Task4目标3先选开2，600步严格失败。Task3首次闭爪即请求上升，300步45次未抬起尝试，严格失败。已抽查两段视频的关键帧并比对实际动作日志。

更正：用户指出该22036步文件来自意外中断的训练，不能作为完成训练的模型或其最终效果。日志显示该轮至少执行至24840步，随后出现CUDA unknown error，报错行是异步错误被检测到的位置，根因尚需查证。上述轨迹仅用于构造已验证的失败状态。

旧完整底座93bd7788不再位于正式best路径。独立候选从未完成文件恢复516份编码器存储，35份与final不同，重新打包使用final 80000步的完整flow动作头。旧编码器、统计、读出保持原来源，Task1的旧配对结论不继承。正式server加载并与原实验比较Task3六个、Task4十三个真实观测，19个首动作的最大差均为0；此项比较复现了策略输出，不是专家动作准确率。原客户端已知seed194951007的Task3 116步、Task4目标0 271步严格成功，387步动作和前后状态均逐值一致。独立90%/80%门槛仍未完成，本轮新增训练0。

后续诊断：27个Task3窗与42个Task4窗联合排练，更新400次、实际0.703797834秒。Task3对应已见失败状态112步严格成功，Task4同一已见场景正在评估；中断训练得到的状态只提供监督，不代表完成训练模型的表现。累计训练79371次、981.904817384秒。

## 96. 固定实际注意力后端后再比较缓存与实时动作

### 遇到的问题

原生动作头缓存验证出现最大动作差0.00146484375。对照时未固定实际注意力后端，不能把此差异记为模型动作缺陷。

### 修改文件

mujoco_pickplace/open_loop_test.py，以及隔离诊断目录中的后端一致性核验脚本。

### 修改前

身份记录只有权重、配置、统计和附件，未记录FlashAttention可选依赖是否真正启用。同权重和配置可因依赖加载情况走不同BF16计算路径。

### 使用的解决方法和对应效果

LocalPredictor身份补充实际视觉及语言注意力后端。缓存对照显式固定普通视觉注意力和语言eager；保留原1e-5一致性容差、全部物理判分及默认模型配置。

首条来源在FlashAttention路径下有效视觉特征差3.609375，在固定普通/eager路径后为0；同状态、掩码及噪声下缓存/实时动作恢复一致。七来源完整核验尚在运行，不提前记通过。原生动作头同初始化纠正A/B尚未启动，Task3/4独立门槛未达标。


**历史方案：真实动作监督原生对照（进行中）**

- 当前正式数据18条严格保存动作重放全部通过：Task3 6条（4训练、2整轨迹验证），Task4四抽屉各3条（各2训练、1整轨迹验证）。图像均为原始状态重放448px真渲染，原判分、物理动作、终点观察不变。对应`current_action_supervision_20261105/progress.json`。
- 动作训练初始化为同轮完整`ckpt/evo1_mujoco_pickplace_stage3/step_final`，80000步，权重SHA256 `d597c39acf5e7bae14280125e0adfbdafef286c122715619c4d60d89240c6209`。使用实际standard视觉/eager语言后端，K6、1s、composed，禁用外部读出。与旧混合恢复诊断身份严格区分。
- 训练前真实动作误差反向检查通过：最后时序qkv/proj的B参数梯度范数0.0039461/0.0064963，未执行优化器更新。不是目标分类梯度或策略成功率。A初始化时B为零，A首步梯度为零符合低秩残差链式法则。
- 已开始同初始化head_only/joint_temporal_and_head各256更新、串行microbatch1、有效batch6的有界对照。训练标签只使用严格专家后缀与当前正式数据；成功策略保持仅使用两条实际成功轨迹的冻结速度场，不将两条失败轨迹充当成功监督。留出按整轨迹隔离。
- 原生动作头已有的验证损失Task3约0.02192、Task4约0.02530，仅用于对照；禁止将其解释为80%任务成功率。训练结束自动进入完整Task3/Task4首轮闭环，后续仍需两组新种子和Task1回归。- 上述首次有界对照在head_only第93次更新后主动中断，计入205.608927秒实际优化耗时；保留原目录和第64步权重。原因：脚本head.train启用了原配置0.1 dropout，与本轮确定性速度场保持设计不同。此处是对照脚本设置修正，不据其验证退化给模型归因。新对照独立目录`action_grounded_native_ab_nodropout_20261105`，从同一完整80000源权重重新开始，head.eval但requires_grad=True，关闭dropout；预算/记录不重置，旧耗时仍累计。此设定只适用于本轮小样本适配，未修改正式完整训练dropout默认值。
- 专家目标原有范围核对：七条纠正中六条全部在原动作统计范围内；目标3的Task4轨迹14/4802个活动维度略超范围，归一化最大1.01737。适配保留原真实动作监督（推理flow不强制[-1,1]截断），没有放宽物理终点或将超范围归因于主要失败；正式loader的clamp语义与本次原值监督差异明确记录。
- 确定性原生动作对照完成：head_only与joint_temporal_and_head各256更新，实际优化耗时546.599585/623.969563秒；完整source80000同初始化，固定相同数据/采样/noise。单模型串行，无完整训练。
- 开发Task3新seed314184300配对严格成功：head_only115步，joint116步；同一个seed的两个不同模型结果不是独立2回合成功率。物理放置与环境终点一致，无质量失败事件。仍须多组新seed及Task1回归，不据这一组宣称80%。
- head_only Task4 seed314184400目标0，首先接触3，325步28次闭爪仍未选屉。为效率保守淘汰此候选，当前回合明确censored/not_determined_full_budget，不按完整600步失败计数，原2条已完成Task3结果保留。剩余Task4联合候选转独立目录`action_grounded_native_joint_task4_screen_20261105`继续完整闭环。修复任务未停止。
- 首轮闭环保留全部14步真实模型动作块（不改输出），用于区分条件目标学习错误、动作序列及执行问题。
- 已核对旧`paired_target_collection_20261029T093920Z`八条配对专家数据的collection_config与当前正式Task4完全相同；准备原物理重放、原状态时间核对和真实448px重渲染重新资格检查，尚未启动。四目标同nuisance的真实历史适合检查目标依赖动作监督；旧特征缓存不能直接代表当前编码器。旧来源/历史开发使用记录保留，不能充作最终独立验收。
**历史方案：Task4选择错误的原生动作条件监督（进行中）**

- 联合原生候选Task4 seed314184400在sim7.878s触发原质量规则wrong_first_drawer（目标0、首先选择3）。第112步终止该已失去严格资格的候选诊断；没有完成600步物理过程，不伪报完整预算测试，stopping_record保存原失败事件。两种候选的Task3各一条配对成功不能代表Task4达标。
- 转向现有真实四目标配对专家数据的当前物理重新资格检查：8条来源原始metadata/动作/状态不修改，collection_config已核对与当前完全一致；每条当前物理保存动作重放、独立严格终点、状态时间误差、真实448px图像均重新核实后才作监督。目前前5条通过（4训练、1验证），余3条继续。全部属历史开发数据，不计最终验收。
- 训练nuisance seed79385488/frame44找到可用四目标动作条件组：当前RGB逐字节相同、完整K6机器人状态最大差0、四条真实展示历史不同；专家第一步XYZ最大两两差24.2368mm，14步中最大24.4016mm。这是实际动作监督，不是目标分类正确率。新源数据派生窗口已标target_training_annotation_only，仅整理/分组，不进入策略输入。
- 已准备下一组有限动作学习对照（尚未启动）：同源80000完整编码器+已通过Task3的v3 head256初始化，先采实际Task1/Task3成功行为保持条件；原生paired flow与paired endpoint+centered action loss两组各256更新。两组同数据/初始化/随机数/更新数；修改组仅在配对训练的偶数更新t=0并加动作残差中心化项，其余原生Beta(2,2)保留。它是针对本项目的训练候选，不宣称论文原样算法或已经有效。
- 配对训练所有四历史使用相同噪声，t=0时动作编码器输入必须完全相同，迫使模型只能从真实历史区别动作；中心化均值用无梯度第一遍、逐条第二遍反向，避免多份VLA/多组大激活。部署仍为正常单样本原生K6+状态flow推理，不需要四目标联合输入、真实目标编号或训练时的中心化运算。
- 放行前置检查：保持教师/学生初始速度场误差须为数值零；t=0四条动作输入最大差0；配对动作误差须产生真实时序参数梯度；所有临时采样修改以finally恢复。动作头FP32，小样本训练关闭dropout。初步条件动作诊断与真正新seed闭环分别记录，不把动作对齐数当成功率。
- 配对专家当前重放8/8严格通过，train79385488与validation79385489均得到frame44四目标同当前RGB/全K6状态、不同真实历史的组；验证组第一步动作XYZ最大两两差24.3428mm。仅1个训练nuisance和1个开发验证nuisance的条件诊断，不可充当多组Task4成功率。
- 新原生v3保持条件已完成：Task1 seeds314194100/314195100均严格通过；Task3训练seed314184300通过115步，重复原始动作/执行动作/前后状态最大差全0（同seed重跑不算新增独立成功）；开发验证314185300失败，sim14.804s触发unsafe_execution_contact，失败样本不纳入成功保持。
- Task3该失败为抓起后异常接触，静态XML几何顺序候选为手掌collision geom72与object geom91，需当前环境运行时名字确认后再归因。已准备从首次闭爪前安全前缀重新专家抓取和运输的独立重放采集脚本（未执行），不将已失败动作当专家标签、不降低接触规则。
- 新配对动作训练已启动：教师/学生初始速度场最大损失0；纯噪声端点四目标动作编码器输入最大差0；清空梯度后的首轮配对动作梯度B范数0.012326/0.023107，A为0符合初始零B。单独general gradient preflight包含上一轮遗留梯度，只作记录，不用于机制证明；文件已标invalid-evidence，脚本已加zero_grad，实际优化器每轮一直先zero_grad，训练更新不受此审计记录问题影响。
- 新候选运行配置显式use_flash_attn=false，与实际standard视觉/eager语言一致；原source config/SHA与后端身份保留，防止以后导出回到不同后端。
**历史方案：低秩候选淘汰、原生共享注意力与多初始条件数据**

- paired_native_flow和paired_endpoint_contrast各完成256更新，完整优化区间1162.107948/1135.322288秒，均保存含优化器的resume.pt。纯噪声/中心化组训练动作对应9/12，但开发留出只有4/12；原生组训练7/12、留出4/12。它们是1个训练nuisance/1个开发验证nuisance、各四目标三噪声的动作诊断，不能当12个独立任务回合。两候选未通过动作前置筛查，已保守淘汰，未执行原计划额外闭环，也未写正式默认模型。conditional_screen_rejection.json保留，所有未运行不记完整任务失败。
- Task3新原生失败的安全前缀51步完整重放最大误差0，生产专家接续87动作，并再次保存动作独立重放严格通过、误差0、物理判定无分歧。原生运行时确认异常geom72属于hand、geom91为object_geom；纠正从首次闭爪57步之前介入，避免后来74步手掌碰物体。数据在task3_native_palm_contact_seed314185300_20261105，未把前缀当专家标签，后续模型仍须实测。
- 下一候选准备改为原生第20层共享空间/时序注意力（norm1、QKV、投影、ls1；约4M参数）和原动作头，所有低秩B保持0，不使用低秩残差或外部目标分类器部署。单候选工程修复包，不能把它与旧低秩实验的差异解释为单因素因果对照。初始化仍为完整80000原生骨干+已验证v3 head256，非70830混合恢复。
- 为提高数据覆盖，新采集最多2个完整训练nuisance组及1个开发验证组，每组四目标；旧配对8条仍保留，合计期望3训练nuisance/2开发验证nuisance。新source为fresh_paired_action_supervision_v3_20261105，原生产专家、真实RGB、正常EpisodeDatasetWriter；每条先原质量通过再独立保存动作重放、原时间状态核对和物理终点。当前seed314204000四目标全部通过，最大状态误差1.4041e-7，时间误差0；其余继续。
- 新采集脚本两次工具修正的未完成记录保留：v1图像字段需images[front]结构，在0合格轨迹时中断；v2独立重放误将展示分割渲染一并略过而报错。均不是模型/专家失败，不计合格数据。v3保留完整展示期分割和_target_visible_times一致性，之后只省略不用于观察/判分/训练的机器人执行重放RGB；训练图像始终为源专家的真实448px RGB。已实际通过首四条重放，绝不跳过可见性判分。
- 原生共享注意力候选训练脚本train_shared_native_memory_repair.py尚未执行；要求新增组配额和Task3纠正均严格通过后启动。真动作配对监督、50%噪声端点/中心化损失、Task1/3成功速度场保持，Task3专家采样50%来自新接触纠正，剩余来自旧独立正式/纠正池。目标与phase只分组/监督，不是策略输入。
- 已准备仅缓存固定前19层、且只对配对条件窗口生效的资源优化。其余RGB、学习的共享第20层、语言与动作路径均正常计算。必须先对真实fused输出和原生注意力梯度分别通过1e-5/1e-6一致性，再允许第1次优化器更新；未运行，不宣称加速或已验证。下层有可训练参数或非eval模式立即拒绝缓存。普通/正式推理不使用该临时缓存。

## 97. 固定训练随机种子与无参数入口读取旧权重

### 遇到的问题

训练初始化和数据worker曾固定随机种子；无参数开环可能读旧模型，无法正常比较新训练结果。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
mujoco_pickplace/open_loop_test.py
```

### 修改前

训练、worker和诊断噪声存在固定seed，默认权重与实际使用不一致。

### 使用的解决方法和对应效果

训练seed默认None，worker随本轮可恢复采样器seed；采样seed与游标仍随checkpoint保存。开环默认权重和用户显式权重区分，诊断噪声随机生成并全部记录。

普通新训练使用新随机初始化，同一次训练仍能精确恢复。没有按场景seed挑权重或动作的分支；显式seed只用于用户要求的可复现对照。

## 98. 原生模型与历史反馈候选的权重来源混用

### 遇到的问题

中断best、完整final、混合恢复骨干和历史反馈读出曾被混作同一模型。历史读出跨骨干使用时，其特征与参数不匹配。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1_server.py
Evo-1/Evo_1/model/action_head/observation_decoder.py
mujoco_pickplace/eval_policy_client.py
```

### 修改前

仅按目录名或loss挑选候选，不能确认实际服务；历史反馈拟合结果被误认为可直接用于新骨干。

### 使用的解决方法和对应效果

区分完整、best、中断和诊断候选；保存并核对checkpoint身份、归一化及源骨干哈希。反馈读出与其训练骨干绑定，历史拟合分支不自动挂入原生模型。

默认使用archive中完整成功权重及其原生动作输出，避免混用骨干、动作头与反馈读出。旧拟合方案只作为历史候选，不改变当前已采用的二分类、时间对齐和积分修复。

## 99. 纠正前缀的失败动作污染归一化统计

### 遇到的问题

失败策略前缀虽未作为动作目标，仍被旧writer计入动作统计，新增纠正后可能改变已学模型的输入输出尺度。

### 修改文件

```text
mujoco_pickplace/episode_dataset.py
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

统计包含policy_context动作；增量初始化没有明确保持匹配任务的原归一化范围。

### 使用的解决方法和对应效果

仅对真实专家监督阶段计算动作统计，前缀保留为历史；权重初始化继承匹配任务归一化并记录来源，新训仍使用实际数据统计。

已有失败动作不再改变监督动作范围。该问题属于新增纠正流程的缺口，不被误当成更早原始训练数据失败的解释。

## 100. 视频编码使当前观测依赖未来专家画面

### 遇到的问题

旧有损时序编码会使解码的当前帧受未来专家画面影响，训练视频与实时PNG不一致。

### 修改文件

```text
mujoco_pickplace/episode_dataset.py
mujoco_pickplace/check_dataset.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

保持前44帧相同而只改变后续画面，旧编码使当前帧29683个像素变化，单通道最大差52。

### 使用的解决方法和对应效果

采集和评估共用libx264rgb、RGB24、CRF0、GOP1、无B帧写入，元数据记录编码。训练与检查拒绝旧有损原生数据；不能用转码或修改元数据冒充原始无损观测。

相同控制下未来帧引起的当前像素变化归零，解码与原始RGB差为0。修复的是输入因果性，不把它直接等同于所有任务失败都已解释。

## 101. 无损数据写入后的重复整段视频校验

### 遇到的问题

每条视频写入后再整段解码，正式数据检查时又重复执行，浪费采集时间。

### 修改文件

```text
mujoco_pickplace/episode_dataset.py
mujoco_pickplace/check_dataset.py
```

### 修改前

写入路径承担视频编码、原子提交和重复整段回读。

### 使用的解决方法和对应效果

写入使用成熟FFmpeg的明确无损RGB参数和原子提交；真实流格式、帧数及必要解码校验集中到check_dataset。

保留RGB与真实数据检查，取消同一视频的重复整段扫描，采集和检查职责明确。

## 102. 训练窗口从视频起点反复解码无用前缀

### 遇到的问题

窗口缓存只有视频路径和时序信息，不含全部图像；旧PyAV路径每次从零解码并转换所有途经帧。

### 修改文件

```text
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

取齐历史帧前仍从头读、每帧转RGB，取齐后还可能继续解至末尾。

### 使用的解决方法和对应效果

按stream.time_base向前一关键帧seek，只在命中目标时间时转RGB，收齐即停；保留first-frame-at-or-after、1e-6容差及重复时间戳的独立图片副本。

两个实际视频的四个窗口逐像素一致。筷子后段解码118→27帧、RGB转换118→6、单窗口3.1464→0.1411秒；抽屉后段346→52帧、RGB转换346→6、6.3819→0.3825秒。是单窗口效果，不把正常流式读取判为故障，也不增加全数据RAM预载。

## 103. 子任务续训丢失父模型其他任务的归一化与来源

### 遇到的问题

只训练抽屉后，输出checkpoint可能缺少方块和筷子所需的元数据，导致共享模型不能按原入口服务其他任务。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

只保存本轮参与任务统计和来源，未参与任务的父信息被丢弃。

### 使用的解决方法和对应效果

保留父模型未参与任务的归一化与环境契约，实际训练来源和继承来源分别记录；准确区分模型初始化与完整optimizer/scheduler/sampler恢复。

子集续训产物保有三任务服务所需元数据；这解决元数据缺失，不把它当成共享动作行为绝不遗忘的保证。

## 104. 纠正回合的初始记忆窗口遗漏意图监督

### 遇到的问题

recover后进入接近阶段的有效记忆窗口没有意图标签；常规已向目标运动的窗口可能提供不恰当辅助监督。

### 修改文件

```text
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
Evo-1/Evo_1/scripts/train.py
```

### 修改前

纠正事件只识别单一阶段，复合事件及初始选屉窗口覆盖不完整。

### 使用的解决方法和对应效果

首个未运动决策和有真实有效历史的纠正memory_probe获得标签；无效或填充历史不提供意图标签。真实动作监督保留，更新缓存版本使旧窗口失效。

纠正历史与专家后缀接入现有正式训练，意图标签不泄漏到部署输入。没有因辅助监督而把场景seed或目标编号传给策略。

## 105. 未验证配对监督改变正式目标与梯度路径

### 遇到的问题

同场景跨目标配对、动作差异损失与意图概率detach曾在效果未确认时进入训练候选。既有100条独立seed数据不形成有效跨目标配对。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/model/action_head/direct_action.py
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

配对分组可能无有效数据，却改变标签范围和梯度路由；历史候选局部拟合不等于正式策略改善。

### 使用的解决方法和对应效果

撤回未采用的配对监督、动作差异损失和额外detach，恢复既有正式动作头目标。未形成可采用效果的adapter和残差候选不纳入默认。

当前默认不依赖这批未验证分支。保留已确认的标签、采样、归一化和编码修复，避免把局部拟合收益当成全任务提升。

## 106. 纠正采集误选成功回合并分派到错误阶段

### 遇到的问题

旧选择器把成功回合当成偏差，并将正确开屉后的卡住状态归入把手接近，不能提供真正缺失阶段的纠正。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/collect_data.py
```

### 修改前

13个旧失败中有7个选屉失败、4个开对后未抓起和2个释放失败，旧分派未覆盖这些真实阶段。

### 使用的解决方法和对应效果

先排除成功回合；按记录的真实物理阶段分别走早期把手、开屉后取物、空夹、释放时机接续。前缀只进历史，专家后缀才是动作监督，写入前仍要求接触、抬升、运输和稳定放置。

旧13个失败都有对应的源分派，不再用把手修正标签代替取物监督。采集合格不计入策略成功率。

## 107. opened_drawer标志不等于已经完全拉开抽屉

### 遇到的问题

滑轨仅移动约20mm时已出现opened_drawer，立即进入专家取物会过早介入；过晚受阻又可能无法安全恢复。

### 修改文件

```text
mujoco_pickplace/tasks.py
```

### 修改前

使用状态标签代替实际开屉量，较晚卡住状态仍尝试按正常路径恢复。

### 使用的解决方法和对应效果

离线源选择检查记录滑轨达到165mm；从真实完全开屉前后安全状态接续松把手、退出、抬升、转腕与取物。最高抽屉历史不足时前移到较早安全前缀。

原完全开屉过渡和最高抽屉接续都能按严格质量要求保存，状态回放差为0。滑轨真值仅用于离线源选择，不给部署策略输入。

## 108. 把手尚未双指接触时提前闭爪

### 遇到的问题

六个旧失败在轴向误差13.5–14.5mm时闭合；成功对照最大约10.3mm。另有最高屉回合在接近阶段已受阻。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/task_env.py
```

### 修改前

提前闭合后仍按一般把手偏差采集，标签不能恢复闭爪之前的正确接近状态。

### 使用的解决方法和对应效果

纠正源改为实际提前闭合前的安全前缀，受阻情况保留独立分派。专家使用开手纠正接近并加入竖直±15mm及物体接近偏移，标签二值化。

提前闭合的真实接续按原物理要求完成。离线12mm诊断分界不进入部署，当前模型仍根据真实观测输出动作。

## 109. 释放纠正没有检查介入时仍在真实持物

### 遇到的问题

物体最终落入接收区，不代表从抬起到放置的运输过程合法；接触已经丢失后再接续会伪造释放监督。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/episode_dataset.py
```

### 修改前

以最终落点代替介入状态，可能把掉落回合当成正常释放。

### 使用的解决方法和对应效果

object_release_timing要求正确开屉、已经抬起、双指仍承重；从真实持物前缀接续下放、完整开爪、撤离和稳定检查。

释放接续保留完整运输与释放证据，区内掉落和非法支承仍拒收。两条原状态接续的回放误差为0，但不计为策略通过。

## 110. 二分类监督与夹爪部署解码没有明确区分

### 遇到的问题

抽屉回归输出在0.5附近徘徊，终点已经放稳仍不开爪；切换训练目标还可能误改旧checkpoint推理语义。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/model/action_head/direct_action.py
Evo-1/Evo_1/scripts/Evo1_server.py
```

### 修改前

意图监督和夹爪二分类混在一起，旧权重未按自身模式解码。

### 使用的解决方法和对应效果

将抽屉夹爪二分类与意图监督解耦，对整块及实际首步的夹爪坐标应用对应损失，其余坐标和任务保留原损失。新二分类按当前logit符号解码，旧权重依自身配置；完整恢复继承原模式并拒绝隐式切换。

按用户选择，新direct+抽屉意图训练默认二分类，可显式关闭。既有权重未被更换，二分类只改变目标坐标，其他任务与填充尾部的梯度处理保留。

## 111. 释放首帧被后续保持窗口淹没

### 遇到的问题

松把手、脱离、下放、松物和撤离混作大阶段，首次释放事件在大量保持窗口中曝光不足。

### 修改文件

```text
Evo-1/Evo_1/dataset/lerobot_dataset_pretrain_mp.py
```

### 修改前

释放开始和释放后的保持没有独立事件，纠正从完全开屉后开始时首次开爪还可能位于recover阶段。

### 使用的解决方法和对应效果

将这些阶段纳入现有事件采样，首次开爪按真实标签单列；只重新分配抽屉原份额，其他任务份额不变。

原索引频数估算普通释放首帧单窗口权重约提高到3.08倍，训练窗口会更多访问转换时刻；部署不增加阶段状态或意图锁存。

## 112. 纠正回放把已碰撞回退的夹爪开度再次回退

### 遇到的问题

用记录中的回退后开度作为原始请求重放时，碰撞保护再次缩小开度，真实前缀无法一致复现。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/task_env.py
```

### 修改前

回放使用已执行的缩减开度，重复应用同一保护变换。

### 使用的解决方法和对应效果

从碰撞回退前的请求开度重放，仍由正式控制器执行保护；样本继续保存真实请求和实际执行证据。

两条实际纠正前缀最大状态误差为0；物理保护和最终成功标准没有放松。

## 113. 物体已放稳但仍闭爪的终态没有纠正入口

### 遇到的问题

physical_complete_terminal_pending回合已稳定支承，但未满足最终松手，普通释放源选择漏掉这类状态。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/episode_dataset.py
```

### 修改前

只匹配一般掉落或未放置，无法从仍双指持物的下放前段提供正确释放后缀。

### 使用的解决方法和对应效果

补齐已有object_release_timing分支，从合法持物状态接续；为已稳定低位放置增加单独的合法采集与证书，旧质量约束独立保留。

失败前缀仍只用于历史。可生成真实的抬离支承后开爪后缀，不把释放困难直接改判为成功。

## 114. 二值开爪1.0被释放掉落筛选漏掉

### 遇到的问题

旧筛选按连续开度区间识别释放，标准二值开爪1.0未进入对应纠正；落点重算还存在list除法错误。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/episode_dataset.py
```

### 修改前

部分真实二分类失败没有源分派，落点计算对Python列表直接做除法。

### 使用的解决方法和对应效果

将二值1.0纳入实际释放事件筛选，落点计算先转数值坐标；新增稳定低位支承纠正证书并拒绝运输掉落、未稳定、仍不满足夹持状态的样本。

两条实际物理接续分别提供31和35步后缀，前缀回放误差为0；筛选异常已修，候选数据没有替换正式训练集或权重。

## 115. 高位空夹与低位不开爪不能仅靠加训练步数解释

### 遇到的问题

相同checkpoint既有未形成双指接触就抬升，也有腕高约149.56mm时继续闭爪下探并被支承面拦截。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/episode_dataset.py
```

### 修改前

所有失败都被笼统归为训练不足，或仅向候选池继续加入同seed数据。

### 使用的解决方法和对应效果

按预测、执行和物理状态的首次分歧区分抓取与释放；离线纠正用实际持物包围盒规划安全释放，保留严格物理要求，不把真值规划加入部署。

同状态抬离再释放可完成，说明状态存在可行控制；该干预只说明故障位置，不当策略成绩。未采用的155/154条候选和退化短训不作为成功基线。

## 116. 错误模型服务被当成最新checkpoint

### 遇到的问题

输出目录写着18887，但端口实际仍服务旧80000步权重，导致新旧结果混淆。

### 修改文件

```text
Evo-1/Evo_1/scripts/Evo1_server.py
mujoco_pickplace/eval_policy_client.py
```

### 修改前

只核对客户端目录名，不能证明服务真实加载的文件。

### 使用的解决方法和对应效果

服务返回实际checkpoint路径、步数与权重哈希；客户端在创建评估输出前检查expected-checkpoint和expected-step，默认归档再核对已知SHA-256。

错误旧服务会在评估开始前被拒绝，结果保存真实身份。中断best、final、诊断与候选分开，不再拿目录名替代实际来源。

## 117. 方块已放稳但开爪分数低于旧阈值

### 遇到的问题

既有权重已搬运方块，但释放分数峰值约0.487，默认0.6阈值使其耗尽步数。

### 修改文件

```text
mujoco_pickplace/eval_policy_client.py
```

### 修改前

方块继续沿用不适合该归档的开爪阈值。

### 使用的解决方法和对应效果

保留闭爪阈值和两步开爪确认，仅将已使用归档的方块开爪阈值设为0.45；其他权重及显式覆盖保留原行为。

原失败在122步完成，两个旧成功场景仍以128/132步通过；没有放松物理接触或稳定放置判分。

## 118. 连续重规划反复丢弃当前时刻应有的开爪信号

### 遇到的问题

真实动作块多次出现首步闭合、随后打开，逐次只用新的首步导致转换不断后移。

### 修改文件

```text
mujoco_pickplace/eval_policy_client.py
Evo-1/Evo_1/scripts/Evo1_server.py
```

### 修改前

只执行当前预测单个二值结果，没有利用过去已预测的同一控制时刻。

### 使用的解决方法和对应效果

对最多四份已产生的动作块按同一当前控制tick对齐，仅聚合夹爪分数，以exp(-0.01*i)加权后通过原滤波。位姿仍来自当前模型，不提前执行未来动作，不用目标真值。

固定40000步权重同10场景从6/10到10/10，原六成功保留；另一组用户场景10/10。四个救回第一次差异位于接近把手阶段，不是只对终点强行开爪。

## 119. 方块接近阶段未触发重观测而过早闭爪

### 遇到的问题

原30场景两次失败在腕高约189mm、XY偏离20mm时闭爪，未接触便空手搬运。

### 修改文件

```text
mujoco_pickplace/eval_policy_client.py
Evo-1/Evo_1/scripts/Evo1_server.py
```

### 修改前

尚未进入精细重规划高度时，一次继续执行旧动作块多步。

### 使用的解决方法和对应效果

成功归档默认每步重观测，服务推荐与客户端一致；显式horizon和其他权重仍按各自设置。夹爪及物理标准保留。

原两个失败148/137步通过，同一组完整30回合为30/30。这是开发回归，不写成新独立成功率。

## 120. 下层取物前力矩饱和导致数值积分振荡

### 遇到的问题

原抽屉26/30中四个失败全是下层，均选对并开屉却没抬起；先出现跟踪偏差，再旋转越界和邻近把手拦截。

### 修改文件

```text
mujoco_pickplace/task_env.py
mujoco_pickplace/eval_policy_client.py
mujoco_pickplace/tasks.py
```

### 修改前

同一失败状态固定关节目标后，2ms积分仍使关节1/3约0.6rad/s振荡、87Nm饱和；IK前后所检查的积分状态不变，取消归一化裁剪也未救回。

### 使用的解决方法和对应效果

抽屉每2ms tick细分为两个1ms步，原100ms控制周期、接触时钟、展示及K=6间隔保留；筷子继续原积分。原训练指纹保留，仅精确已知控制器转换被认可，其他环境差异仍拒绝。

同一状态1ms、0.5ms均消除振荡。四个旧失败分别270/272/279/271步完成，首次抬起203–204步，保持拦截与状态越界均0；两个旧成功299/274步保留，正式入口完成7/7后按用户要求停止剩余回合。没有改权重、数据、力限、摩擦或判分，未采用位姿聚合和旋转取消裁剪。

## 121. 链接资产路径使相同环境的指纹不同

### 遇到的问题

Menagerie为链接目录时，按解析后的外部路径生成资产名会改变指纹，相同资产被误判不同环境。

### 修改文件

```text
mujoco_pickplace/tasks.py
```

### 修改前

资产物理路径与仓库逻辑相对路径混用。

### 使用的解决方法和对应效果

按仓库逻辑路径规范化XML引用，文件内容仍实际计算哈希；源控制器与真实MuJoCo版本继续记录。

相同资产在正常目录和链接目录下得到相同环境指纹，不屏蔽内容、引擎或控制器差异。

## 122. 采集覆盖后训练仍读取额外快照中的旧数据

### 遇到的问题

采集写cache，--pi-mem却被切换到另建releases数据层及专用YAML，重新采集的新数据未进入默认训练。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/dataset/config.yaml
README.md
```

### 修改前

同一项目有两套默认数据读取路径；额外拆分又使默认值分散。

### 使用的解决方法和对应效果

删除额外数据快照与重复YAML，训练回到原dataset/config.yaml，采集与训练同指cache。默认覆盖按用户要求保持True，--append仅显式启用。执行逻辑和训练参数归回现有入口，默认权重仍为archive成功归档。

现有100条方块、100条筷子、152条抽屉与成功权重所用数据逐文件一致；原150条抽屉保留，只归并既有两条纠正。以后覆盖cache后默认训练读取更新后的同一位置，不再冻结读取旧副本。

## 123. 终端输出被额外诊断日志与进度设置改写

### 遇到的问题

新增诊断改变了用户原有的训练进度条、参数输出和服务端mask输出。

### 修改文件

```text
Evo-1/Evo_1/scripts/train.py
Evo-1/Evo_1/scripts/Evo1.py
Evo-1/Evo_1/scripts/Evo1_server.py
Evo-1/Evo_1/model/internvl3/internvl3_embedder.py
```

### 修改前

额外提示混入正常输出，原MINT训练提示及默认刷新方式被改变。

### 使用的解决方法和对应效果

恢复tqdm默认刷新、完整参数明细和原优化器/初始化提示，服务端恢复image_mask/action_mask输出；附加运行诊断降为模块DEBUG，真实警告和异常保留。

正常loss、epoch、best/final及采集评估输出沿用原格式，不因此改变模型、数据读取、评分或保存逻辑。

## 124. 公开任务编号与历史数据内部ID混用

### 遇到的问题

任务合并重命名后，直接改数据或权重中的内部ID会使现有训练与评估契约失效。

### 修改文件

```text
mujoco_pickplace/tasks.py
mujoco_pickplace/collect_data.py
mujoco_pickplace/eval_policy_client.py
README.md
```

### 修改前

历史Task2是方块的记忆阶段，Task3是筷子，Task4是抽屉，公开入口与持久化ID混杂。

### 使用的解决方法和对应效果

公开入口统一Task1方块拾放（包含π-MEM）、Task2筷子、Task3抽屉；只在入口映射到持久化内部ID，原数据键、动作语义与权重契约保留。

用户使用当前三个编号与简短命令，既有数据与成功权重继续可用；本次目录归并没有再次修改这些任务行为。
