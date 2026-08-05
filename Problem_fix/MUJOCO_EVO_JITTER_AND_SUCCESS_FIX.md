# MuJoCo + Evo-1 抖动 / 成功率 / 视频差异 修复报告

记录针对三个问题（机械手颤动、正确率偏低、视频与 LIBERO 差异大）的排查流程与修改。

## 1. 排查流程（按"先数据、再开环、后闭环"）

### 1.1 先确认训练数据采集没问题

在 WSL 里用实际环境（`mujoco` conda env, `MUJOCO_GL=egl`）逐条动作回放，量化轨迹：

- 原环境：单条命令位移为 0 时，eef 每步仍漂移 **0.014–0.070**（根本停不住）；
  命令 +0.02 x，实际位移可能为负；0.6s 子步内振幅达命令的 **2.4 倍**。
  → 控制环节欠阻尼极限环，采集到的专家轨迹本身就抖（x 方向速度 26 步里翻转 15 次）。
- 修复后：5/5 成功，速度精确匹配命令，整条轨迹 x 翻转仅 2-3 次。

结论：**采集入口的第一个 bug 在控制层**，不修控制，采出来的数据就是抖的。

### 1.2 开环测试模型动作是否本身抖动

写 `eval_open_loop.py`：一次推理拿到 action chunk，不回传观测整段执行，并统计
action chunk 内部平滑度。

对旧模型（`evo1_mujoco_panda7_multiview_h10_clean_ds/step_best`，旧抖动数据 + 2000 步）：

```text
action_jitter(mean|dd|) = 0.053 ~ 0.128     (专家数据基线 ~0.003-0.007)
dx 连续方向翻转 = 2~6 / 9
开环成功率 0/6
```

结论：**模型输出本身就在抖**（旧数据抖 + 欠训练），不是单纯的执行问题。

## 2. 控制层修复（pick_place_env.py）

| 项 | 原值 | 新值 | 说明 |
|---|---|---|---|
| joint damping | 4.0 | **40.0** | 临界阻尼，消除极限环震荡（historical tuning script 扫描验证） |
| nstep | 30 (0.6s/动作) | **100 (0.2s/动作)** | 闭环更紧，动作→位移映射可靠 |

参数扫描结果：`(damping=4, nstep=30)` 单命令误差 0.0084、保持漂移 0.044；
`(damping=10, nstep=10)` 误差 0.0028、保持漂移 0.0054。

另新增 `PickPlaceEnv.step_video(action, frames_per_step)`：把 0.2s 的物理推进分摊到
多帧渲染，eval 视频从"每帧跳 0.6s / 6 倍速"变成接近实时、平滑。

## 3. 数据层（collect_data.py / convert_to_evo_lerobot.py）

- `collect_data.py`：加 argparse + 质量门（成功、长度 10-120、eef 速度翻转比、动作范围），
  用固定控制重采了 **100 个干净 episode**（`current MuJoCo_Panda7_Multiview_V2 collection`）。
  数据 action mean-abs-diff 从旧的 0.0177 降到 ~0.003-0.007。
- `convert_to_evo_lerobot.py`：输出标准 **LeRobot v2.1**（Evo-1 loader 必需）+
  可选 **v3.0**（与 `so100_evo1/lerobot-main` 一致），`meta/info.json` 完整 schema
  （`codebase_version`, `robot_type: panda`, features 含 `video_info` 等）。
- 新数据集：`Mujoco_training_dataset/MuJoCo_Panda7_Multiview_V2`（100 集）。
  `check_dataset.py` 验证 2712 个 horizon=50 窗口可加载。

### 重要坑：LeRobot 窗口缓存

缓存路径 key 用的是 config 里的 dataset *名字*（`MuJoCo_PickPlace_Dataset`），
不是数据目录路径。换数据集后若不删除
`Mujoco_training_dataset/cache/mujoco_pickplace/mujoco_pickplace/MuJoCo_PickPlace_Dataset/`，
loader 会**静默复用旧数据的窗口样本**。重训前必须清空。

## 4. 训练

Evo-1 original `Evo-1/Evo_1/scripts/train.py`：
- 干净数据 + horizon=10 + per_action_dim=24 + 4000 步
- `LD_PRELOAD=/home/user/miniconda3/envs/Evo1/lib/libstdc++.so.6`（flash_attn 需要
  CXXABI_1.3.15，系统 libstdc++ 没有）
- 保存到 `ckpt/evo1_mujoco_panda7_multiview_h10_clean_v2/`

（结果以训练日志 + 开环/闭环测试为准，见文末。）

## 5. 推理与评测

- `Evo1_server.py`：加 `--ckpt-dir` / `--port` 参数，切换 checkpoint 不再改源码。
- `eval_policy_client.py`：用 `step_video` 生成自然速度视频；`--horizon` 对齐模型
  horizon；视频不再是 6 倍速 + 逐帧跳变。
- `eval_open_loop.py`：先开环确认模型动作平滑，再上闭环。

## 6. 视频与 LIBERO 的差异

已处理：抖动（控制+模型）、播放速度（原来每帧=0.6s 模拟、6 倍速）。场景本身仍是
"简化 Panda"（capsule 连杆 + 球状 eef），与 robosuite 的真实外观有差距，这属于
"简单搭建"的范围；画面主体、光照已可看清 cube/goal/eef。

## 7. 结论（训练 + 评测结果）

- 抖动根因 = 控制欠阻尼 + 模型在抖动数据上欠训练，两层都修了。
- 训练：干净数据（100 集）训到 4000 步，最佳 loss **0.217**（旧模型 0.33）。
- 闭环评测（`eval_policy_client.py --horizon 10`，10 集）：
  - **success_rate = 0.600（6/10）**，平均完成 72.2 步。
  - 旧模型（h10_clean_ds）为 0/10。
- 开环测试（`eval_open_loop.py`）：
  - 新模型位置动作抖动 pos ≈ 0.008-0.014（旧模型 0.05-0.13，大幅改善）；
    夹爪输出从"开"平滑过渡到"关"（正确抓取行为）。
- 推理端改进（`eval_policy_client.py`）：
  - 夹爪**滞回**（>0.7 开 / <0.3 关 / 否则保持），避免 flow-matching 输出在 0/1 之间翻腾；
  - 位置增量轻量 EMA 平滑（`--smooth-alpha`），压低残余噪声；
  - `--no-save-video` 时用 `env.step()` 快速评测，`--save-video` 时用 `step_video` 渲染自然速度视频。
- 视频：速度/抖动修复后观感接近 LIBERO 的实时回放，场景外观为简化为题（capsule 连杆 + 球状 eef）。

## 8. 第二轮视觉/夹爪整改（V2 场景）

针对用户的 4 点新反馈，重设计 `pick_place_env.py`：

- `pick_place_env.py`：手指红点——去掉原来"一个大红球"的 eef 表示，改为**两根手指，整个指尖渲染为红色**
  （像 libero 的指甲），front/overhead/wrist 视角均可见。
- `pick_place_env.py`：夹持接触——立方体缩小到 40mm（配套手指间距）；手指闭合位置在立方体表面内侧 6mm，
  **手指真实压紧立方体（接触力）**，不再是 teleport 悬浮在下方；抓取时立方体精确位于手指中点、与手指同高。
- `pick_place_env.py`：IK 朝向约束——位置 IK 会让手腕 roll 任意、手指不朝下；新增 `_solve_ik` 的 z 轴对齐约束，
  使夹爪保持竖直、手指朝下，专家 10/10 成功。
- `pick_place_env.py`：wrist 相机——不再安装在手腕上（手指会穿过近平面），改为**桌面固定高度 + 水平平移跟踪立方体**
  （`_update_tracking_cam`），朝向立方体，既不过近穿模也始终框住抓取点。
- `pick_place_env.py`：光照——提升 headlight/主光强度，场景不再偏暗。
- `eval_open_loop.py` / 抖动源：确认抖动来自**模型生成的 action 方向反转**（相邻动作方向相反 ~18%，专家为 0%），
  控制层已修复；数据侧用平滑专家重采。
- `Evo1_server.py`：flow-matching ODE 步数 32→64（更平滑的积分，减少反转）。
- `eval_policy_client.py`：EMA 平滑 + 夹爪滞回（沿用第 7 段的推理端改进）。

V2 数据：`raw_mujoco_panda7_multiview_v2/` → `MuJoCo_Panda7_Multiview_V2`（lerobot v2.1 + v3）。

### V2 训练与评测结果

- 训练：V2 干净数据（100 集）训到 4000 步，最佳 loss **0.2305**。
- 开环测试（`eval_open_loop.py`，v2 step_best）：
  - 位置抖动 pos ≈ 0.0097-0.0145；**dx 方向翻转降到 0-4（均值 ~2.3）**，比旧模型（均值 ~4）减少一半——抖动修复见效。
  - 夹爪从"开"平滑过渡到"关"。
- 闭环评测（`eval_policy_client.py --horizon 10`，10 集）：
  - **success_rate = 0.500（5/10）**（模型 ~50% 上限，波动较大；一次视频评测为 1/6）。
- 视频：6 集已保存到 `outputs/eval_videos/20260805_015906/task1/`，帧分析确认**红色指尖（1121px）与蓝色 cube（1226px）清晰可见**。
- 剩余限制：成功率受模型位置引导能力限制（40mm 小 cube 更难精确抓取/放置）；夹爪原始输出偶有搬运中重新打开，
  `eval_policy_client.py` 已用滞回 + 平滑处理。
