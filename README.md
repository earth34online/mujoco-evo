# MuJoCo + Evo-1 π-MEM

同一份视觉语言动作模型在 Franka Panda 上依次完成方块拾放、筷子入盒和记忆抽屉取物。模型使用 6 帧、间隔 1 秒的真实观测短期记忆。环境、数据和评估按三个当前任务运行，旧数据中的内部标识由代码兼容。

| 任务 | 行为 | 严格判分 |
|---|---|---|
| Task1 | 方块拾放 | 拾取后放入目标区，松手且放置稳定 |
| Task2 | 筷子入盒 | 双指夹起，搬运无掉落，入盒后松手静置 |
| Task3 | 记忆抽屉取物 | 基于观测历史打开正确抽屉，取物并稳定放入接收区 |

当前发布权重为 `ckpt/archive/evo_pi-mem_success`，来源为 40,000 步续训的 `step_final`。本次修复继续使用这份权重，无需为了积分修复重训或重新采集数据。

2026-11-18 的默认执行设置和验证结果如下。历史独立留出与修复后的开发回归分别记录，详见 [修复与评估记录](Problem_Fix/README.md)。

| 任务 | 原独立留出 | 当前修复及已完成验证 |
|---|---|---|
| Task1 方块 | 28/30 | 每步重观测；原同组完整回归 30/30 |
| Task2 筷子 | 30/30 | 原物理积分调用保留；一例 112 步控制器回放状态差为 0 |
| Task3 抽屉 | 26/30 | 双半步积分消除下层振荡；四个原失败 4/4，两个原成功对照 2/2，另经正式入口完成 7/7 |

最后一组抽屉评估原计划 30 回合，按用户收尾要求在完成 7 回合后停止，新随机三任务扩展评估未启动。这些修复后结果属于开发回归，尚无修复后的完整 30 回合或新独立成功率，不将其写成 30/30。

## 准备环境

在 WSL 从仓库根目录运行项目命令。现有本机环境为 `Evo1` 和 `mujoco`。训练和服务端使用 Evo1 环境，采集和闭环客户端使用 mujoco 环境。MuJoCo Python 和运行库均为 3.11.0。Evo-1 环境安装说明和依赖见 [上游说明](Evo-1/README.md) 与 [依赖列表](Evo-1/Evo_1/requirements.txt)。场景依赖仓库根目录下的 MuJoCo Menagerie Panda 资产。

权重和原始视频数据在本机单独保存。发布权重目录应包含 `checkpoint.json`、`config.json`、`norm_stats.json` 和 `mp_rank_00_model_states.pt`。从空仓库准备训练数据时，按下文采集。原发布训练数据的本机快照保存在 `Mujoco_training_dataset/releases/evo_pi-mem_success`。

## 运行发布模型

服务端终端

```bash
conda activate Evo1
cd Evo-1/Evo_1
python scripts/Evo1_server.py
```

客户端终端，从仓库根目录执行

```bash
conda activate mujoco
python -m mujoco_pickplace.eval_policy_client
```

默认连接端口 9000，检查归档 checkpoint 的路径、40,000 步身份和权重哈希，按现行三个任务各评估 10 回合。发布权重会自动启用 Task3 的时间对齐夹爪预测，Task1 开爪阈值为 0.45，发布权重每个动作后重新观测。抽屉每个 2 ms 物理 tick 细分为两个 1 ms 积分步，动作周期仍为 100 ms；新旧控制器兼容关系与原训练指纹差异会记录在评估报告中。评估会生成随机种子并记录，结果目录包含视频、逐步动作、运行日志 `client.log`、权重身份和 `summary.json`。单任务用 `--task 3`，三任务较大评估用 `--num-episodes 30`。需要明确对照时传入已登记的 `--start-seed`。

另选权重时，服务端传 `--ckpt-dir`，客户端传相同的 `--expected-checkpoint` 和实际的 `--expected-step`。基于新代码的二分类抽屉权重可用 `--task3-gripper-temporal-ensemble` 显式试验，在发布权重上用 `--no-task3-gripper-temporal-ensemble` 做同权重对照。

## 采集和检查数据

采集前请选择新目录以保留原始数据。每种任务默认采集 150 条严格合格的轨迹。可用 `--task` 和 `--num-episodes` 限定范围。

```bash
conda activate mujoco
python -m mujoco_pickplace.collect_data --dataset-dir Mujoco_training_dataset/next
python -m mujoco_pickplace.check_dataset --dataset-dir Mujoco_training_dataset/next --raw-only
```

采集保存 RGB 无损视频、带物理时间戳的 Parquet 和质量元数据。训练窗口使用回合内历史，不跨 episode。采集和评估的物理成功标准一致，并分别记录双指接触、运输、选屉和释放证据。

## 训练

本机快照和归档权重就绪后，从仓库根目录开始。

```bash
conda activate Evo1
cd Evo-1/Evo_1
python scripts/train.py --pi-mem
```

该配置沿用已运行的单卡 BF16 训练配方，从归档加载模型参数，开始新的优化阶段。批大小 2，学习率 1e-5，新阶段 40,000 步，每 2,000 步写一次 checkpoint，保存目录按时间生成。训练种子默认随机，原归档权重保留。需要用其他数据时，以 [发布数据配置](mujoco_pickplace/pi_mem_release_data.yaml) 为模板，使用 `--dataset_config_path` 指向新的三任务配置。建议先用正式数据检查入口核对快照和模型维度。

恢复已有训练的优化器、调度和采样游标时，使用该训练本身的配置、数据与输出路径，按 `scripts/train.py --help` 中的 `--resume` 和 `--resume_path` 恢复。模型初始化和精确续训有独立语义。闭环成绩由评估确定，保留周期权重以便按同场景比较。

## 诊断与代码结构

- `mujoco_pickplace/` 包含场景任务、数据写入与检查、WebSocket 客户端和闭环判分。
- `Evo-1/Evo_1/` 包含训练、推理、数据读取和动作头，沿用 [MINT Evo-1](https://github.com/MINT-SJTU/Evo-1) 的入口布局。
- `mujoco_pickplace/open_loop_test.py` 支持数据审计、动作预测和物理回放。开环误差与闭环任务结果分别报告。
- `Problem_Fix/` 保留已落地的修复、判分口径和关键证据。

快速检查发布入口及夹爪时序时，在装有 pytest 的环境中，从仓库根目录运行 `python -m pytest tests/test_pi_mem_release.py`。
