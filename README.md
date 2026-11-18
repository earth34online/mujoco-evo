# MuJoCo + Evo-1 π-MEM

同一份视觉语言动作模型在 Franka Panda 上依次完成方块拾放、筷子入盒和记忆抽屉取物。模型使用 6 帧、间隔 1 秒的真实观测短期记忆。环境、数据和评估按三个当前任务运行，旧数据中的内部标识由代码兼容。

| 任务 | 行为 |
|---|---|
| Task1 | 方块拾放 |
| Task2 | 筷子入盒 |
| Task3 | 记忆抽屉取物 |


## 准备环境

在 WSL 从仓库根目录运行项目命令。现有本机环境为 `Evo1` 和 `mujoco`。训练和服务端使用 Evo1 环境，采集和闭环客户端使用 mujoco 环境。MuJoCo Python 和运行库均为 3.11.0。Evo-1 环境安装说明和依赖见 [上游说明](Evo-1/README.md) 与 [依赖列表](Evo-1/Evo_1/requirements.txt)。场景依赖仓库根目录下的 MuJoCo Menagerie Panda 资产。

权重和原始视频数据在本机单独保存。发布权重目录应包含 `checkpoint.json`、`config.json`、`norm_stats.json` 和 `mp_rank_00_model_states.pt`。从空仓库准备训练数据时，按下文采集。采集与默认训练统一使用 `Mujoco_training_dataset/cache` 下的三个任务目录。

## 采集和检查数据

每种任务默认采集 150 条严格合格的轨迹，按要求覆盖所选任务的原数据；未选任务不受影响。可用 `--task` 和 `--num-episodes` 限定范围，明确追加时使用 `--append`。

```bash
conda activate mujoco
python -m mujoco_pickplace.collect_data
python -m mujoco_pickplace.check_dataset --raw-only
```

采集保存 RGB 无损视频、带物理时间戳的 Parquet 和质量元数据。训练窗口使用回合内历史，不跨 episode。采集和评估的物理成功标准一致，并分别记录双指接触、运输、选屉和释放证据。

## 训练

`cache` 中的训练数据和 `archive` 中的归档权重就绪后，从仓库根目录开始。

```bash
conda activate Evo1
cd Evo-1/Evo_1
python scripts/train.py --pi-mem
```

该配置沿用已运行的单卡 BF16 训练配方，从归档加载模型参数，开始新的优化阶段。需要用其他数据时，直接调整 [数据配置](Evo-1/Evo_1/dataset/config.yaml) 中的三任务路径，或用 `--dataset_config_path` 明确指定另一份配置。建议先用正式数据检查入口核对数据和模型维度。

恢复已有训练的优化器、调度和采样游标时，使用该训练本身的配置、数据与输出路径，按 `scripts/train.py --help` 中的 `--resume` 和 `--resume_path` 恢复。模型初始化和精确续训有独立语义。闭环成绩由评估确定，保留周期权重以便按同场景比较。

## 诊断与代码结构

- `mujoco_pickplace/` 包含场景任务、数据写入与检查、WebSocket 客户端和闭环判分。
- `Evo-1/Evo_1/` 包含训练、推理、数据读取和动作头，沿用 [MINT Evo-1](https://github.com/MINT-SJTU/Evo-1) 的入口布局。
- `mujoco_pickplace/open_loop_test.py` 支持数据审计、动作预测和物理回放。开环误差与闭环任务结果分别报告。
- [Problem_Fix/README.md](Problem_Fix/README.md) 是唯一修复记录，集中保存已落地修复、判分口径和关键证据。


## 运行发布模型

默认服务端和客户端均使用 `ckpt/archive/evo_pi-mem_success`，自动启用已验证的执行设置，无需额外参数。

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
