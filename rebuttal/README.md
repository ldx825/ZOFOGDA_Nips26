# Rebuttal experiments

这套目录用于重跑论文中除 LLM 微调外的实验。正式比较统一按实际训练 forward query 预算停止，同时记录 backward、sample-forward 和 wall-clock。实验矩阵、8 卡队列、结果聚合、Excel 与论文图均可重复生成。

## 已规划实验

共 1,082 个唯一 run，使用 seeds `14/42/81`。相同配置被多个消融复用，不会重复运行。

| Phase | 内容 | Runs |
|---:|---|---:|
| 1 | 主方法对比、`M={1,5,10,15}`、四种动量、重跑 TTGDA/TTSGDA | 270 |
| 2 | `delta`、two/one-sided/direct、sphere/Rademacher | 210 |
| 3 | `c1/c2 × batch size`，按新 query 标准重跑旧消融 | 192 |
| 4 | hidden dimension | 90 |
| 5 | `M×delta`、`M×momentum` 交互 | 144 |
| 6 | 一阶方法学习率与时间尺度诊断 | 176 |

数据集包括 6 个 AUC 数据集与 4 个稳健二分类数据集。LLM 微调实验不在本队列中。

## 公平计数口径

- Two-sided ZO：每步 `2M+1` 次训练 forward；其中 `2M` 为 x 的 function query，`1` 为 y 梯度图。
- One-sided ZO（共享 base）：每步 `M+2` 次训练 forward。
- Direct/TTGDA/TTSGDA：x/y 共用一张 autograd 图，每步 `1` 次 forward 和 `1` 次 backward。
- evaluation forward 单独计数，不进入训练预算。
- batch size 默认 128；Excel 同时提供 sample-forward，避免 forward-call 口径掩盖 batch 差异。
- 因优化步具有不可拆分的 query 成本，最终实际 query 是不超过 `Q` 的最大整步值，最多比 `Q` 少 `step_cost-1`。

主默认值对齐原代码：AUC 使用 `gamma=lambda=0.01`，稳健分类使用 `gamma=lambda=1`，默认 `delta=1e-4`。

## 文件入口

- `run_experiment.py`：统一 AUC/稳健分类、ZO/FO runner 和真实计数器。
- `build_matrix.py`：生成完整配置、注册表与 8 个负载均衡队列。
- `worker.py`：断点续跑；已完成 run 自动跳过，失败记日志后继续。
- `aggregate_results.py`：聚合原始曲线、三 seed mean/std、生成 Excel 和 query 曲线图。
- `test_query_accounting.py`：CPU 快速回归测试，验证 three estimator 的实际 query 数。
- `tables/experiment_registry.csv`：1,082 个 run 的唯一注册表。
- `tables/rebuttal_experiment_results.xlsx`：自动刷新的结果记录工作簿。
- `tables/rebuttal_results_simple.xlsx`：老师版六列结果简表；每个实验条件按 ZOFOSGDA/TTGDA/TTSGDA 连续三行排列（推荐直接查看）。
- `plots/<group>/`：按实验组、任务、数据集生成的 mean±std PNG。

## 常用命令

在仓库根目录执行：

```bash
python rebuttal/test_query_accounting.py
python rebuttal/build_matrix.py
bash rebuttal/scripts/launch_tmux.sh
bash rebuttal/scripts/status.sh
bash rebuttal/scripts/refresh_results.sh
```

后台 session 为 `zofo_rebuttal_gpu0` 至 `zofo_rebuttal_gpu7`；`zofo_rebuttal_report` 每 10 分钟刷新 Excel 和图片。查看某张卡：

```bash
tmux attach -t zofo_rebuttal_gpu0
```

退出查看但不停止任务：按 `Ctrl-b`，再按 `d`。如确需停止整组队列：

```bash
bash rebuttal/scripts/stop_tmux.sh
```

重复执行启动脚本不会创建重名 session；worker 会读取 `summary.json` 并跳过已完成项。单个失败不会阻塞其余任务，详情在 `rebuttal/logs/<run_id>.log`。

## 结果填写与出图

Excel 不是空模板：它直接读取每个 run 的 `summary.json` 与 `metrics.csv`。主要分页：

- `Status Dashboard`：phase/GPU/status 数量；
- `Run Registry`：全部计划与分配；
- `Raw Results`：每个 run 的最终结果和资源计数；
- `Raw Curves`：每个 query checkpoint；
- `Summary`：三 seed 的 mean/std/count；
- `M Sweep`、`Momentum`、`Perturbation` 等：可直接整理进 rebuttal 的分组表。

画图横轴固定为 training forward queries，阴影为不同 seed 的标准差。任意时刻手工刷新一次即可得到当前已完成 run 的表和图；自动刷新不会修改原始结果。

## 依赖与复现

```bash
pip install -r rebuttal/requirements.txt
```

正式配置都保存于 `configs/runs/`，结果目录名就是 `run_id`。若修改矩阵，重新运行 `build_matrix.py` 后应先确认 `manifest.json` 和 registry；不要在旧 worker 仍运行时重建队列。
