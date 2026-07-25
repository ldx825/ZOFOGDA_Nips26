# AUC Rebuttal：原论文代码与重跑范围审计

更新时间：2026-07-25

## 结论

本次只负责 **AUC maximization**，不再运行 robust classification 或 LLM 实验。

正式实验必须直接基于原论文的三份独立实现：

- `26.1.29.ZOFOGDA_AUC/ldx_zofogda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttgda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttsgda_auc.py`

昨天新增的 `rebuttal/run_experiment.py` 改写了训练框架，不能作为论文原实验的正式复现入口；其已经产生的结果只能视为 exploratory results，不进入 rebuttal 表格。

现有 `ldx_run_*.sh` 也不能直接整份执行。这些文件经过多次临时修改，文件名与当前未注释的实际命令已经不一致。例如名为 `ldx_run_ttgda_auc_0.sh` 的活动命令实际运行的是 `ldx_zofogda_auc.py`。新启动脚本应重新生成，但只能调用上述三份原始 Python 实现。

## 判断依据

用于恢复原训练协议和超参数的历史参考来源是：

```text
26.1.29.ZOFOGDA_AUC/result_plot/<dataset>/*.csv
```

原因如下：

1. 该目录保存了当时绘图/汇总使用的历史结果，但其数值不进入本次正式汇总；
2. 每个 CSV 的第二行保存了完整 `#cfg`；
3. CSV 的命名格式分别与三份 `ldx_*.py` 的 `build_log_path()` 完全对应；
4. `plot_res.py` 直接引用其中的三方法、三种子文件；
5. CSV 中的模型、数据预处理和参数与三份原脚本的数据类字段一致。

Git 历史只能证明这些代码和结果在仓库首次提交时一起进入版本库，不能证明结果的生成过程。`result_plot` 仅用于恢复模型、数据处理和超参数；本次正式数值全部来自独占 GPU 上使用 seeds `14/42/81` 的可审计重跑。

## 原论文 AUC 协议

共同设置：

- 数据集：`climate, german, svmguide1, spambase, a9a, ijcnn1`
- epochs：`500`
- seeds：`14, 42, 81`
- hidden：`128`
- standardize：`True`
- CLI batch size：`128`
- dual clip：`10`

注意：TTGDA 原代码会把训练 batch 强制设为整个训练集大小；命令行的 `batch_size=128` 对 TTGDA 训练不生效，只影响测试 loader。为了保持“与论文代码一样”，本次不能擅自把 TTGDA 改成 minibatch 128。

### 精确超参数

| Dataset | TTGDA `(lr_x, lr_y)` | TTSGDA `(lr_x, lr_y)` | ZOFOSGDA `(gamma, lam)` |
|---|---:|---:|---:|
| Climate | `(0.01, 0.0001)` | `(0.01, 0.0001)` | `(0.01, 0.01)` |
| German | `(0.01, 0.0001)` | `(0.01, 0.0001)` | `(0.005, 0.005)` |
| Svmguide1 | `(0.01, 0.0001)` | `(0.01, 0.0001)` | `(0.001, 0.001)` |
| Spambase | `(0.01, 0.0001)` | `(0.01, 0.0001)` | `(0.04, 0.04)` |
| A9a | `(0.01, 0.0001)` | `(0.01, 0.0001)` | `(0.01, 0.01)` |
| Ijcnn1 | `(0.08, 0.0008)` | `(0.01, 0.0001)` | `(0.01, 0.01)` |

ZOFOSGDA 的共同设置：

```text
eta_k=10
eta_m=1000
c1=3
c2=3
zo_delta=0.001
zo_q=5
batch_size=128
```

原脚本 CLI 默认的 `zo_delta=1e-4` 和 `epochs=200` 不是论文最终配置，不能使用默认值启动。

## 需要重跑的实验

### 1. 三方法原配置干净复跑

旧 `result/` 和 `result_plot/` 文件只保留作历史审计，不进入正式表格或默认绘图。三种方法均使用原配置、500 epochs 和统一 seeds `14/42/81` 独立重跑：

- 6 datasets × 3 methods × 3 seeds = 54 runs；
- 每张 GPU 同时最多一个训练上下文；
- 外部上下文进入时自动终止并从 epoch 1 重跑；
- 每个 run 保存原始 CSV、逐 epoch wall-time、GPU 独占审计和 summary。

### 2. `M` 的影响

当前实验代码中对应 `zo_q`：

```text
M ∈ {1, 5, 10, 15}
```

除 `zo_q` 外，每个数据集保持上表中的原论文 ZOFOSGDA 配置。`M=5` 是原论文设置，可与 timed baseline 复用。

### 3. 零阶扰动半径

评审明确询问 smoothing parameter 的敏感性。按下列设置：

```text
delta ∈ {1e-4, 1e-3, 1e-2}
```

`delta=1e-3` 是原论文设置，可与 timed baseline 复用。除非老师另外明确要求，本组不额外引入 one-sided、Rademacher 或 direct-gradient 等改变算法定义的变体。

### 4. x/y 动量消融

只在原 `ldx_zofogda_auc.py` 上增加最小开关：

```text
full momentum
no x momentum
no y momentum
no x and no y momentum
```

原递推为：

```python
w  = (1 - beta_t)  * w  + beta_t  * w_hat
vy = (1 - alpha_t) * vy + alpha_t * vy_hat
```

关闭对应动量时直接使用当前估计：

```python
w = w_hat
vy = vy_hat
```

不能用 `c1=0` 或 `c2=0` 代替；那会冻结历史估计，不等于去掉动量。`full momentum` 与 timed baseline 复用。

## 实验数量

若以上三组新消融都覆盖 6 个论文 AUC 数据集和 3 个种子：

| 内容 | 唯一 runs |
|---|---:|
| 三方法原配置 timed rerun | 54 |
| M sweep 新增值：1、10、15 | 54 |
| delta sweep 新增值：1e-4、1e-2 | 36 |
| momentum 新增：no-x、no-y、no-both | 54 |
| 合计 | 198 |

其中所有 baseline 条件只跑一次，不重复排队。

## Epoch、query 与时间的记录方式

所有实验仍以原论文的 `500 epochs` 为停止条件，不再按统一 query budget 提前停止。

原训练 CSV 继续记录原有的 `epoch/train_loss/test_auc` 等字段；外层 worker 另存：

```text
epoch_times.csv          # epoch / elapsed_wall_time_sec / time_ms
gpu_audit_attempt*.csv  # elapsed time / GPU context count / exclusive
summary.json             # AUC / query / 总时间 / 独占状态
```

`summary.json` 记录总 wall-time、优化步数、forward/backward calls、sample-function queries、GPU 独占状态和结果文件路径。

公平比较时同时报告：

1. 500 epoch 后的 AUC；
2. AUC vs cumulative sample-function queries；
3. AUC vs wall-clock。

不能只比较 Python 层的 forward 调用次数：原 TTGDA 一次训练 forward 包含整个训练集，而 TTSGDA/ZOFOSGDA 一次 forward 只包含一个 minibatch。主 query 口径应按实际处理的样本目标函数次数累计，原始 forward-call 数作为审计辅助列保留。

## 代码修改边界

允许对原脚本做的修改只有：

- 增加 x/y 动量独立开关；
- 增加 query、运行时间和显存日志；
- 增加独立输出目录/实验标签；
- 保证已有 CLI 参数和默认训练数学逻辑不变。

不允许：

- 把三种方法重新写入一个共享训练循环；
- 修改模型、数据划分、standardization 或初始化；
- 把 TTGDA 从 full-batch 改成 minibatch；
- 修改原论文超参数来追求更好的结果；
- 覆盖旧论文 CSV。

正式开跑前应先对每种方法各执行一个 1-epoch smoke test，并检查新旧脚本在同一 seed 下首个 epoch 的 AUC、loss 和参数轨迹一致；只有一致后才启动 8 卡队列。
