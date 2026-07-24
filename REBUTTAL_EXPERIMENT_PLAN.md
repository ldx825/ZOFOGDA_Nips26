# NeurIPS 2026 Rebuttal 实验执行计划（按老师要求修订）

> 论文：*Hybrid Zeroth- and First-Order Method for Stochastic Nonsmooth Minimax Optimization*
> Submission：10118
> 项目目录：`/hy-tmp/ZOFOGDA_Nips26`
> 修订日期：2026-07-24

## 1. 最终实验主线

老师要求的三组实验是：

1. **M 的影响**：`M in {1, 5, 10, 15}`；
2. **零阶扰动的消融**；
3. **分别去掉 x/y 动量**：去掉后直接使用当前梯度估计更新，并在相同 query 下比较；所有一阶方法也要重新跑。

所有实验的共同条件：

- 使用相同的累计训练 forward 次数作为停止条件；
- 报告指标随 cumulative forward queries 的曲线；
- 额外报告 wall-clock、一阶 backward 次数和 sample-forward 数；
- evaluation forward 单独计数，不能混入训练 query；
- batch size、数据划分、模型、seed、评估频率保持一致；
- 旧实验只有 epoch 横轴且没有 query 计数，不能直接作为这次公平比较的主结果。

建议主数据集使用：

- Svmguide1：低维；
- Spambase：中维；
- A9a：高维。

每组正式实验统一使用 seeds `{14, 42, 81}`，报告 mean ± std；时间紧时可先用 seed 42 做 smoke test，但最终进入 rebuttal 的结果必须补齐三组 seed。

## 2. 与我之前理解的差异

| 项目 | 之前的理解 | 按老师要求修正后 |
|---|---|---|
| `M` | 我使用代码变量 `q` 表示方向数，并计划额外做维度扩展 | 主实验直接按老师指定的 `M={1,5,10,15}`；代码中暂映射为 `zo_q`，图表统一写 `M` |
| 零阶扰动 | 主要理解为 smoothing radius `delta` 敏感性 | 为避免漏项，拆成“扰动半径”和“扰动形式”两层 |
| 公平预算 | 区分 ZO function oracle 与 FO gradient oracle | 主横轴按老师指定的实际 forward 次数统一，同时附列 backward 次数 |
| 一阶 baseline | 原计划只补缺失 seed，部分旧结果可能复用 | TTGDA、TTSGDA 必须全部用新计数逻辑重跑 |
| 动量消融 | Full / No-x / No-y / No-both | 版本不变，但关闭后必须是直接梯度更新，不能通过 `c1=0` 或 `c2=0` 实现 |

### 2.1 `M` 的术语风险

论文和代码里存在命名混用：论文理论中的 `M` 可能指 batch/样本平均数，而当前代码使用 `zo_q` 表示每次 x 更新的随机方向数。

本文档暂按下面的映射执行：

```text
B = minibatch size，固定为 128
M = 每次 x 更新的随机方向数，对应代码参数 zo_q
```

从 reviewer 对“一次零阶估计包含多次 function evaluations”的质疑，以及老师强调相同 forward 次数来看，老师所说的 `M` 更可能是随机方向数。如果老师实际指 minibatch size，则需要把实验改为 `batch_size in {1,5,10,15}`，不能混着解释。

## 3. 相同 query 的严格定义

### 3.1 主口径：实际训练 forward 次数

主结果统一使用：

```text
cum_train_forward_queries
```

它统计参数更新阶段实际调用目标函数 forward 的次数，不包括 train-set full evaluation、validation 和 test evaluation。

当前 two-sided ZO 更新一次时：

- x 的零阶估计：每个方向需要 `f(x+delta*u)` 和 `f(x-delta*u)`，共 `2M` 次 forward；
- y 的一阶梯度：当前代码 `grad_y_g(...)` 另做一次 forward + backward；
- 因此每个 ZOFOSGDA optimization step 约为：

```text
train_forward_per_ZO_step = 2M + 1
train_backward_per_ZO_step = 1
```

如果一阶 TTGDA/TTSGDA 在同一个 loss graph 上同时取得 x/y 梯度，则每步约为：

```text
train_forward_per_FO_step = 1
train_backward_per_FO_step = 1
```

在统一总预算 `Q` 下：

```text
ZO steps = floor(Q / (2M + 1))
FO steps = Q
```

例如 `M=5`、`Q=100000`：

```text
ZO 最多运行 floor(100000 / 11) = 9090 steps
FO 最多运行 100000 steps
```

这正是 reviewer 要求按 query 而不是 iteration 比较的含义：不能让所有方法跑相同 step/epoch 后再声称 ZO 收敛更快。

### 3.2 同时保留的辅助口径

CSV 还应记录：

```text
global_step
cum_train_forward_queries
cum_x_function_queries
cum_train_backward_calls
cum_train_sample_forwards
cum_train_sample_backwards
train_wall_time_sec
eval_forward_queries
eval_wall_time_sec
```

其中：

```text
cum_train_sample_forwards = 每次 forward 实际处理的样本数之和
```

因为 batch size 固定为 128，forward-call 曲线是主图，sample-forward 曲线用于审计。

### 3.3 一阶和零阶不是完全等价的 oracle

一次 FO forward+backward 与一次纯 function forward 的计算成本并不相同。因此 rebuttal 应同时给：

1. Test AUC vs training forward queries；
2. Test AUC vs wall-clock；
3. forward / backward 次数表。

正文按老师要求使用 forward 次数对齐，但不能省略 backward 和时间，否则仍可能被质疑隐藏反传成本。

## 4. 开跑前必须修改的代码

### 4.1 增加统一预算和计数器

三个训练入口都增加：

```text
--max_train_forward_queries <Q>
--eval_every_forward_queries <K>
```

不要再只用 `epochs` 控制正式实验。训练循环应在下一步会超过预算前停止，并把最后一个不超过预算的 checkpoint 写入日志。

### 4.2 增加 x/y 动量独立开关

当前 ZOFOSGDA 更新：

```python
w  = (1 - beta_t)  * w  + beta_t  * w_hat
vy = (1 - alpha_t) * vy + alpha_t * vy_hat
```

新增：

```text
--x_momentum {on,off}
--y_momentum {on,off}
```

关闭时必须直接更新：

```python
if x_momentum == "off":
    w = w_hat

if y_momentum == "off":
    vy = vy_hat
```

不能用 `c1=0/c2=0`：这会冻结旧估计，不是去掉动量。也不建议用一个固定 `c` 试图让 `alpha_t/beta_t=1`，因为 `eta_t` 随时间变化。

### 4.3 修正 TTGDA 的 batch size 问题

`26.1.29.ZOFOGDA_AUC/ldx_ttgda_auc.py` 当前强制：

```python
N = len(train_ds)
train_loader = DataLoader(train_ds, batch_size=N, ...)
```

所以命令行 `--batch_size 128` 实际只影响 test loader，训练是 full batch。这与论文“所有方法 batch size 128”不一致，也使 sample-query 无法公平比较。

开跑前必须二选一：

- 如果论文实验本意是 minibatch 128：修正 TTGDA 并全部重跑；
- 如果算法定义要求 full batch：论文中明确披露，并主要使用 sample-forward 与 wall-clock 对齐。

按老师“共同条件”理解，建议本次统一训练 batch size 128。

### 4.4 固定评估时点

不同 `M` 每一步消耗的 query 不同，不能按相同步数评估。建议公共评估点：

```text
Q_checkpoints = {0.05Q, 0.10Q, 0.20Q, 0.40Q, 0.60Q, 0.80Q, Q}
```

方法跨过 checkpoint 时，记录第一个达到或超过该点的模型，并同时记录真实 query 数；最终表使用不超过 `Q` 的最后结果。

### 4.5 ZO direction 与论文统一

当前代码使用归一化 Gaussian，即 unit sphere：

```python
u = torch.randn(d)
u = u / ||u||
```

论文 Eq. (7) 写 unit ball。实验实现建议保持 sphere，并在 config 中记录：

```text
direction_distribution = sphere
```

论文公式和 Lemma 5.4 由理论负责人同步修正。该问题不是靠实验能证明正确的。

## 5. 正式实验矩阵

### E1. `M in {1,5,10,15}` 的影响

目的：回答方向数增加是否降低估计方差、提高性能，以及在相同 forward 预算下是否真的更划算。

固定：

```text
M in {1,5,10,15}
delta = 1e-4
estimator = two-sided
x_momentum = on
y_momentum = on
batch_size = 128
seeds = {14,42,81}
datasets = {svmguide1, spambase, a9a}
```

必须使用相同 `max_train_forward_queries=Q`。`Q` 应先通过 smoke test 确定，并对同一数据集的所有方法固定；命令模板中的 `100000` 只是示例，不是最终论文预算。不同 `M` 的允许步数自动变为：

```text
T_M = floor(Q / (2M+1))
```

报告：

- Test AUC vs forward queries；
- Test AUC vs wall-clock；
- fixed-Q 最终 Test AUC mean ± std；
- 达到指定 AUC 阈值所需 forward queries；
- 可选：固定 checkpoint 重复采样 estimator，报告 variance proxy。

运行数：

```text
3 datasets x 4 M values x 3 seeds = 36 runs
```

注意：原计划里的 `q={1,5,10,20}` 已改成老师明确要求的 `M={1,5,10,15}`。维度扩展实验降为非必要附加项，不占用 rebuttal 主预算。

### E2. 零阶扰动消融

“零阶扰动”可能指扰动半径，也可能指扰动估计形式。建议分两层，先完成 E2-A；若老师确认只需要 `delta`，E2-B 可放附录或不跑。

#### E2-A. 扰动半径 `delta`

固定 `M=5`，比较：

```text
delta in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
estimator = two-sided
```

其余条件与 E1 一致，仍使用相同总 forward 预算。

报告：

- final/best-validation Test AUC；
- train objective；
- gradient-estimator norm/variance proxy；
- NaN、爆炸、过小差分 cancellation 等数值稳定性；
- AUC vs log(delta) 折线图。

运行数：

```text
3 datasets x 5 deltas x 3 seeds = 45 runs
```

其中 `delta=1e-3, M=5, Full momentum` 可以复用 E1 的对应结果，不重复跑。

#### E2-B. 扰动形式

建议比较：

| Variant | x 更新 | 每步 x-forward |
|---|---|---:|
| Two-sided ZO | `(d/(2delta)) [f(x+delta*u)-f(x-delta*u)]u` | `2M` |
| One-sided ZO | `(d/delta) [f(x+delta*u)-f(x)]u`，同一步共享 base `f(x)` | `M+1` |
| Direct FO-x | autograd 直接取得 x 梯度/次梯度 | 与实现图复用有关，通常 1 |

计入 y-gradient 后，预计总 forward：

```text
two-sided: 2M + 1
one-sided: M + 2
FO-x/FO-y joint graph: 1
```

必须在代码中按真实调用计数，不要只根据公式硬编码。如果 base forward 未共享，one-sided 会变成 `2M`，必须在日志中反映。

E2-B 不是要证明 one-sided 一定更好，而是回答性能来自零阶扰动设计还是仅来自其他训练技巧。

### E3. 分别去掉不同动量

四个必要版本：

| Variant | x EMA | y EMA | 关闭后的更新 |
|---|---:|---:|---|
| Full ZOFOSGDA | 开 | 开 | EMA |
| No-x-momentum | 关 | 开 | `w=w_hat` |
| No-y-momentum | 开 | 关 | `vy=vy_hat` |
| No-momentum | 关 | 关 | 两侧都直接使用当前估计 |

固定：

```text
M = 5
delta = 1e-4
estimator = two-sided
batch_size = 128
seeds = {14,42,81}
datasets = {svmguide1, spambase, a9a}
max_train_forward_queries = Q
```

四个版本每步 forward 成本相同，因此 query 曲线可直接比较，但仍必须由计数器停止，而不是跑相同 epoch。

报告：

- Test AUC vs forward queries；
- Test AUC vs wall-clock；
- fixed-Q mean ± std；
- `||w_hat-w||`、`|vy_hat-vy|` 或 estimator variance proxy。

运行数：

```text
3 datasets x 4 variants x 3 seeds = 36 runs
```

Full ZOFOSGDA 可复用 E1 中 `M=5` 的结果。

### E4. 一阶 baseline 全部重跑

必须重跑：

- TTGDA；
- TTSGDA。

推荐附加：

- EMA-TTSGDA 或 EMA-TTGDA，用于回答“只有本文方法用了 momentum”的公平性质疑。

统一条件：

```text
batch_size = 128
seeds = {14,42,81}
datasets = {svmguide1, spambase, a9a}
max_train_forward_queries = Q
```

一阶方法不能继续使用旧的 `epochs=500` 结果作为主对照，因为：

- 旧日志没有 forward 计数；
- TTGDA 当前可能实际使用 full batch；
- 旧结果的评估时点与 ZO 的 query checkpoint 不一致。

运行数：

```text
3 datasets x 2 FO methods x 3 seeds = 18 runs
```

EMA baseline 若加入，再增加 9 runs。

### E5. 论文 Table 3 的补充结果

老师指定的三组主实验完成后，再决定是否扩展到论文六个 AUC 数据集。当前旧结果仍可用于排查超参数，但不能替代同-query重跑。

旧结果缺口：

- A9a：TTSGDA 缺 seed 42；ZOFOSGDA 缺 seed 81；
- Ijcnn1：TTGDA 缺 seed 81；TTSGDA 缺 seed 14、81；ZOFOSGDA 缺 seed 81。

由于这次明确要求一阶也重跑，不建议只机械补这 6 个旧 run；应先在三个主数据集完成全套 query-fair 结果，再按时间扩展六数据集。

论文 claim 同时需要降调：

- German：TTGDA 均值高于 ZOFOSGDA；
- Spambase：ZOFOSGDA 与 TTSGDA 在报告精度下打平；
- 不再使用“consistently outperforms”描述所有数据集。

## 6. 推荐运行顺序

### P0：代码准备与审计

1. 实现真实 forward/backward/sample 计数器；
2. 实现 `max_train_forward_queries` 停止条件；
3. 实现 x/y momentum 独立开关；
4. 修正 TTGDA batch size；
5. 对每个方法跑 `1 dataset x 1 seed x 2 epochs` smoke test；
6. 手算并验证每步 query 增量。

### P1：老师明确要求的主实验

1. E1：`M={1,5,10,15}`；
2. E3：四种 momentum variant；
3. E4：TTGDA/TTSGDA 全部重跑；
4. E2-A：`delta` 消融。

### P2：确认“零阶扰动”含义后执行

1. E2-B：two-sided / one-sided / Direct FO-x；
2. EMA 一阶 baseline；
3. 扩展到论文全部六个 AUC 数据集；
4. Robust Classification 对应实验。

当前不优先重跑 LLM meta-training，因为它没有直接回答老师列出的三项要求，GPU 成本也更高。

## 7. GPU 分配建议

若有 4 张 GPU，第一阶段：

| GPU | 任务 |
|---|---|
| 0 | Svmguide1：M sweep + momentum |
| 1 | Spambase：M sweep + momentum |
| 2 | A9a：M sweep + momentum |
| 3 | TTGDA/TTSGDA 重跑 + smoke tests |

第二阶段：

| GPU | 任务 |
|---|---|
| 0 | delta sweep on Svmguide1 |
| 1 | delta sweep on Spambase |
| 2 | delta sweep on A9a |
| 3 | perturbation-form / EMA baseline |

不要用 GPU 编号写死在公共脚本里；由启动脚本或调度器传入 `--device`。

## 8. 命令模板

以下新参数需要先在代码中实现，模板目前不能直接运行。

### 8.1 M sweep

```bash
cd /hy-tmp/ZOFOGDA_Nips26/26.1.29.ZOFOGDA_AUC

for M in 1 5 10 15; do
  python ldx_zofogda_auc.py \
    --dataset spambase \
    --data_dir ./data \
    --log_dir "./rebuttal_results/E1_M/spambase/M${M}/seed42" \
    --batch_size 128 --hidden 128 \
    --eta_k 10 --eta_m 1000 --c1 3 --c2 3 \
    --gamma 0.01 --lam 0.01 \
    --zo_delta 1e-3 --zo_q "${M}" \
    --x_momentum on --y_momentum on \
    --max_train_forward_queries 100000 \
    --eval_every_forward_queries 5000 \
    --seed 42 --device cuda:0
 done
```

### 8.2 Momentum 消融

```text
Full:    --x_momentum on  --y_momentum on
No-x:    --x_momentum off --y_momentum on
No-y:    --x_momentum on  --y_momentum off
No-both: --x_momentum off --y_momentum off
```

共同使用：

```text
--zo_q 5 --zo_delta 1e-3 --max_train_forward_queries 100000
```

### 8.3 Delta sweep

```bash
for delta in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  python ldx_zofogda_auc.py \
    --dataset spambase \
    --data_dir ./data \
    --log_dir "./rebuttal_results/E2_delta/spambase/delta${delta}/seed42" \
    --batch_size 128 --hidden 128 \
    --zo_delta "${delta}" --zo_q 5 \
    --x_momentum on --y_momentum on \
    --max_train_forward_queries 100000 \
    --eval_every_forward_queries 5000 \
    --seed 42 --device cuda:0
 done
```

### 8.4 一阶 baseline

TTGDA/TTSGDA 也必须支持：

```text
--max_train_forward_queries 100000
--eval_every_forward_queries 5000
--batch_size 128
```

不要把 `epochs` 当作不同方法之间的公平预算。

## 9. 结果目录

```text
rebuttal_results/
  E1_M/
    <dataset>/M_<M>/seed_<seed>/
  E2_perturbation/
    <dataset>/delta_<delta>/<estimator>/seed_<seed>/
  E3_momentum/
    <dataset>/<full|no_x|no_y|no_both>/seed_<seed>/
  E4_first_order/
    <dataset>/<ttgda|ttsgda|ema_ttsgda>/seed_<seed>/
```

每个 run 至少保存：

```text
metrics.csv
config.json
stdout.log
environment.txt
```

`metrics.csv` 必须包含：

```text
global_step
train_loss
test_auc
cum_train_forward_queries
cum_x_function_queries
cum_train_backward_calls
cum_train_sample_forwards
train_wall_time_sec
eval_forward_queries
```

## 10. 最终交付的表和图

### 主图

1. `M={1,5,10,15}`：Test AUC vs forward queries；
2. Full / No-x / No-y / No-both：Test AUC vs forward queries；
3. ZOFOSGDA 与重跑 TTGDA/TTSGDA：Test AUC vs forward queries；
4. 同样三组比较的 Test AUC vs wall-clock；
5. delta 敏感性图；
6. 若执行 E2-B：不同扰动形式的 fixed-query 曲线。

### 主表

1. 相同 forward budget `Q` 下的最终 AUC mean ± std；
2. 达到目标 AUC 所需 forward queries 和 wall-clock；
3. 每个方法的 forward/backward/sample-forward 总数；
4. Momentum 四版本消融；
5. M sweep 和 delta sweep。

### 结果解释模板

结果出来后再填数字：

```text
Under the same budget of Q training forward evaluations, M=1/5/10/15 obtains
[...]. Increasing M reduces the number of optimization steps from [...] to [...],
so its variance reduction is evaluated under the actual query trade-off.
```

```text
Removing x-EMA / y-EMA changes the fixed-query AUC from [...] to [...] / [...],
while removing both gives [...]. The no-momentum variants directly use the current
gradient estimates rather than freezing historical estimates.
```

```text
At the same number of training forward evaluations, ZOFOSGDA reaches [...],
compared with [...] for TTGDA and [...] for TTSGDA. We additionally report
backward calls and wall-clock because a first-order oracle is not computationally
identical to a function-value query.
```

## 11. 开跑检查清单

- [ ] 向老师确认 `M` 是随机方向数而不是 minibatch size；
- [ ] 向老师确认“零阶扰动”是否仅指 `delta`；
- [ ] 三个方法统一 batch size 128；
- [ ] TTGDA 不再静默强制 full batch；
- [ ] 新增真实 forward/backward/sample 计数；
- [ ] 新增 fixed-forward-budget 停止条件；
- [ ] evaluation query 与 training query 分开；
- [ ] 新增 x/y momentum 独立开关；
- [ ] 验证 No-x/No-y 使用直接当前梯度，而不是 `c=0`；
- [ ] 验证 two-sided 每步总 forward 约为 `2M+1`；
- [ ] 验证一阶共同反传每步 forward 约为 1；
- [ ] M 严格使用 `{1,5,10,15}`；
- [ ] seeds 使用 `{14,42,81}`；
- [ ] 超参数只用 validation 选择；
- [ ] 画 query 和 wall-clock 两套曲线；
- [ ] 汇总 mean ± std，并核对论文 claim。

## 12. 实验之外的并行理论事项

以下问题仍需理论负责人处理，不能靠新增实验覆盖：

1. Eq. (7) 的 ball/sphere 与 Lemma 5.4 无偏性；
2. Lemma B.1 中 `argmax E[f]` 与单次 perturbation 后 `argmax f` 的等式；
3. Eqs. (40)、(44)、(48) 及主定理对 Lemma B.1 的依赖；
4. “matches best-known rate”的过强表述；
5. 与 arXiv:2604.21371 的 stationarity、assumption 和 rate 对比；
6. GFM 引用编号、ZOFOSGDA/ZOFOGDA 命名和 Limitations checklist。

## 12. 已落地的自动化实验系统

完整实现位于 `rebuttal/`，使用说明见 `rebuttal/README.md`。当前矩阵包含 1,082 个去重后的 run，按六个 phase 排序并均衡分配至 8 张 GPU。结果 Excel、原始曲线和 mean±std 图片由聚合脚本持续刷新。主默认值已对齐现有论文代码：AUC 为 `gamma=lambda=0.01`，稳健分类为 `gamma=lambda=1`，默认 `delta=1e-4`。
