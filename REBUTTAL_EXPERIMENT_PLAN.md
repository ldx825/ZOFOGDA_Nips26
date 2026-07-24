# NeurIPS 2026 Rebuttal 实验梳理与执行计划

> 论文：*Hybrid Zeroth- and First-Order Method for Stochastic Nonsmooth Minimax Optimization*  
> Submission：10118  
> 项目目录：`/hy-tmp/LiuDongxu`  
> 整理日期：2026-07-24

## 1. 先说结论

本次 rebuttal 最需要补的不是更多“最终 AUC/Accuracy 数字”，而是以下四组能直接回答 reviewer 的实验：

1. **公平的 oracle/query-cost 对比**：横轴不能只用 epoch/iteration；至少补 cumulative function evaluations、processed samples 和 wall-clock time。
2. **x-momentum 与 y-momentum 独立消融**：必须有 Full、No-x、No-y、No-both 四个版本，不能只调 `c1=c2`。
3. **查询方向数 `q` 与维度 `d1` 的关系**：回答固定 `q=5` 在高维是否退化，以及 `q` 增大后方差/性能/成本如何变化。
4. **平滑参数 `delta` 敏感性**：说明实践中的 `delta` 是否必须跟目标精度绑定，以及结果对 `delta` 是否稳定。

建议的 rebuttal 最小闭环为：

- 先修正公平性与日志统计；
- 在 Svmguide1、Spambase、A9a（低/中/高维）上完成四种 momentum 消融；
- 在同三组数据上完成 `q`/维度实验和 `delta` 敏感性；
- 对论文 Table 3 的六个数据集补齐缺失 seed，并用统一实现重画 query-cost/time 曲线；
- Robust Classification 至少在 Svmguide1 上复核一次；若原始四个数据集和配置无法及时恢复，不要声称当前仓库能完整复现 Table 2。

理论错误（unit ball/sphere、Lemma B.1、复杂度比较）不是实验能够补救的，应由理论负责人并行处理。实验侧只能确保 estimator 实现、成本口径和论文字句一致。

---

## 2. 项目结构与论文对应关系

### 2.1 主要目录

| 路径 | 内容 | 与本次 rebuttal 的关系 |
|---|---|---|
| `26.1.29.ZOFOGDA_AUC/` | AUC maximization：ZOFOGDA、TTGDA、TTSGDA、数据、CSV 结果 | **主战场，优先使用** |
| `ZOFOGDA_AUC/` | 较早版本的 AUC 代码和少量结果 | 仅作参考，不建议混用 |
| `ZOFOGDA_GAN/` | GAN 代码与早期 robust classification 脚本混在一起 | Robust classification 可参考，但需整理后再跑 |
| `26.1.24.ZOFO_AdaGDA-main/` | OPT-1.3B / PEFT / bilevel meta-training | 对应附录 A.5；评审未要求，**非优先项** |
| `10118_Hybrid_Zeroth_and_First_.pdf` | 投稿论文 | 实验设定和原始表格依据 |

### 2.2 论文中的已有实验

- Robust classification：
  - 数据集：Australian、Svmguide1、Breast-cancer、Ionosphere_scale；
  - 指标：test accuracy；
  - baseline：TTGDA、TTSGDA；
  - 论文称三组随机种子、batch size 128、ZO 查询方向数/批量参数为 5。
- AUC maximization：
  - 主表数据集：Climate、German、Svmguide1、Spambase、A9a、Ijcnn1；
  - 附录还描述了 Covtype、Connect、Acoustic；
  - 指标：test AUC；
  - baseline：TTGDA、TTSGDA；
  - 三组随机种子。
- 已有消融：
  - `c1=c2 in {1,2,3}`；
  - batch size in `{64,128,256}`。

现有 `c1/c2 + batch size` 消融不能替代 reviewer 要求的 momentum 消融。改变 EMA 系数只是在改变动量强弱，并不等价于严格关闭 x 或 y 的 EMA。

---

## 3. 评审意见到实验的映射

| 来源 | 评审关切 | 是否要实验 | 对应实验 |
|---|---|---:|---|
| 8HnE / Meta Review | 只按 iteration 比较不公平，ZO 一步有多个 function evaluations | 是，最高优先级 | E1：query-cost / sample-cost / wall-clock 公平比较 |
| 8HnE | 分别去掉 x-momentum 和 y-momentum | 是，最高优先级 | E2：四版本 momentum 消融 |
| M6Go | ZOFOSGDA 有 momentum，baseline 没有 | 是 | E2；另加 EMA-TTSGDA/EMA-TTGDA wrapper |
| M6Go | 实践 `q=5`，理论查询量随维度增长；高维且 q 固定是否退化 | 是 | E3：`q x d1` 实验 |
| M6Go | `delta` 是否和目标精度绑定；对 `delta` 是否敏感 | 是 | E4：平滑参数敏感性 |
| 8HnE | German 上 TTGDA 更高，Spambase 只是打平，论文 claim 过强 | 可用已有结果回答，但建议复核 | E5：补齐 seed、统一报告 mean/std 和显著性 |
| dYQd | GGSP/Clarke 的直观解释、最优维度依赖、limitations | 否 | 文字回复/示例，不是实验 |
| C8sk | 与 concurrent work 的 stationarity/rate 比较 | 否 | 理论对照表，不是实验 |
| 8HnE / Meta Review | estimator unbiasedness、Lemma B.1 | 否，实验无法修复 | 理论证明必须单独修订 |

---

## 4. 开跑前必须处理的实现问题

这些问题不先处理，跑出来的结果不适合放进 rebuttal。

### 4.1 TTGDA 的 batch size 与论文描述不一致

文件：`26.1.29.ZOFOGDA_AUC/ldx_ttgda_auc.py`

当前实现使用：

```python
N = len(train_ds)
train_loader = DataLoader(train_ds, batch_size=N, ...)
```

因此命令行的 `--batch_size 128` 只影响 test loader，训练实际上是 full batch。论文却写“all methods batch size 128”。这会同时破坏：

- 每 epoch 成本公平性；
- stochastic oracle 次数对齐；
- wall-clock 对比；
- TTGDA 与 TTSGDA 的算法区别解释。

**要求**：明确 TTGDA 的原始算法设定。如果 TTGDA 本应 full-batch，就在论文和图表中明确写出，并按 sample-oracle 成本归一化；如果本应 batch 128，则修改实现并重新跑。

### 4.2 当前日志没有 query/oracle 计数

当前 CSV 主要只有：

```text
epoch, train_loss, test_auc, ...
```

至少增加：

```text
global_step
cum_x_function_evals
cum_x_fo_evals
cum_y_fo_evals
cum_train_samples
train_wall_time_sec
eval_wall_time_sec
```

训练成本不要混入周期性 full-train evaluation 和 test evaluation。建议分别记录：

- `train_wall_time_sec`：只累计参数更新；
- `total_wall_time_sec`：包含评估；
- oracle 计数：只统计训练更新所使用的 oracle。

### 4.3 现有 CLI 不能真正关闭某一侧 momentum

当前更新为：

```python
w  = (1 - beta_t)  * w  + beta_t  * w_hat
vy = (1 - alpha_t) * vy + alpha_t * vy_hat
```

不能简单用 `c1=0` 或 `c2=0` 表示“无 momentum”：

- 设为 0 会冻结历史估计；
- `beta_t=1` 才等于无 x-EMA，但当前 `beta_t=c2*eta_t` 随时间变化，无法通过一个固定 `c2` 保证恒为 1。

建议增加明确开关：

```text
--x_momentum {on,off}
--y_momentum {on,off}
```

关闭时使用：

```python
w = w_hat
vy = vy_hat
```

### 4.4 ZO estimator 的代码与论文文字需要统一

论文 Eq. (7) 写的是 direction 从 unit ball 采样，但无偏公式通常对应 unit sphere。当前 AUC 代码：

```python
u = torch.randn(d)
u = u / ||u||
```

实际采的是 sphere，这与 reviewer 建议一致。应：

- 保留 sphere 实现；
- 在日志中记录 `direction_distribution=sphere`；
- 论文公式和 Lemma 5.4 统一修正；
- 不要把该问题写成“实验验证了无偏性”，它首先是数学定义问题。

### 4.5 理论中的 `M` 与代码中的 `q` 不应混用

论文 Eq. (7) 的 `M` 表示 batch 中样本/方向的平均数，方差界也写成随 `M` 衰减；当前代码则：

- batch size 默认为 128；
- 对同一个 minibatch 采 `q=5` 个方向；
- 每个方向做正负两次 batched forward。

因此必须在论文和回复中区分：

- `B`：minibatch size；
- `q`：每次 x-update 的随机方向数；
- batched function evaluation 次数：`2q`；
- sample-function evaluations：`2qB`。

---

## 5. 实验设计

### E0. 统一复现实验协议

所有新增实验统一：

- seeds：`{14, 42, 81}`；
- preprocessing：同一个 train/test split，同一个 standardization；
- architecture：同一个 one-hidden-layer ReLU MLP；
- 默认 hidden size：128；
- 默认 batch size：128；
- 默认 `q=5`；
- 默认 `delta=1e-3`（与现有主结果文件一致）；
- 每项报告 mean ± std，最好同时报告 95% bootstrap CI；
- 超参数选择只能使用验证集，不能按 test AUC 挑最好配置；
- 同一比较中的各方法使用同一训练预算和相同 seed；
- 保存完整 config、git/代码快照标识、GPU 型号、PyTorch/CUDA 版本。

主数据集建议按维度分层：

| 数据集 | 原始特征维度 | hidden=128 时约 `d1` | 角色 |
|---|---:|---:|---|
| Svmguide1 | 4 | 771 | 低维 |
| Spambase | 56（论文表写 57，需核对索引口径） | 7,427 | 中维 |
| A9a | 123 | 16,003 | 高维 |
| Connect | 125（论文表写 126，需核对索引口径） | 16,259 | 额外高维验证 |

这里的 `d1` 按当前 AUC primal 参数计算：

```text
d1 = hidden * (D + 2) + 3
```

注意：本地 LIBSVM 文件的最大特征索引与论文 Table 5 在 Climate/German/Spambase/Connect 上存在 1 的差异，应确认是否来自 0/1-based 统计或数据预处理版本。

### E1. 公平的 query/oracle-cost 对比

**目的**：直接回答“按 iteration 比较不公平”。

方法：

- TTGDA；
- TTSGDA；
- ZOFOSGDA；
- 推荐增加 EMA-TTSGDA（或 EMA-TTGDA），回应“baseline 没有 momentum”。

数据集：

- 最小版：Svmguide1、Spambase、A9a；
- 完整版：论文 Table 3 的全部六个数据集。

每个方法至少画三种横轴：

1. `cumulative processed training samples`；
2. `cumulative x-oracle cost`；
3. `training wall-clock seconds`。

ZO 的成本同时报告两种定义，避免口径争论：

```text
batched x-function evaluations per update = 2q
sample-function evaluations per update = 2qB
```

FO baseline 则报告：

```text
x first-order oracle calls per update = 1
sample-gradient evaluations per update = B
```

不能把一次 FO backward 武断地等同为一次纯 function forward。最稳妥的呈现是把 ZO function queries、FO gradient queries 和 wall-clock 分列，并说明它们是不同 oracle。

主要产出：

- Test AUC vs sample-function/sample-gradient evaluations；
- Test AUC vs wall-clock；
- 达到某阈值 AUC 所需成本表；
- 最终 AUC mean ± std 表。

公平性检查：

- 相同 batch size；
- 相同模型初始化；
- 相同数据顺序（可保存 sampler seed）；
- 相同评估频率；
- evaluation forward 不计入 training oracle；
- wall-clock 预热 GPU 后再计时，并同步 CUDA。

### E2. x/y momentum 独立消融

**目的**：判断提升来自 hybrid oracle，还是来自某一侧 EMA。

四个必要版本：

| Variant | x EMA | y EMA |
|---|---:|---:|
| Full ZOFOSGDA | 开 | 开 |
| No-x-momentum | 关：`w=w_hat` | 开 |
| No-y-momentum | 开 | 关：`vy=vy_hat` |
| No-momentum | 关 | 关 |

建议数据集：

- Svmguide1；
- Spambase；
- A9a。

固定：

- `q=5`；
- `delta=1e-3`；
- batch size 128；
- seeds `{14,42,81}`；
- 同一 query budget，而不是只用相同 epoch。

报告：

- 最终 test AUC；
- best validation checkpoint 对应的 test AUC；
- AUC vs query；
- `||w_hat-w||`、`|vy_hat-vy|` 或 gradient-estimator variance proxy；
- 三个 seed 的 mean ± std/CI。

最小运行数：

```text
3 datasets x 4 variants x 3 seeds = 36 runs
```

若时间非常紧，可以先只跑 Spambase 和 A9a，共 24 runs；但至少保留一个高维数据集。

### E3. `q` 与维度 `d1` 的实验

**目的**：回答固定 `q=5` 在更高维是否退化，以及增大 `q` 的收益是否值得额外查询成本。

建议分成两个子实验。

#### E3-A：跨数据集、固定模型宽度

- datasets：Svmguide1、Spambase、A9a；
- hidden：128；
- `q in {1, 5, 10, 20}`；
- `delta=1e-3`；
- seeds `{14,42,81}`；
- 固定总 function-query budget。

这一实验易做，但数据集变化会引入任务难度混杂，因此不能单独作为“维度规律”的严格证据。

#### E3-B：同一数据集内改变参数维度

推荐 Svmguide1：

| hidden | 约 `d1` | `ceil(sqrt(d1))` |
|---:|---:|---:|
| 16 | 99 | 10 |
| 64 | 387 | 20 |
| 128 | 771 | 28 |
| 256 | 1,539 | 40 |

每个维度比较：

- 固定 `q=5`；
- `q=ceil(sqrt(d1))`；
- 可加 `q=1` 作为低成本端点。

必须使用**固定 query budget**：当 `q` 增大时，允许的 optimization steps 按 `1/q` 缩短。否则大 `q` 只是花了更多计算，不能证明 query efficiency。

报告：

- AUC vs total function queries；
- AUC vs wall-clock；
- ZO estimator 的方差代理：

```text
mean_t ||g_hat_t - mean_r(g_hat_t^(r))||^2
```

可在固定 checkpoint 上重复采 20 次 estimator，仅用于测方差，不参与训练，并将这些诊断 queries 与训练 queries 分开统计。

### E4. `delta` 敏感性

**目的**：回答 smoothing radius 是否敏感、实践中是否必须随目标精度变化。

推荐：

```text
delta in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
q = 5
datasets = {Svmguide1, Spambase, A9a}
seeds = {14, 42, 81}
```

固定 query budget，报告：

- final/best-validation test AUC；
- train objective；
- ZO estimator norm 与 variance proxy；
- 数值稳定性（NaN、爆炸、过小差分导致的 cancellation）。

建议画：

- x 轴为 log-scale `delta`，y 轴为最终 AUC；
- 每个数据集一条曲线，误差条为 std/CI；
- 另给 `delta` 与 estimator variance 的关系图。

解释边界：

- 实验只能说明有限范围内的鲁棒性；
- 理论中设 `delta=O(epsilon)` 与实践固定 `delta` 不矛盾，但需要明确理论 target accuracy 和经验调参不是同一层面的声明；
- 不要根据 test set 选择 `delta`。

### E5. 补齐 Table 3 与克制论文 claim

当前 `26.1.29.ZOFOGDA_AUC/result/` 中，论文六个 AUC 数据集的大部分三 seed 结果已经存在，但按主配置检查仍有缺口：

- A9a：
  - TTSGDA 缺 seed 42；
  - ZOFOSGDA 缺 seed 81。
- Ijcnn1：
  - TTGDA 缺 seed 81；
  - TTSGDA 缺 seed 14、81；
  - ZOFOSGDA 缺 seed 81。

至少需补 6 个 run，之后统一汇总。还需核对：

- Climate 当前有 500/1000/5000 epochs 混杂，最终表使用哪个预算；
- German、Svmguide1、Spambase 有多组调参结果，必须固定 validation-based 选择规则；
- A9a 有一条 `delta=1e-4, epochs=200` 的旧结果，不要与 `delta=1e-3, epochs=500` 混入同一组统计。

论文文字应改为：

- German：承认 TTGDA 的均值高于 ZOFOSGDA；
- Spambase：写“matches TTSGDA at reported precision”，不要写 consistently outperforms；
- 使用 paired seeds 后可补置信区间，但不要仅凭均值做显著性结论。

### E6. Robust Classification 复核

早期 robust classification 实现在：

```text
ZOFOGDA_GAN/zofoGDA.py
ZOFOGDA_GAN/TTSGDA.py
ZOFOGDA_GAN/TTGDA .py
```

目前仓库中只看到 Svmguide1/Splice 数据和少量旧 CSV，没有发现论文 Table 2 所需的 Australian、Breast-cancer、Ionosphere_scale 数据文件与完整运行脚本。因此：

1. 先向合作者确认论文 Table 2 的原始代码、数据 split、超参数和 seed；
2. 不要直接把 AUC 脚本当作 robust classification 脚本；
3. 如果原始配置能恢复，至少在四个数据集上补 E1 和 E2；
4. 如果不能及时恢复，rebuttal 实验重点放 AUC，并如实说明新实验在哪个任务上完成。

### E7. EMA baseline（推荐）

为回答 M6Go 的“baseline 没有 momentum”，在 TTSGDA 或 TTGDA 外包一层与 ZOFOSGDA 同形式的 EMA：

```text
gx_ema = (1-beta) * gx_ema + beta * gx
gy_ema = (1-alpha) * gy_ema + alpha * gy
```

比较：

- 原始 TTSGDA；
- EMA-TTSGDA；
- ZOFOSGDA No-momentum；
- Full ZOFOSGDA。

这样能把两个因素拆开：

- oracle 类型：FO vs ZO/FO hybrid；
- momentum：无 vs EMA。

如果 rebuttal 时间有限，只在 Spambase、A9a 上跑即可。

---

## 6. 运行优先级与规模

### P0：必须完成

1. 修正 TTGDA batch/oracle 口径；
2. 增加 query/sample/time 日志；
3. 实现四种 momentum variant；
4. 在 Svmguide1、Spambase、A9a 上跑 E2；
5. 在同三数据集上重画 E1；
6. 补齐 Table 3 缺失的 6 个 seed run。

预计新增/重跑规模：

- Momentum：36 runs；
- 缺失主表：6 runs；
- 若 E1 需要所有方法统一重跑：最多 `6 datasets x 3 methods x 3 seeds = 54 runs`，其中可与其他实验复用 Full ZOFOSGDA 结果。

### P1：强烈建议

1. E3：`q x d1`；
2. E4：`delta`；
3. E7：EMA baseline。

建议先用单 seed 做筛查，再对最终选定矩阵补三 seeds。筛查结果不能直接进入主表。

### P2：有余力再做

- 将 E1/E2 扩展到 Robust Classification 四数据集；
- 将 AUC 扩展到 Connect/Covtype/Acoustic；
- 重跑 LLM meta-training。

LLM 实验没有直接回应当前 reviewer 的核心质疑，GPU 成本又高，当前不建议优先。

---

## 7. 建议的 GPU 排队方式

若有 4 张 GPU：

| GPU | 第一阶段 |
|---|---|
| 0 | Svmguide1 momentum variants |
| 1 | Spambase momentum variants |
| 2 | A9a momentum variants |
| 3 | Table 3 缺失 seeds + smoke tests |

第二阶段：

| GPU | 任务 |
|---|---|
| 0 | `q x d1` |
| 1 | `delta` on Svmguide1 |
| 2 | `delta` on Spambase |
| 3 | `delta` on A9a / EMA baseline |

每个正式任务开跑前先做：

```text
1 dataset x 1 seed x 2 epochs
```

检查 CSV 是否包含 query/time 字段、各 variant 是否真的走不同分支、query 计数是否单调且符合手算。

---

## 8. 命令模板

以下参数中的 momentum/query-budget 开关是建议新增的接口；当前代码尚不支持，不能原样直接跑。

### 8.1 Full / momentum ablation

```bash
cd /hy-tmp/LiuDongxu/26.1.29.ZOFOGDA_AUC

python ldx_zofogda_auc.py \
  --dataset spambase \
  --data_dir ./data \
  --log_dir ./rebuttal_results/E2/spambase/full/seed42 \
  --epochs 500 \
  --batch_size 128 \
  --hidden 128 \
  --eta_k 10 --eta_m 1000 --c1 3 --c2 3 \
  --gamma 0.01 --lam 0.01 \
  --zo_delta 1e-3 --zo_q 5 \
  --x_momentum on --y_momentum on \
  --seed 42 \
  --device cuda:0
```

其余三种只改：

```text
No-x:    --x_momentum off --y_momentum on
No-y:    --x_momentum on  --y_momentum off
No-both: --x_momentum off --y_momentum off
```

### 8.2 `q` sweep

```bash
for q in 1 5 10 20; do
  python ldx_zofogda_auc.py \
    --dataset a9a \
    --data_dir ./data \
    --log_dir "./rebuttal_results/E3/a9a/q${q}/seed42" \
    --batch_size 128 --hidden 128 \
    --zo_delta 1e-3 --zo_q "${q}" \
    --max_x_function_evals 200000 \
    --seed 42 --device cuda:0
done
```

### 8.3 `delta` sweep

```bash
for delta in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  python ldx_zofogda_auc.py \
    --dataset spambase \
    --data_dir ./data \
    --log_dir "./rebuttal_results/E4/spambase/delta${delta}/seed42" \
    --batch_size 128 --hidden 128 \
    --zo_delta "${delta}" --zo_q 5 \
    --max_x_function_evals 200000 \
    --seed 42 --device cuda:0
done
```

---

## 9. 结果目录和命名规范

建议不要继续把所有 CSV 堆进现有 `result/`。使用：

```text
rebuttal_results/
  E1_query_cost/
    <dataset>/<method>/seed_<seed>/
  E2_momentum/
    <dataset>/<variant>/seed_<seed>/
  E3_q_dimension/
    <dataset>/hidden_<h>/q_<q>/seed_<seed>/
  E4_delta/
    <dataset>/delta_<delta>/seed_<seed>/
  E7_ema_baseline/
    <dataset>/<method>/seed_<seed>/
```

每个运行目录至少保存：

```text
metrics.csv
config.json
stdout.log
environment.txt
```

文件名不要再依赖浮点数拼接作为唯一标识；`config.json` 才是事实来源。

---

## 10. 最终需要交给论文负责人的表和图

### 主 rebuttal 图

1. 三个数据集的 Test AUC vs cumulative oracle/sample cost；
2. 三个数据集的 Test AUC vs wall-clock；
3. Momentum 四版本消融图；
4. `q` 在低/中/高维下的 fixed-budget 曲线；
5. `delta` 敏感性折线图。

### 主 rebuttal 表

1. 公平预算下的 final AUC mean ± std；
2. 达到目标 AUC 的 query/time；
3. Full / No-x / No-y / No-both；
4. Table 3 修正版；
5. 若完成：原始 baseline vs EMA baseline。

### 回复中建议使用的实验结论句式

只有结果出来后再填数字：

```text
At an equal budget of [Q] x-function evaluations, ZOFOSGDA reaches [AUC],
compared with [...] for the baselines. We additionally report wall-clock and
sample-oracle costs because zeroth- and first-order oracle calls are not directly
equivalent.
```

```text
Removing x-EMA / y-EMA changes the mean AUC from [...] to [...] / [...],
while removing both gives [...]. This shows that [结论以实际结果为准],
rather than attributing all gains to momentum without evidence.
```

```text
With q fixed at 5, performance [保持稳定/下降] as d1 grows from [...] to [...].
Increasing q reduces the estimator-variance proxy by [...], but under an equal
query budget the gain is [...], revealing the actual cost-quality trade-off.
```

---

## 11. 不属于“跑实验”的并行事项

必须提醒老板/理论负责人：以下问题决定 rebuttal 是否成立，但不能靠实验覆盖。

1. Eq. (7) 的 ball/sphere 与 Lemma 5.4 无偏性；
2. Lemma B.1 中 `argmax E[f]` 与单个 perturbation 后 `argmax f` 的错误等式；
3. Eqs. (40)、(44)、(48) 及主定理对 Lemma B.1 的依赖；
4. “matches best-known rate” 的错误/过强表述；
5. 与 arXiv:2604.21371 的 stationarity、assumption 和 rate 对比；
6. GFM 引用编号错误；
7. ZOFOSGDA / ZOFOGDA 命名不一致；
8. Limitations checklist 中 `[TODO]`；
9. German/Spambase 的实验 claim 降调。

如果理论主结果无法在 rebuttal 时限内修复，实验再多也无法消除 Meta Review 的核心否定意见。实验工作应以“补齐公平性证据、为后续重投留下可信结果”为目标，避免为了赶时间扩大不必要的实验面。

---

## 12. 开跑检查清单

- [ ] 确认理论负责人决定 sphere/ball 的最终版本；
- [ ] 确认 TTGDA 应为 full batch 还是 minibatch；
- [ ] 三个方法的数据 split、standardization、模型完全一致；
- [ ] 新增 x/y momentum 独立开关；
- [ ] 新增 function/FO/sample query 计数；
- [ ] 新增纯训练 wall-clock；
- [ ] 新增 fixed-query-budget 停止条件；
- [ ] 先跑 2-epoch smoke test；
- [ ] 验证 `q=5` 时每个 x-update 增加 `2q=10` 个 batched function evals；
- [ ] 验证 No-x/No-y 分支不是通过 `c=0` 实现；
- [ ] seeds 固定为 14、42、81；
- [ ] 超参数选择不看 test set；
- [ ] 补齐 A9a/Ijcnn1 缺失 seed；
- [ ] 汇总 mean/std/CI；
- [ ] 重画 query、sample、time 三种横轴；
- [ ] 论文 claim 与实际表格逐句核对。

