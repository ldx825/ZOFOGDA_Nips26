#!/usr/bin/env python3
"""Build the comprehensive rebuttal run matrix and eight balanced GPU queues."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REB = ROOT / "rebuttal"
CONFIG_DIR = REB / "configs" / "runs"
QUEUE_DIR = REB / "queues"
RESULT_DIR = REB / "results"
TABLE_DIR = REB / "tables"

SEEDS = [14, 42, 81]
AUC_DATASETS = ["climate", "german", "svmguide1", "spambase", "a9a", "ijcnn1"]
ROBUST_DATASETS = ["australian", "svmguide1", "breast-cancer", "ionosphere_scale"]
ALL = [("auc", x) for x in AUC_DATASETS] + [("robust", x) for x in ROBUST_DATASETS]
REPRESENTATIVE = [("auc", "svmguide1"), ("auc", "spambase"), ("auc", "a9a"), ("robust", "svmguide1")]
LEGACY_ABLATION = [("auc", x) for x in ["climate", "german", "svmguide1", "spambase"]] + [("robust", x) for x in ["australian", "svmguide1", "breast-cancer", "ionosphere_scale"]]
BUDGET = {
    "climate": 30000, "german": 30000, "australian": 30000,
    "breast-cancer": 30000, "ionosphere_scale": 30000,
    "svmguide1": 50000, "spambase": 75000,
    "a9a": 100000, "ijcnn1": 100000,
}

runs = {}
tags = defaultdict(set)


def defaults(task, dataset, seed):
    q = BUDGET[dataset]
    base = dict(
        phase=99, task=task, dataset=dataset, method="zofo", estimator="two_sided",
        seed=seed, data_seed=0, hidden=128, batch_size=128, standardize=True,
        max_train_forward_queries=q, eval_every_forward_queries=max(1000, q // 20),
        eval_train_batches=8, M=5, delta=1e-4, direction_distribution="sphere",
        x_momentum=True, y_momentum=True, eta_k=10.0, eta_m=1000.0,
        c1=3.0, c2=3.0, gamma=0.01, lam=0.01,
        lr_x=0.01, lr_y=0.0001, dual_clip=10.0, theta1=0.01,
        save_model=False, notes="",
    )
    if task == "robust":
        base.update(
            gamma=1.0, lam=1.0, lr_x=0.1, lr_y=1.0, theta1=0.001
        )
    return base


def key_for(cfg):
    ignored = {"run_id", "output_dir", "experiment_groups", "notes", "phase"}
    payload = {k: cfg[k] for k in sorted(cfg) if k not in ignored}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def add(group, cfg, phase, note=""):
    cfg = dict(cfg)
    cfg["phase"] = min(int(cfg.get("phase", phase)), phase)
    cfg["notes"] = note or cfg.get("notes", "")
    key = key_for(cfg)
    if key not in runs:
        runs[key] = cfg
    else:
        runs[key]["phase"] = min(runs[key]["phase"], phase)
    tags[key].add(group)


# Phase 1: main comparisons, M sweep, momentum ablation, and rerun FO baselines.
for task, dataset in ALL:
    for seed in SEEDS:
        base = defaults(task, dataset, seed)
        add("main_comparison", base, 1)
        for M in [1, 5, 10, 15]:
            c = dict(base, M=M)
            add("M_sweep", c, 1)
        for xm, ym, name in [(True, True, "full"), (False, True, "no_x"), (True, False, "no_y"), (False, False, "no_both")]:
            c = dict(base, x_momentum=xm, y_momentum=ym, notes=f"momentum_variant={name}")
            add("momentum", c, 1, f"momentum_variant={name}")
        ttgda = dict(base, method="ttgda", estimator="direct", x_momentum=False, y_momentum=False)
        ttsgda = dict(base, method="ttsgda", estimator="direct", x_momentum=False, y_momentum=False)
        if task == "auc":
            ttsgda.update(lr_x=1e-3, lr_y=5e-2)
        else:
            ttsgda.update(lr_x=1e-3, lr_y=5e-2)
        add("main_comparison", ttgda, 1, "FO baseline rerun under forward budget")
        add("main_comparison", ttsgda, 1, "FO baseline rerun under forward budget")

# Phase 2: delta, perturbation form, direction distribution.
for task, dataset in ALL:
    for seed in SEEDS:
        base = defaults(task, dataset, seed)
        for delta in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            add("delta_sweep", dict(base, delta=delta), 2)
        add("perturbation_form", dict(base, estimator="one_sided"), 2, "shared-base one-sided estimator")
        add("perturbation_form", dict(base, estimator="direct"), 2, "direct FO x/y with ZOFOSGDA schedule")
        add("direction_ablation", dict(base, direction_distribution="rademacher"), 2)

# Phase 3: redo paper c1/c2 and batch-size ablations with fixed forward budget.
for task, dataset in LEGACY_ABLATION:
    for seed in SEEDS:
        base = defaults(task, dataset, seed)
        for c in [1.0, 2.0, 3.0]:
            for batch in [64, 128, 256]:
                add("coefficient_batch", dict(base, c1=c, c2=c, batch_size=batch), 3)

# Phase 4: model dimension ablation.
for task, dataset in ALL:
    for seed in SEEDS:
        base = defaults(task, dataset, seed)
        for hidden in [32, 64, 128, 256]:
            add("dimension", dict(base, hidden=hidden), 4)

# Phase 5: interactions on representative datasets.
for task, dataset in REPRESENTATIVE:
    for seed in SEEDS:
        base = defaults(task, dataset, seed)
        for M in [1, 5, 10, 15]:
            for delta in [1e-4, 1e-3, 1e-2]:
                add("M_delta_interaction", dict(base, M=M, delta=delta), 5)
        for M in [1, 5, 15]:
            for xm, ym, name in [(True, True, "full"), (False, True, "no_x"), (True, False, "no_y"), (False, False, "no_both")]:
                add("M_momentum_interaction", dict(base, M=M, x_momentum=xm, y_momentum=ym), 5, f"momentum_variant={name}")

# Phase 6: single-seed FO learning-rate sweep for diagnostic completeness.
for task, dataset in ALL:
    for method in ["ttgda", "ttsgda"]:
        base = defaults(task, dataset, 42)
        for lr_y in [1.0, 0.1, 0.01]:
            for ratio in [10, 100, 1000]:
                lr_x = lr_y / ratio
                add("fo_lr_sweep", dict(base, method=method, estimator="direct", x_momentum=False,
                                         y_momentum=False, lr_x=lr_x, lr_y=lr_y), 6,
                    f"lr_ratio={ratio}")

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

records = []
for key, cfg in runs.items():
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    mom = ("x" if cfg["x_momentum"] else "nx") + ("y" if cfg["y_momentum"] else "ny")
    stem = f"p{cfg['phase']}_{cfg['task']}_{cfg['dataset']}_{cfg['method']}_{cfg['estimator']}_M{cfg['M']}_d{cfg['delta']:g}_{mom}_b{cfg['batch_size']}_h{cfg['hidden']}_s{cfg['seed']}_{digest}"
    run_id = stem.replace(".", "p")
    cfg["run_id"] = run_id
    cfg["experiment_groups"] = ",".join(sorted(tags[key]))
    cfg["output_dir"] = f"rebuttal/results/{run_id}"
    path = CONFIG_DIR / f"{run_id}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    cost = cfg["max_train_forward_queries"] * (1.8 if cfg["method"] != "zofo" or cfg["estimator"] == "direct" else 1.0)
    cost *= cfg["batch_size"] / 128.0
    records.append({**cfg, "config_path": str(path.relative_to(ROOT)), "estimated_cost": cost, "status": "pending"})

# Phase-aware greedy balancing across 8 GPUs.
queues = [[] for _ in range(8)]
loads = [0.0] * 8
for phase in sorted({r["phase"] for r in records}):
    phase_rows = sorted((r for r in records if r["phase"] == phase), key=lambda r: r["estimated_cost"], reverse=True)
    for r in phase_rows:
        gpu = min(range(8), key=lambda i: loads[i])
        r["assigned_gpu"] = gpu
        queues[gpu].append(r)
        loads[gpu] += r["estimated_cost"]

for gpu, items in enumerate(queues):
    (QUEUE_DIR / f"gpu_{gpu}.txt").write_text("".join(r["config_path"] + "\n" for r in items), encoding="utf-8")

columns = sorted({k for r in records for k in r})
with (TABLE_DIR / "experiment_registry.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
    w.writeheader(); w.writerows(records)

manifest = {
    "total_runs": len(records),
    "by_phase": {str(p): sum(r["phase"] == p for r in records) for p in sorted({r["phase"] for r in records})},
    "by_group": {g: sum(g in r["experiment_groups"].split(",") for r in records) for g in sorted({g for r in records for g in r["experiment_groups"].split(",")})},
    "queue_lengths": [len(q) for q in queues],
    "estimated_loads": loads,
}
(QUEUE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
