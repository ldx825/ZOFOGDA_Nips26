#!/usr/bin/env python3
"""Unified query-budgeted runner for NeurIPS 2026 rebuttal experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
AUC_DATA = ROOT / "26.1.29.ZOFOGDA_AUC" / "data"
ROBUST_DATA = ROOT / "rebuttal" / "data" / "robust"
RESULT_ROOT = ROOT / "rebuttal" / "results"


@dataclass
class Config:
    run_id: str = "debug"
    experiment_groups: str = "smoke"
    phase: int = 0
    task: str = "auc"  # auc | robust
    dataset: str = "svmguide1"
    method: str = "zofo"  # zofo | ttgda | ttsgda
    estimator: str = "two_sided"  # two_sided | one_sided | direct
    seed: int = 42
    data_seed: int = 0
    device: str = "cuda:0"
    output_dir: str = ""

    hidden: int = 128
    batch_size: int = 128
    standardize: bool = True
    max_train_forward_queries: int = 50000
    eval_every_forward_queries: int = 2500
    eval_train_batches: int = 8

    M: int = 5
    delta: float = 1e-4
    direction_distribution: str = "sphere"
    x_momentum: bool = True
    y_momentum: bool = True
    eta_k: float = 10.0
    eta_m: float = 1000.0
    c1: float = 3.0
    c2: float = 3.0
    gamma: float = 0.01
    lam: float = 0.01

    lr_x: float = 0.01
    lr_y: float = 0.0001
    dual_clip: float = 10.0
    theta1: float = 0.01
    save_model: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**raw)


class IndexedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], idx


class ScalarMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


class AUCPrimal(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = ScalarMLP(in_dim, hidden)
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RobustMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


@dataclass
class QueryCounter:
    train_forward_queries: int = 0
    train_backward_calls: int = 0
    train_sample_forwards: int = 0
    train_sample_backwards: int = 0
    x_function_queries: int = 0
    eval_forward_queries: int = 0
    eval_sample_forwards: int = 0

    def train_forward(self, batch: int, *, x_function: bool = False) -> None:
        self.train_forward_queries += 1
        self.train_sample_forwards += int(batch)
        if x_function:
            self.x_function_queries += 1

    def train_backward(self, batch: int) -> None:
        self.train_backward_calls += 1
        self.train_sample_backwards += int(batch)

    def eval_forward(self, batch: int) -> None:
        self.eval_forward_queries += 1
        self.eval_sample_forwards += int(batch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_paths(cfg: Config) -> Tuple[Path, Optional[Path]]:
    base = AUC_DATA if cfg.task == "auc" else ROBUST_DATA
    train = base / cfg.dataset
    candidates = [base / f"{cfg.dataset}.t", base / f"{cfg.dataset}.test"]
    test = next((p for p in candidates if p.exists()), None)
    if not train.exists():
        raise FileNotFoundError(f"Missing train data: {train}")
    return train, test


def load_data(cfg: Config):
    train_path, test_path = resolve_paths(cfg)
    if test_path is not None:
        X_tr, y_tr_raw, X_te, y_te_raw = load_svmlight_files([str(train_path), str(test_path)])
        X_tr = X_tr.toarray().astype(np.float32)
        X_te = X_te.toarray().astype(np.float32)
    else:
        X, y_raw = load_svmlight_file(str(train_path))
        X = X.toarray().astype(np.float32)
        y_raw = np.asarray(y_raw)
        X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(
            X, y_raw, test_size=0.2, random_state=cfg.data_seed, stratify=y_raw
        )
    labels = sorted(np.unique(np.concatenate([y_tr_raw, y_te_raw])).tolist())
    if len(labels) != 2:
        raise ValueError(f"Expected binary labels, found {labels}")
    mapping = {label: i for i, label in enumerate(labels)}
    y_tr = np.array([mapping[v] for v in y_tr_raw], dtype=np.int64)
    y_te = np.array([mapping[v] for v in y_te_raw], dtype=np.int64)
    if cfg.standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, y_tr, X_te, y_te


@torch.no_grad()
def params_to_vec(module: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()


@torch.no_grad()
def vec_to_params(module: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, module.parameters())


def grads_to_vec(module: nn.Module) -> torch.Tensor:
    parts = []
    for p in module.parameters():
        parts.append(torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().flatten())
    return torch.cat(parts)


@torch.no_grad()
def project_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not torch.any(cond):
        return torch.full_like(v, z / v.numel())
    rho = torch.nonzero(cond, as_tuple=False).max().item()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def sample_direction(d: int, device: torch.device, distribution: str) -> torch.Tensor:
    if distribution == "sphere":
        u = torch.randn(d, device=device)
        return u / (u.norm() + 1e-12)
    if distribution == "rademacher":
        u = torch.empty(d, device=device).bernoulli_(0.5).mul_(2).sub_(1)
        return u / math.sqrt(d)
    raise ValueError(f"Unknown direction distribution: {distribution}")


def auc_loss(scores, labels01, a, b, dual, p_pos: float):
    y = labels01.float() * 2.0 - 1.0
    p = torch.as_tensor(p_pos, device=scores.device, dtype=scores.dtype)
    q = 1.0 - p
    pos = (y > 0).to(scores.dtype)
    neg = (y < 0).to(scores.dtype)
    sq = q * (scores - a) ** 2 * pos + p * (scores - b) ** 2 * neg
    lin = 2.0 * (1.0 + dual) * (p * scores * neg - q * scores * pos)
    concave = -p * q * dual ** 2
    return (sq + lin + concave).mean()


def robust_hinge(logits, labels01):
    score = logits[:, 1] - logits[:, 0]
    signed = labels01.float() * 2.0 - 1.0
    return F.relu(1.0 - signed * score)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        X_tr, y_tr, X_te, y_te = load_data(cfg)
        self.N, self.D = len(y_tr), X_tr.shape[1]
        self.y_train_np = y_tr
        self.p_pos = float((y_tr == 1).mean())
        self.train_ds = IndexedDataset(X_tr, y_tr)
        self.test_ds = IndexedDataset(X_te, y_te)
        gen = torch.Generator().manual_seed(cfg.seed)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True,
            drop_last=False, generator=gen, num_workers=0
        )
        self.eval_train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=cfg.batch_size, shuffle=False)
        if cfg.task == "auc":
            self.model = AUCPrimal(self.D, cfg.hidden).to(self.device)
            self.dual = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif cfg.task == "robust":
            self.model = RobustMLP(self.D, cfg.hidden).to(self.device)
            self.dual = torch.full((self.N,), 1.0 / self.N, device=self.device, requires_grad=True)
        else:
            raise ValueError(f"Unknown task: {cfg.task}")
        self.counter = QueryCounter()
        self.gx_state = None
        self.gy_state = None
        self.step = 0
        self.train_wall = 0.0

    def objective(self, xb, yb, idx, dual):
        if self.cfg.task == "auc":
            scores = self.model(xb)
            return auc_loss(scores, yb, self.model.a, self.model.b, dual, self.p_pos)
        logits = self.model(xb)
        hinge = robust_hinge(logits, yb)
        weighted = (dual[idx] * hinge).mean()
        reg = 0.5 * self.cfg.theta1 * torch.sum((dual - 1.0 / self.N) ** 2)
        return weighted - reg

    def expected_step_cost(self) -> int:
        if self.cfg.method != "zofo" or self.cfg.estimator == "direct":
            return 1
        if self.cfg.estimator == "two_sided":
            return 2 * self.cfg.M + 1
        if self.cfg.estimator == "one_sided":
            return self.cfg.M + 2
        raise ValueError(self.cfg.estimator)

    @torch.no_grad()
    def zo_x_grad(self, xb, yb, idx) -> torch.Tensor:
        base_vec = params_to_vec(self.model)
        d = base_vec.numel()
        est = torch.zeros_like(base_vec)
        if self.cfg.estimator == "one_sided":
            vec_to_params(self.model, base_vec)
            f0 = self.objective(xb, yb, idx, self.dual.detach())
            self.counter.train_forward(len(yb), x_function=True)
        for _ in range(self.cfg.M):
            u = sample_direction(d, self.device, self.cfg.direction_distribution)
            vec_to_params(self.model, base_vec + self.cfg.delta * u)
            fp = self.objective(xb, yb, idx, self.dual.detach())
            self.counter.train_forward(len(yb), x_function=True)
            if self.cfg.estimator == "two_sided":
                vec_to_params(self.model, base_vec - self.cfg.delta * u)
                fm = self.objective(xb, yb, idx, self.dual.detach())
                self.counter.train_forward(len(yb), x_function=True)
                est += (d / (2.0 * self.cfg.delta)) * (fp - fm) * u
            else:
                est += (d / self.cfg.delta) * (fp - f0) * u
        vec_to_params(self.model, base_vec)
        return est / float(self.cfg.M)

    def y_grad(self, xb, yb, idx) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        if self.dual.grad is not None:
            self.dual.grad.zero_()
        loss = self.objective(xb, yb, idx, self.dual)
        self.counter.train_forward(len(yb))
        loss.backward()
        self.counter.train_backward(len(yb))
        gy = self.dual.grad.detach().clone()
        self.model.zero_grad(set_to_none=True)
        return gy

    def joint_grads(self, xb, yb, idx):
        self.model.zero_grad(set_to_none=True)
        if self.dual.grad is not None:
            self.dual.grad.zero_()
        loss = self.objective(xb, yb, idx, self.dual)
        self.counter.train_forward(len(yb), x_function=False)
        loss.backward()
        self.counter.train_backward(len(yb))
        gx = grads_to_vec(self.model)
        gy = self.dual.grad.detach().clone()
        return gx, gy

    def momentum(self, old, new, coefficient: float, enabled: bool):
        if (not enabled) or old is None:
            return new.detach().clone()
        coefficient = min(max(float(coefficient), 0.0), 1.0)
        return (1.0 - coefficient) * old + coefficient * new

    @torch.no_grad()
    def update_dual(self, direction: torch.Tensor, step_size: float, extrapolation: Optional[float] = None):
        current = self.dual.detach()
        proposal = current + step_size * direction
        if self.cfg.task == "auc":
            proposal = torch.clamp(proposal, -self.cfg.dual_clip, self.cfg.dual_clip)
        else:
            proposal = project_simplex(proposal)
        new = proposal if extrapolation is None else current + extrapolation * (proposal - current)
        self.dual.copy_(new)
        self.dual.requires_grad_(True)

    def train_step(self, batch):
        xb, yb, idx = batch
        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)
        idx = idx.to(self.device, non_blocking=True)
        sync(self.device)
        start = time.perf_counter()
        if self.cfg.method == "zofo":
            if self.cfg.estimator == "direct":
                gx_hat, gy_hat = self.joint_grads(xb, yb, idx)
            else:
                gx_hat = self.zo_x_grad(xb, yb, idx)
                gy_hat = self.y_grad(xb, yb, idx)
            eta = self.cfg.eta_k / math.sqrt(self.cfg.eta_m + self.step + 1)
            self.gx_state = self.momentum(
                self.gx_state, gx_hat, self.cfg.c2 * eta, self.cfg.x_momentum
            )
            self.gy_state = self.momentum(
                self.gy_state, gy_hat, self.cfg.c1 * eta, self.cfg.y_momentum
            )
            with torch.no_grad():
                x = params_to_vec(self.model) - eta * self.cfg.gamma * self.gx_state
                vec_to_params(self.model, x)
            self.update_dual(self.gy_state, self.cfg.lam, extrapolation=eta)
        else:
            gx, gy = self.joint_grads(xb, yb, idx)
            with torch.no_grad():
                x = params_to_vec(self.model) - self.cfg.lr_x * gx
                vec_to_params(self.model, x)
            self.update_dual(gy, self.cfg.lr_y)
        sync(self.device)
        self.train_wall += time.perf_counter() - start
        self.step += 1

    @torch.no_grad()
    def evaluate(self):
        sync(self.device)
        start = time.perf_counter()
        self.model.eval()
        preds, labels = [], []
        for xb, yb, _ in self.test_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            if self.cfg.task == "auc":
                out = self.model(xb)
            else:
                out = self.model(xb).argmax(dim=1)
            self.counter.eval_forward(len(yb))
            preds.append(out.detach().cpu())
            labels.append(yb.detach().cpu())
        pred = torch.cat(preds).numpy()
        lab = torch.cat(labels).numpy()
        metric = float(roc_auc_score(lab, pred)) if self.cfg.task == "auc" else float((pred == lab).mean())

        losses = []
        for i, (xb, yb, idx) in enumerate(self.eval_train_loader):
            if i >= self.cfg.eval_train_batches:
                break
            xb, yb, idx = xb.to(self.device), yb.to(self.device), idx.to(self.device)
            loss = self.objective(xb, yb, idx, self.dual.detach())
            self.counter.eval_forward(len(yb))
            losses.append(float(loss.item()))
        sync(self.device)
        eval_wall = time.perf_counter() - start
        self.model.train()
        return metric, float(np.mean(losses)) if losses else float("nan"), eval_wall

    def run(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.csv"
        fieldnames = [
            "run_id", "task", "dataset", "method", "estimator", "seed", "global_step",
            "target_query", "test_metric", "train_objective", "cum_train_forward_queries",
            "cum_x_function_queries", "cum_train_backward_calls", "cum_train_sample_forwards",
            "cum_train_sample_backwards", "train_wall_time_sec", "eval_forward_queries",
            "eval_sample_forwards", "eval_wall_time_sec", "M", "delta", "x_momentum", "y_momentum"
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        checkpoints = list(range(0, self.cfg.max_train_forward_queries + 1, self.cfg.eval_every_forward_queries))
        if checkpoints[-1] != self.cfg.max_train_forward_queries:
            checkpoints.append(self.cfg.max_train_forward_queries)
        next_checkpoint_index = 0
        eval_wall_total = 0.0

        def log_eval(target):
            nonlocal eval_wall_total
            metric, train_obj, ew = self.evaluate()
            eval_wall_total += ew
            row = {
                "run_id": self.cfg.run_id, "task": self.cfg.task, "dataset": self.cfg.dataset,
                "method": self.cfg.method, "estimator": self.cfg.estimator, "seed": self.cfg.seed,
                "global_step": self.step, "target_query": target, "test_metric": metric,
                "train_objective": train_obj,
                "cum_train_forward_queries": self.counter.train_forward_queries,
                "cum_x_function_queries": self.counter.x_function_queries,
                "cum_train_backward_calls": self.counter.train_backward_calls,
                "cum_train_sample_forwards": self.counter.train_sample_forwards,
                "cum_train_sample_backwards": self.counter.train_sample_backwards,
                "train_wall_time_sec": self.train_wall,
                "eval_forward_queries": self.counter.eval_forward_queries,
                "eval_sample_forwards": self.counter.eval_sample_forwards,
                "eval_wall_time_sec": eval_wall_total,
                "M": self.cfg.M, "delta": self.cfg.delta,
                "x_momentum": self.cfg.x_momentum, "y_momentum": self.cfg.y_momentum,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            return row

        rows = [log_eval(0)]
        next_checkpoint_index = 1
        loader_iter = iter(self.train_loader)
        while self.counter.train_forward_queries + self.expected_step_cost() <= self.cfg.max_train_forward_queries:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)
            self.train_step(batch)
            if (next_checkpoint_index < len(checkpoints) and
                    self.counter.train_forward_queries >= checkpoints[next_checkpoint_index]):
                rows.append(log_eval(checkpoints[next_checkpoint_index]))
                next_checkpoint_index += 1

        if rows[-1]["cum_train_forward_queries"] != self.counter.train_forward_queries:
            rows.append(log_eval(self.cfg.max_train_forward_queries))
        final = rows[-1]
        summary = {
            "status": "completed", **asdict(self.cfg),
            "n_train": self.N, "n_test": len(self.test_ds), "feature_dim": self.D,
            "parameter_dim": sum(p.numel() for p in self.model.parameters()),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "final_test_metric": final["test_metric"],
            "best_test_metric": max(r["test_metric"] for r in rows),
            "final_train_objective": final["train_objective"],
            "global_steps": self.step,
            "cum_train_forward_queries": self.counter.train_forward_queries,
            "cum_x_function_queries": self.counter.x_function_queries,
            "cum_train_backward_calls": self.counter.train_backward_calls,
            "cum_train_sample_forwards": self.counter.train_sample_forwards,
            "train_wall_time_sec": self.train_wall,
            "eval_wall_time_sec": sum(float(r.get("eval_wall_time_sec", 0)) for r in rows[-1:]),
            "metrics_path": (str(metrics_path.relative_to(ROOT))
                             if metrics_path.is_relative_to(ROOT) else str(metrics_path)),
        }
        if self.cfg.save_model:
            torch.save({"model": self.model.state_dict(), "dual": self.dual.detach().cpu()}, out_dir / "final.pt")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    config_path = Path(args.config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device:
        raw["device"] = args.device
    cfg = Config.from_dict(raw)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else RESULT_ROOT / cfg.run_id
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    running = {"status": "running", **asdict(cfg), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / "summary.json").write_text(json.dumps(running, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        trainer = Trainer(cfg)
        summary = trainer.run(out_dir)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        failed = {**running, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        (out_dir / "summary.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
