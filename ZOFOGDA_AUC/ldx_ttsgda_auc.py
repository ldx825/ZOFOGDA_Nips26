import os
import csv
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed, load_libsvm_dataset, IndexedTabularDataset


class OneHiddenReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


@dataclass
class AUCCfg:
    dataset: str = "svmguide1"
    data_dir: str = "./data"
    train_path: str = ""
    test_path: str = ""

    standardize: bool = True

    log_dir: str = "./result"
    log_prefix: str = "ttsgda_auc"

    hidden: int = 128

    epochs: int = 200
    batch_size: int = 256

    lr_x: float = 1e-3
    lr_y: float = 5e-2
    y_steps_per_batch: int = 1

    v_clip: float = 10.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def ensure_tensor(x, dtype=None, device=None):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.from_numpy(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def labels_to_pm1_tensor(y_t: torch.Tensor) -> torch.Tensor:
    y_t = y_t.detach()
    uniq = torch.unique(y_t).cpu().tolist()
    if set(uniq) == {0, 1}:
        return torch.where(y_t > 0, torch.ones_like(y_t), -torch.ones_like(y_t))
    if set(uniq) == {-1, 1}:
        return y_t
    y_max = max(uniq)
    return torch.where(y_t == y_max, torch.ones_like(y_t), -torch.ones_like(y_t))


def resolve_paths(cfg: AUCCfg):
    if cfg.train_path and cfg.train_path.strip():
        train_path = cfg.train_path
    else:
        train_path = os.path.join(cfg.data_dir, cfg.dataset)

    if cfg.test_path and cfg.test_path.strip():
        test_path = cfg.test_path
    else:
        cand = [
            os.path.join(cfg.data_dir, f"{cfg.dataset}.t"),
            os.path.join(cfg.data_dir, f"{cfg.dataset}.test"),
            os.path.join(cfg.data_dir, f"{cfg.dataset}_test"),
        ]
        test_path = None
        for p in cand:
            if os.path.exists(p):
                test_path = p
                break

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    return train_path, test_path


def build_log_path(cfg: AUCCfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    return os.path.join(cfg.log_dir, f"{cfg.log_prefix}_{cfg.dataset}_epochs{cfg.epochs}_seed{cfg.seed}_batch{cfg.batch_size}_lrx{cfg.lr_x}_lry{cfg.lr_y}_log.csv")


def auc_minimax_loss(scores, y_pm1, a, b, v, p_pos: float):
    p = torch.tensor(p_pos, device=scores.device, dtype=scores.dtype)
    q = 1.0 - p

    Ipos = (y_pm1 > 0).to(scores.dtype)
    Ineg = (y_pm1 < 0).to(scores.dtype)

    term_sq = q * (scores - a) ** 2 * Ipos + p * (scores - b) ** 2 * Ineg
    term_lin = 2.0 * (1.0 + v) * (p * scores * Ineg - q * scores * Ipos)
    term_v = -p * q * (v ** 2)

    return (term_sq + term_lin + term_v).mean()


@torch.no_grad()
def fast_auc(scores: torch.Tensor, y_pm1: torch.Tensor) -> float:
    scores = scores.flatten().cpu()
    y_pm1 = y_pm1.flatten().cpu()
    pos = (y_pm1 > 0)
    neg = (y_pm1 < 0)
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    idx = torch.argsort(scores)
    s_sorted = scores[idx]

    ranks_sorted = torch.empty_like(s_sorted, dtype=torch.float64)
    i = 0
    r = 1
    while i < s_sorted.numel():
        j = i
        while j + 1 < s_sorted.numel() and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks_sorted[i:j + 1] = avg_rank
        r += (j - i + 1)
        i = j + 1

    ranks = torch.empty_like(ranks_sorted)
    ranks[idx] = ranks_sorted

    sum_ranks_pos = ranks[pos].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train_auc_ttsgda(cfg: AUCCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_path, test_path = resolve_paths(cfg)
    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        train_path, test_path, standardize=cfg.standardize, seed=cfg.seed
    )
    if K != 2:
        raise ValueError(f"Binary-only AUC here, but got K={K}")

    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    y_tr_t = ensure_tensor(y_tr)
    y_tr_pm1 = labels_to_pm1_tensor(y_tr_t)
    p_pos = float((y_tr_pm1 > 0).float().mean().item())

    model = OneHiddenReLUMLP(in_dim=D, hidden=cfg.hidden).to(device)

    a = torch.tensor(0.0, device=device, requires_grad=True)
    b = torch.tensor(0.0, device=device, requires_grad=True)
    v = torch.tensor(0.0, device=device, requires_grad=True)

    opt_x = torch.optim.SGD(list(model.parameters()) + [a, b], lr=cfg.lr_x)

    log_path = build_log_path(cfg)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "test_auc", "a", "b", "v"])
        w.writerow(["#cfg", str(asdict(cfg))])

    @torch.no_grad()
    def eval_auc():
        model.eval()
        all_s, all_y = [], []
        for xb, yb, _ in test_loader:
            xb = ensure_tensor(xb, dtype=torch.float32, device=device)
            yb = ensure_tensor(yb, device=device)
            yb = labels_to_pm1_tensor(yb)
            s = model(xb)
            all_s.append(s.detach().cpu())
            all_y.append(yb.detach().cpu())
        return fast_auc(torch.cat(all_s, 0), torch.cat(all_y, 0))

    @torch.no_grad()
    def full_train_loss():
        model.eval()
        total, cnt = 0.0, 0
        for xb, yb, _ in train_loader:
            xb = ensure_tensor(xb, dtype=torch.float32, device=device)
            yb = ensure_tensor(yb, device=device)
            yb = labels_to_pm1_tensor(yb)
            s = model(xb)
            loss = auc_minimax_loss(s, yb, a, b, v, p_pos)
            bs = xb.size(0)
            total += float(loss.item()) * bs
            cnt += bs
        return total / max(cnt, 1)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, _ in train_loader:
            xb = ensure_tensor(xb, dtype=torch.float32, device=device)
            yb = ensure_tensor(yb, device=device)
            yb = labels_to_pm1_tensor(yb)

            if v.grad is not None:
                v.grad.zero_()
            opt_x.zero_grad()

            scores = model(xb)
            loss = auc_minimax_loss(scores, yb, a, b, v, p_pos)
            loss.backward()

            opt_x.step()

            with torch.no_grad():
                for _ in range(max(1, cfg.y_steps_per_batch)):
                    if v.grad is None:
                        break
                    v += cfg.lr_y * v.grad
                    if cfg.v_clip is not None and cfg.v_clip > 0:
                        v.clamp_(-cfg.v_clip, cfg.v_clip)

            v.requires_grad_(True)

        tr_loss = full_train_loss()
        te_auc = eval_auc()

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, tr_loss, te_auc, float(a.item()), float(b.item()), float(v.item())])

        print(
            f"[TTSGDA|{cfg.dataset}] Epoch {ep:03d} | TrainLoss={tr_loss:.6f} | TestAUC={te_auc:.4f} "
            f"| a={a.item():.3f}, b={b.item():.3f}, v={v.item():.3f}"
        )

    print("Log saved to:", log_path)
    return model, (a.detach().cpu().item(), b.detach().cpu().item(), v.detach().cpu().item())


def parse_args():
    p = argparse.ArgumentParser("TTSGDA-AUC (LIBSVM) - CLI")

    p.add_argument("--dataset", type=str, default="svmguide1",
                   choices=["climate", "german", "svmguide1", "spambase", "a9a", "ijcnn1", "covtype", "connect", "acoustic"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--train_path", type=str, default="")
    p.add_argument("--test_path", type=str, default="")

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--lr_x", type=float, default=1e-3)
    p.add_argument("--lr_y", type=float, default=5e-2)
    p.add_argument("--y_steps_per_batch", type=int, default=1)

    p.add_argument("--v_clip", type=float, default=10.0)

    p.add_argument("--standardize", action="store_true")
    p.add_argument("--no_standardize", action="store_true")

    p.add_argument("--log_dir", type=str, default="./result")
    p.add_argument("--log_prefix", type=str, default="ttsgda_auc")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    return p.parse_args()


def main():
    args = parse_args()
    standardize = False if args.no_standardize else True

    cfg = AUCCfg(
        dataset=args.dataset,
        data_dir=args.data_dir,
        train_path=args.train_path,
        test_path=args.test_path,
        standardize=standardize,

        log_dir=args.log_dir,
        log_prefix=args.log_prefix,

        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,

        lr_x=args.lr_x,
        lr_y=args.lr_y,
        y_steps_per_batch=args.y_steps_per_batch,

        v_clip=args.v_clip,
        seed=args.seed,
        device=args.device,
    )

    model, (a, b, v) = train_auc_ttsgda(cfg)
    print("Done. a,b,v =", a, b, v)


if __name__ == "__main__":
    main()
