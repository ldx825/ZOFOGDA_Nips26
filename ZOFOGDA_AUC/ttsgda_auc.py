import os
import csv
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 复用你项目的工具（和 TTSGDA.py 一致）
from utils import set_seed, load_libsvm_dataset, IndexedTabularDataset


# -------------------------
# 1-hidden-layer ReLU MLP (score h(x) \in R)
# -------------------------
class OneHiddenReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        s = self.fc2(x).squeeze(-1)  # (B,)
        return s


# -------------------------
# Config (仿照 TTSGDA.py)
# -------------------------
@dataclass
class AUCCfg:
    # data
    train_path: str = "./data/svmguide1"
    test_path: str = "./data/svmguide1.t"
    standardize: bool = True

    log_path: str = "./result/ttsgda_auc_svmguide1"

    # model
    hidden: int = 128

    # TTSGDA hyperparams (ηx << ηy) —— 和你文件一致风格 :contentReference[oaicite:2]{index=2}
    epochs: int = 200
    batch_size: int = 256
    lr_x: float = 1e-3     # η_x: for theta,a,b 下降
    lr_y: float = 5e-2     # η_y: for v 上升
    y_steps_per_batch: int = 1  # 可以 >1

    # dual projection / clip
    v_clip: float = 10.0

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -------------------------
# utilities: numpy/torch safe conversion
# -------------------------
def ensure_tensor(x, dtype=None, device=None):
    """Convert numpy/torch to torch.Tensor safely."""
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
    """
    输入 torch.Tensor labels（二分类），输出 torch.Tensor in {-1, +1}
    支持 {0,1}, {-1,1}, {1,2} 等
    """
    y_t = y_t.detach()
    uniq = torch.unique(y_t).cpu().tolist()
    if set(uniq) == {0, 1}:
        return torch.where(y_t > 0, torch.ones_like(y_t), -torch.ones_like(y_t))
    if set(uniq) == {-1, 1}:
        return y_t
    # 其它：最大值视为正类
    y_max = max(uniq)
    return torch.where(y_t == y_max, torch.ones_like(y_t), -torch.ones_like(y_t))


# -------------------------
# AUC minimax surrogate (batch)
# min over (theta,a,b), max over v
# -------------------------
def auc_minimax_loss(scores, y_pm1, a, b, v, p_pos: float):
    """
    scores: (B,)
    y_pm1 : (B,) in {-1,+1}
    a,b,v: scalar tensors
    p_pos: float
    """
    # p, q in tensor for device/dtype safety
    p = torch.tensor(p_pos, device=scores.device, dtype=scores.dtype)
    q = 1.0 - p

    Ipos = (y_pm1 > 0).to(scores.dtype)
    Ineg = (y_pm1 < 0).to(scores.dtype)

    # (1-p)(h-a)^2 I[y=1] + p(h-b)^2 I[y=-1]
    term_sq = q * (scores - a) ** 2 * Ipos + p * (scores - b) ** 2 * Ineg

    # 2(1+v)( p h I[y=-1] - (1-p) h I[y=1] )
    term_lin = 2.0 * (1.0 + v) * (p * scores * Ineg - q * scores * Ipos)

    # - p(1-p) v^2   (放进 mean，避免 batch 尺度不一致)
    term_v = -p * q * (v ** 2)

    return (term_sq + term_lin + term_v).mean()


# -------------------------
# rank-based AUC evaluation (no sklearn)
# -------------------------
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

    # average ranks for ties
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


# -------------------------
# Training (TT-SGDA style, matches your file structure)
# -------------------------
def train_auc_ttsgda(cfg: AUCCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # 读取方式与 TTSGDA.py 一样 :contentReference[oaicite:3]{index=3}
    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        cfg.train_path, cfg.test_path, standardize=cfg.standardize, seed=cfg.seed
    )

    # 用你的 Dataset 包装（内部可能转 torch，也可能不转——我们后面统一 ensure_tensor）
    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # 估计 p_pos：用原始 y_tr（可能是 numpy）先转 torch 再映射 {-1,+1}
    y_tr_t = ensure_tensor(y_tr)  # cpu tensor
    y_tr_pm1 = labels_to_pm1_tensor(y_tr_t)
    p_pos = float((y_tr_pm1 > 0).float().mean().item())

    # model
    model = OneHiddenReLUMLP(in_dim=D, hidden=cfg.hidden).to(device)

    # primal variables a,b (minimize)
    a = torch.tensor(0.0, device=device, requires_grad=True)
    b = torch.tensor(0.0, device=device, requires_grad=True)

    # dual variable v (maximize)
    v = torch.tensor(0.0, device=device, requires_grad=True)

    # optimizer for theta,a,b (下降)
    opt_x = torch.optim.SGD(list(model.parameters()) + [a, b], lr=cfg.lr_x)

    # logging
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True) if os.path.dirname(cfg.log_path) else None
    log_path = f"{cfg.log_path}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_auc", "a", "b", "v"])

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
        return fast_auc(torch.cat(all_s, dim=0), torch.cat(all_y, dim=0))

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

    # -------------------------
    # TT-SGDA main loop (matches your update pattern)
    # -------------------------
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, _ in train_loader:
            xb = ensure_tensor(xb, dtype=torch.float32, device=device)
            yb = ensure_tensor(yb, device=device)
            yb = labels_to_pm1_tensor(yb)

            # 清梯度：对齐你文件 :contentReference[oaicite:4]{index=4}
            if v.grad is not None:
                v.grad.zero_()
            opt_x.zero_grad()

            # forward & loss & backward
            scores = model(xb)
            loss = auc_minimax_loss(scores, yb, a, b, v, p_pos)
            loss.backward()

            # x-step: theta,a,b 下降
            opt_x.step()

            # y-step: v 上升（可多步）
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
            writer = csv.writer(f)
            writer.writerow([ep, tr_loss, te_auc, float(a.item()), float(b.item()), float(v.item())])

        print(
            f"Epoch {ep:03d} | TrainLoss={tr_loss:.6f} | TestAUC={te_auc:.4f} "
            f"| a={a.item():.3f}, b={b.item():.3f}, v={v.item():.3f}"
        )

    print("Log saved to:", log_path)
    return model, (a.detach().cpu().item(), b.detach().cpu().item(), v.detach().cpu().item())


if __name__ == "__main__":
    cfg = AUCCfg(
        train_path="./data/svmguide1",
        test_path="./data/svmguide1.t",
        log_path="./result/ttsgda_auc_svmguide1",

        standardize=True,
        hidden=128,

        # ηx << ηy
        lr_x=1e-2,
        lr_y=1e-1,
        y_steps_per_batch=1,

        epochs=200,
        batch_size=128,

        v_clip=10.0,
        seed=42,
    )

    model, (a, b, v) = train_auc_ttsgda(cfg)
    print("Done. a,b,v =", a, b, v)
