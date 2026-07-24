import os
import csv
import time
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        return self.fc2(x).squeeze(-1)  # (B,)


# -------------------------
# Wrap model + (a,b) into a single Module so ZO vectorization covers all x vars
# -------------------------
class AUCPrimalModule(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = OneHiddenReLUMLP(in_dim=in_dim, hidden=hidden)
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x)


# -------------------------
# Config: same schedule style as zofoGDA.py
# -------------------------
@dataclass
class ZOFOGDA_AUCCfg:
    # data
    train_path: str = "./data/svmguide1"
    test_path: str = "./data/svmguide1.t"
    standardize: bool = True
    log_path: str = "./result/zofoGDA_auc_svmguide1"

    # model
    hidden: int = 128

    # training
    epochs: int = 200
    batch_size: int = 128

    # ===== schedules ===== (same as zofoGDA.py) :contentReference[oaicite:5]{index=5}
    eta_k: float = 10.0
    eta_m: float = 1000.0
    c1: float = 3.0   # alpha_{t+1} = c1 * eta_t
    c2: float = 3.0   # beta_{t+1}  = c2 * eta_t

    gamma: float = 1.0  # x update: x~ = x - gamma*w ; x <- x + eta(x~ - x) :contentReference[oaicite:6]{index=6}
    lam: float = 1.0    # y prox: y~ = Pi(y - lam*v) ; y <- y + eta(y~ - y) :contentReference[oaicite:7]{index=7}

    # Zeroth-order gradient settings for x (same meaning as zofoGDA.py) :contentReference[oaicite:8]{index=8}
    zo_delta: float = 1e-4
    zo_q: int = 5

    # dual variable v projection bound (we replace simplex projection by interval projection)
    v_clip: float = 10.0

    # returning x-hat
    save_all_theta: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -------------------------
# helpers: flatten / set params (copying zofoGDA.py style) :contentReference[oaicite:9]{index=9}
# -------------------------
@torch.no_grad()
def params_to_vec(module: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p for p in module.parameters()]).detach()

@torch.no_grad()
def vec_to_params_(module: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, [p for p in module.parameters()])

@torch.no_grad()
def sample_unit_vec(d: int, device: torch.device) -> torch.Tensor:
    u = torch.randn(d, device=device)
    u = u / (u.norm() + 1e-12)
    return u


def labels_to_pm1(y: torch.Tensor) -> torch.Tensor:
    """
    y: torch tensor, binary labels from libsvm. Map to {-1,+1}.
    """
    y = y.detach()
    uniq = torch.unique(y).cpu().tolist()
    if set(uniq) == {0, 1}:
        return torch.where(y > 0, torch.ones_like(y), -torch.ones_like(y))
    if set(uniq) == {-1, 1}:
        return y
    y_max = max(uniq)
    return torch.where(y == y_max, torch.ones_like(y), -torch.ones_like(y))


def proj_interval(x: torch.Tensor, bound: float) -> torch.Tensor:
    """Projection onto [-bound, bound]."""
    if bound is None or bound <= 0:
        return x
    return torch.clamp(x, -bound, bound)


# -------------------------
# AUC minimax objective (min over x=(theta,a,b), max over v)
# -------------------------
def auc_minimax_loss(scores, y_pm1, a, b, v, p_pos: float):
    """
    scores: (B,)
    y_pm1 : (B,) in {-1,+1}
    a,b   : scalar tensors
    v     : scalar tensor
    p_pos : float
    """
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


def train_zofoGDA_auc_libsvm(cfg: ZOFOGDA_AUCCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # data loading: same path/function style as your files :contentReference[oaicite:10]{index=10}
    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        cfg.train_path, cfg.test_path, standardize=cfg.standardize, seed=cfg.seed
    )
    if K != 2:
        raise ValueError(f"AUC minimax here is binary-only; need K=2 but got K={K}")

    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # estimate p_pos using whole training labels
    y_tr_t = torch.as_tensor(y_tr) if not torch.is_tensor(y_tr) else y_tr
    y_tr_pm1 = labels_to_pm1(y_tr_t)
    p_pos = float((y_tr_pm1 > 0).float().mean().item())

    # primal module (x variables)
    primal = AUCPrimalModule(in_dim=D, hidden=cfg.hidden).to(device)

    # dual variable v (scalar), and its momentum gradient variable "g_y" like zofoGDA's v_t = ∇_y g
    dual = torch.tensor(0.0, device=device, requires_grad=True)

    # step schedules (same as zofoGDA.py) :contentReference[oaicite:11]{index=11}
    def eta_t(t: int) -> float:
        return float(cfg.eta_k) / float((cfg.eta_m + t) ** 0.5)

    def alpha_t(t: int) -> float:
        return float(cfg.c1) * eta_t(t)

    def beta_t(t: int) -> float:
        return float(cfg.c2) * eta_t(t)

    # f(x,y;B): stochastic objective for min_x max_y
    def f_batch(xb: torch.Tensor, yb_pm1: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        s = primal(xb)
        return auc_minimax_loss(s, yb_pm1, primal.a, primal.b, y_scalar, p_pos)

    # g = -f (same pattern as zofoGDA.py) :contentReference[oaicite:12]{index=12}
    def g_batch(xb: torch.Tensor, yb_pm1: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        return -f_batch(xb, yb_pm1, y_scalar)

    # first-order gradient for y on g: v_hat = ∇_y g  (same idea as grad_y_g) :contentReference[oaicite:13]{index=13}
    def grad_y_g(xb, yb_pm1):
        if dual.grad is not None:
            dual.grad.zero_()
        g = g_batch(xb, yb_pm1, dual)
        g.backward()
        v_hat = dual.grad.detach().clone()
        dual.requires_grad_(True)
        return v_hat

    # ZO gradient estimator for x = params(primal): two-point estimator (same as zofoGDA.py) :contentReference[oaicite:14]{index=14}
    @torch.no_grad()
    def zo_grad_x_f(xb, yb_pm1, q=1):
        x_vec = params_to_vec(primal)
        d = x_vec.numel()
        grad_est = torch.zeros_like(x_vec)

        for _ in range(q):
            u = sample_unit_vec(d, device=device)
            xp = x_vec + cfg.zo_delta * u
            xm = x_vec - cfg.zo_delta * u

            vec_to_params_(primal, xp)
            fp = f_batch(xb, yb_pm1, dual.detach())  # detach dual for x-gradient estimation (same pattern) :contentReference[oaicite:15]{index=15}
            vec_to_params_(primal, xm)
            fm = f_batch(xb, yb_pm1, dual.detach())

            vec_to_params_(primal, x_vec)  # restore

            coeff = (d / (2.0 * cfg.zo_delta)) * (fp - fm)
            grad_est += coeff * u

        grad_est /= float(q)
        return grad_est

    @torch.no_grad()
    def eval_test_auc():
        primal.eval()
        all_s, all_y = [], []
        for xb, yb, _ in test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            yb_pm1 = labels_to_pm1(yb)
            s = primal(xb)
            all_s.append(s.detach().cpu())
            all_y.append(yb_pm1.detach().cpu())
        return fast_auc(torch.cat(all_s, 0), torch.cat(all_y, 0))

    @torch.no_grad()
    def full_train_obj():
        # report F(x,y)=E_B f(x,y;B) (approx by full pass)
        primal.eval()
        total, cnt = 0.0, 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            yb_pm1 = labels_to_pm1(yb)
            loss = f_batch(xb, yb_pm1, dual.detach())
            bs = xb.size(0)
            total += float(loss.item()) * bs
            cnt += bs
        return total / max(cnt, 1)

    # logging
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True) if os.path.dirname(cfg.log_path) else None
    log_path = f"{cfg.log_path}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_auc", "a", "b", "dual", "w_norm", "vy"])

    # save x-hat
    thetas = []
    if cfg.save_all_theta:
        thetas.append({k: v.detach().cpu().clone() for k, v in primal.state_dict().items()})

    # -------------------------
    # Initialization with first batch: compute v1, w1 (same as zofoGDA.py) :contentReference[oaicite:16]{index=16}
    # -------------------------
    train_iter = iter(train_loader)
    try:
        xb, yb, _ = next(train_iter)
    except StopIteration:
        raise RuntimeError("Empty train loader.")
    xb = xb.to(device).float()
    yb = yb.to(device)
    yb_pm1 = labels_to_pm1(yb)

    vy = grad_y_g(xb, yb_pm1)                 # v1 = ∇_y g
    w = zo_grad_x_f(xb, yb_pm1, q=cfg.zo_q)   # w1 = ZO ∇_x f

    # -------------------------
    # Main loop: identical Step4/5/6/7 structure + momentum :contentReference[oaicite:17]{index=17}
    # -------------------------
    t = 1
    for ep in range(1, cfg.epochs + 1):
        primal.train()

        if ep != 1:
            train_iter = iter(train_loader)

        # we already consumed 1 batch for init in ep==1; reuse xb/yb_pm1 as current batch
        cur_xb, cur_yb_pm1 = xb, yb_pm1

        while True:
            et = eta_t(t)
            at = alpha_t(t)
            bt = beta_t(t)

            # ---- Step 4: x update: x~ = x - gamma*w ; x <- x + eta(x~ - x)  :contentReference[oaicite:18]{index=18}
            with torch.no_grad():
                x_vec = params_to_vec(primal)
                x_tilde = x_vec - cfg.gamma * w
                x_next = x_vec + et * (x_tilde - x_vec)     # = x_vec - eta*gamma*w
                vec_to_params_(primal, x_next)

            # ---- Step 5: y update: y~ = Pi(y - lam*vy) ; y <- y + eta(y~ - y)  :contentReference[oaicite:19]{index=19}
            # Note: vy = ∇_y g, g=-f => y - lam*vy = y + lam∇_y f (ascent)
            with torch.no_grad():
                y_t = dual.detach()
                y_tilde = proj_interval(y_t - cfg.lam * vy, cfg.v_clip)
                y_next = y_t + et * (y_tilde - y_t)
                dual.copy_(y_next)
            dual.requires_grad_(True)

            if cfg.save_all_theta:
                thetas.append({k: v.detach().cpu().clone() for k, v in primal.state_dict().items()})

            # ---- Step 6/7: need next batch B_{t+1} to compute w_hat, v_hat :contentReference[oaicite:20]{index=20}
            try:
                xb2, yb2, _ = next(train_iter)
            except StopIteration:
                break
            xb2 = xb2.to(device).float()
            yb2 = yb2.to(device)
            yb2_pm1 = labels_to_pm1(yb2)

            w_hat = zo_grad_x_f(xb2, yb2_pm1, q=cfg.zo_q)   # :contentReference[oaicite:21]{index=21}
            vy_hat = grad_y_g(xb2, yb2_pm1)                 # :contentReference[oaicite:22]{index=22}

            # momentum updates (same formulas) :contentReference[oaicite:23]{index=23}
            with torch.no_grad():
                w = (1.0 - bt) * w + bt * w_hat
                vy = (1.0 - at) * vy + at * vy_hat

            # advance
            cur_xb, cur_yb_pm1 = xb2, yb2_pm1
            t += 1

        # epoch end: evaluate
        tr_f = full_train_obj()
        te_auc = eval_test_auc()

        with torch.no_grad():
            w_norm = float(w.norm().item())
            vy_val = float(vy.item()) if vy.numel() == 1 else float(vy.norm().item())

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, tr_f, te_auc, float(primal.a.item()), float(primal.b.item()),
                             float(dual.item()), w_norm, vy_val])

        print(
            f"Epoch {ep:03d} | TrainLoss={tr_f:.6f} | TestAUC={te_auc:.4f} "
            f"| a={primal.a.item():.3f}, b={primal.b.item():.3f}, dual(v)={dual.item():.3f}"
        )

    print("Log saved to:", log_path)

    # return x-hat like the original code (uniformly sample from stored iterates)
    if cfg.save_all_theta and len(thetas) > 0:
        tt = random.randrange(len(thetas))
        primal.load_state_dict(thetas[tt])
        return primal, float(dual.detach().cpu().item()), tt, log_path
    return primal, float(dual.detach().cpu().item()), None, log_path


if __name__ == "__main__":
    cfg = ZOFOGDA_AUCCfg(
        train_path="./data/svmguide1",
        test_path="./data/svmguide1.t",
        log_path="./result/zofoGDA_auc_svmguide1",
        standardize=True,
        hidden=128,

        epochs=200,
        batch_size=128,

        eta_k=10.0,
        eta_m=1000.0,
        c1=3.0,
        c2=3.0,
        gamma=0.01,
        lam=0.01,

        zo_delta=1e-4,
        zo_q=5,

        v_clip=10.0,
        save_all_theta=True,
        seed=42,
    )

    model, dual_v, t, log_path = train_zofoGDA_auc_libsvm(cfg)
    print("Returned x-hat from step:", t)
    print("Final dual(v):", dual_v)
    print("Log:", log_path)
