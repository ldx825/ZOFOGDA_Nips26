import os
import csv
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler


# =========================
# Utils (standalone)
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedTabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def load_libsvm_dataset(train_path: str,
                        test_path: str ,
                        standardize: bool = True,
                        seed: int = 42):
    """
    Loads LIBSVM formatted dataset.
    - If test_path exists, load it; else split train 80/20.
    - Ensures train/test have consistent n_features by using max feature index.
    - Maps labels to {0..K-1}.
    Returns:
        X_tr (N,D) float32, y_tr (N,) int64
        X_te (Nt,D) float32, y_te (Nt,) int64
        num_classes K, feature_dim D
    """
    # First load train with feature inference
    X_tr_sp, y_tr_raw = load_svmlight_file(train_path)
    D = X_tr_sp.shape[1]

    # Load test with same feature dimension (n_features=D)
    if test_path is not None and os.path.exists(test_path):
        X_te_sp, y_te_raw = load_svmlight_file(test_path, n_features=D)
    else:
        X_te_sp, y_te_raw = None, None

    X_tr = X_tr_sp.toarray().astype(np.float32)
    y_tr_raw = np.asarray(y_tr_raw)

    if X_te_sp is not None:
        X_te = X_te_sp.toarray().astype(np.float32)
        y_te_raw = np.asarray(y_te_raw)
    else:
        # split 80/20
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X_tr.shape[0])
        split = int(0.8 * len(idx))
        tr_idx, te_idx = idx[:split], idx[split:]
        X_te = X_tr[te_idx]
        y_te_raw = y_tr_raw[te_idx]
        X_tr = X_tr[tr_idx]
        y_tr_raw = y_tr_raw[tr_idx]

    # Map labels to 0..K-1 (works for {-1,+1} too)
    all_labels = np.unique(np.concatenate([y_tr_raw, y_te_raw], axis=0))
    label2id = {lab: i for i, lab in enumerate(all_labels.tolist())}
    y_tr = np.vectorize(label2id.get)(y_tr_raw).astype(np.int64)
    y_te = np.vectorize(label2id.get)(y_te_raw).astype(np.int64)
    K = len(all_labels)

    # Handle multiclass labels by converting them to binary labels (1 vs -1)
    if K > 2:
        y_tr = np.where(y_tr == 0, -1, 1).astype(np.int64)  # Example: map class 0 to -1, others to +1
        y_te = np.where(y_te == 0, -1, 1).astype(np.int64)
        K = 2  # Force K to 2 for binary classification

    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

    return X_tr, y_tr, X_te, y_te, K, X_tr.shape[1]



def labels_to_pm1(y: torch.Tensor) -> torch.Tensor:
    """
    Map labels to {-1,+1}. Supports {0,1} or {-1,+1} or arbitrary two labels.
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
    if bound is None or bound <= 0:
        return x
    return torch.clamp(x, -bound, bound)


@torch.no_grad()
def fast_auc(scores: torch.Tensor, y_pm1: torch.Tensor) -> float:
    """
    AUC via rank statistic (handles ties with average ranks).
    """
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


# =========================
# Model
# =========================
class OneHiddenReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class AUCPrimalModule(nn.Module):
    """
    x variables = (theta, a, b)
    """
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = OneHiddenReLUMLP(in_dim=in_dim, hidden=hidden)
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x)


# =========================
# Config
# =========================
@dataclass
class ZOFOGDA_AUCCfg:
    # dataset control
    dataset: str = "svmguide1"
    data_dir: str = "./data"
    train_path: str = ""
    test_path: str = ""

    standardize: bool = True
    log_dir: str = "./result"
    log_prefix: str = "zofoGDA_auc"

    # model
    hidden: int = 128

    # training
    epochs: int = 200
    batch_size: int = 128

    # schedules
    eta_k: float = 10.0
    eta_m: float = 1000.0
    c1: float = 3.0
    c2: float = 3.0

    gamma: float = 0.01
    lam: float = 0.01

    # ZO gradient
    zo_delta: float = 1e-4
    zo_q: int = 5
    x_momentum: bool = True
    y_momentum: bool = True

    # dual projection
    v_clip: float = 10.0

    save_all_theta: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_paths(cfg: ZOFOGDA_AUCCfg):
    """
    Make it so you only change --dataset and it works:
      train = {data_dir}/{dataset}
      test  = prefer {dataset}.t, else {dataset}.test, else None
    If user passes explicit --train_path/--test_path, use them.
    """
    # train
    if cfg.train_path and cfg.train_path.strip():
        train_path = cfg.train_path
    else:
        train_path = os.path.join(cfg.data_dir, cfg.dataset)

    # test
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


def build_log_path(cfg: ZOFOGDA_AUCCfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    # e.g. ./result/zofoGDA_auc_svmguide1_log.csv
    stem = f"{cfg.log_prefix}_{cfg.dataset}_epochs{cfg.epochs}_seed{cfg.seed}_batch{cfg.batch_size}_gamma{cfg.gamma}_lam{cfg.lam}_delta{cfg.zo_delta}q{cfg.zo_q}"
    return os.path.join(cfg.log_dir, f"{stem}_log.csv")


# =========================
# Vector helpers
# =========================
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


# =========================
# AUC minimax objective (same as your code)
# =========================
def auc_minimax_loss(scores, y_pm1, a, b, v, p_pos: float):
    """
    Same function as your version.
    """
    p = torch.tensor(p_pos, device=scores.device, dtype=scores.dtype)
    q = 1.0 - p

    Ipos = (y_pm1 > 0).to(scores.dtype)
    Ineg = (y_pm1 < 0).to(scores.dtype)

    term_sq = q * (scores - a) ** 2 * Ipos + p * (scores - b) ** 2 * Ineg
    term_lin = 2.0 * (1.0 + v) * (p * scores * Ineg - q * scores * Ipos)
    term_v = -p * q * (v ** 2)

    return (term_sq + term_lin + term_v).mean()


# =========================
# Train
# =========================
def train_zofoGDA_auc_libsvm(cfg: ZOFOGDA_AUCCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_path, test_path = resolve_paths(cfg)
    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        train_path, test_path, standardize=cfg.standardize, seed=cfg.seed
    )
    if K != 2:
        raise ValueError(f"AUC minimax here is binary-only; need K=2 but got K={K}")

    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # p_pos from training set (after mapping labels)
    y_tr_t = torch.as_tensor(y_tr)
    y_tr_pm1 = labels_to_pm1(y_tr_t)
    p_pos = float((y_tr_pm1 > 0).float().mean().item())

    # primal and dual
    primal = AUCPrimalModule(in_dim=D, hidden=cfg.hidden).to(device)
    dual = torch.tensor(0.0, device=device, requires_grad=True)

    def eta_t(t: int) -> float:
        return float(cfg.eta_k) / float((cfg.eta_m + t) ** 0.5)

    def alpha_t(t: int) -> float:
        return float(cfg.c1) * eta_t(t)

    def beta_t(t: int) -> float:
        return float(cfg.c2) * eta_t(t)

    def f_batch(xb: torch.Tensor, yb_pm1: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        s = primal(xb)
        return auc_minimax_loss(s, yb_pm1, primal.a, primal.b, y_scalar, p_pos)

    def g_batch(xb: torch.Tensor, yb_pm1: torch.Tensor, y_scalar: torch.Tensor) -> torch.Tensor:
        return -f_batch(xb, yb_pm1, y_scalar)

    def grad_y_g(xb, yb_pm1):
        if dual.grad is not None:
            dual.grad.zero_()
        g = g_batch(xb, yb_pm1, dual)
        g.backward()
        v_hat = dual.grad.detach().clone()
        dual.requires_grad_(True)
        return v_hat

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
            fp = f_batch(xb, yb_pm1, dual.detach())

            vec_to_params_(primal, xm)
            fm = f_batch(xb, yb_pm1, dual.detach())

            vec_to_params_(primal, x_vec)

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
    log_path = build_log_path(cfg)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_auc", "a", "b", "dual", "w_norm", "vy"])
        # also dump config in header-friendly way (optional)
        writer.writerow(["#cfg", str(asdict(cfg))])

    # save x-hat
    thetas = []
    if cfg.save_all_theta:
        thetas.append({k: v.detach().cpu().clone() for k, v in primal.state_dict().items()})

    # init with first batch
    train_iter = iter(train_loader)
    try:
        xb, yb, _ = next(train_iter)
    except StopIteration:
        raise RuntimeError("Empty train loader.")
    xb = xb.to(device).float()
    yb = yb.to(device)
    yb_pm1 = labels_to_pm1(yb)

    vy = grad_y_g(xb, yb_pm1)
    w = zo_grad_x_f(xb, yb_pm1, q=cfg.zo_q)

    t = 1
    for ep in range(1, cfg.epochs + 1):
        primal.train()
        if ep != 1:
            train_iter = iter(train_loader)

        while True:
            et = eta_t(t)
            at = alpha_t(t)
            bt = beta_t(t)

            # Step 4: x update
            with torch.no_grad():
                x_vec = params_to_vec(primal)
                x_tilde = x_vec - cfg.gamma * w
                x_next = x_vec + et * (x_tilde - x_vec)
                vec_to_params_(primal, x_next)

            # Step 5: y update (interval projection)
            with torch.no_grad():
                y_t = dual.detach()
                y_tilde = proj_interval(y_t - cfg.lam * vy, cfg.v_clip)
                y_next = y_t + et * (y_tilde - y_t)
                dual.copy_(y_next)
            dual.requires_grad_(True)

            if cfg.save_all_theta:
                thetas.append({k: v.detach().cpu().clone() for k, v in primal.state_dict().items()})

            # Step 6/7: next batch
            try:
                xb2, yb2, _ = next(train_iter)
            except StopIteration:
                break
            xb2 = xb2.to(device).float()
            yb2 = yb2.to(device)
            yb2_pm1 = labels_to_pm1(yb2)

            w_hat = zo_grad_x_f(xb2, yb2_pm1, q=cfg.zo_q)
            vy_hat = grad_y_g(xb2, yb2_pm1)

            with torch.no_grad():
                w = (1.0 - bt) * w + bt * w_hat if cfg.x_momentum else w_hat
                vy = (1.0 - at) * vy + at * vy_hat if cfg.y_momentum else vy_hat

            t += 1

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
            f"[{cfg.dataset}] Epoch {ep:03d} | TrainLoss={tr_f:.6f} | TestAUC={te_auc:.4f} "
            f"| a={primal.a.item():.3f}, b={primal.b.item():.3f}, dual(v)={dual.item():.3f}"
        )

    print("Log saved to:", log_path)

    if cfg.save_all_theta and len(thetas) > 0:
        tt = random.randrange(len(thetas))
        primal.load_state_dict(thetas[tt])
        return primal, float(dual.detach().cpu().item()), tt, log_path

    return primal, float(dual.detach().cpu().item()), None, log_path


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser("ZOFOGDA-AUC (LIBSVM) - CLI version")

    # dataset knobs
    p.add_argument("--dataset", type=str, default="svmguide1",
                   choices=["climate", "german", "svmguide1", "spambase", "a9a", "ijcnn1", "covtype", "connect", "acoustic"],
                   help="Dataset name (used to resolve default train/test under data_dir).")
    p.add_argument("--data_dir", type=str, default="./data", help="Directory that stores libsvm files.")
    p.add_argument("--train_path", type=str, default="", help="Override train file path.")
    p.add_argument("--test_path", type=str, default="", help="Override test file path (optional).")

    # model/training
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)

    # schedules
    p.add_argument("--eta_k", type=float, default=10.0)
    p.add_argument("--eta_m", type=float, default=1000.0)
    p.add_argument("--c1", type=float, default=3.0)
    p.add_argument("--c2", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.01)

    # zeroth-order
    p.add_argument("--zo_delta", type=float, default=1e-4)
    p.add_argument("--zo_q", type=int, default=5)
    p.add_argument("--no_x_momentum", action="store_true")
    p.add_argument("--no_y_momentum", action="store_true")

    # dual
    p.add_argument("--v_clip", type=float, default=10.0)

    # misc
    p.add_argument("--standardize", action="store_true", help="Enable standardization (default False if not set).")
    p.add_argument("--no_standardize", action="store_true", help="Disable standardization.")
    p.add_argument("--save_all_theta", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # logging
    p.add_argument("--log_dir", type=str, default="./result")
    p.add_argument("--log_prefix", type=str, default="zofoGDA_auc")

    args = p.parse_args()
    return args


def main():
    args = parse_args()

    # resolve standardize flag
    if args.no_standardize:
        standardize = False
    elif args.standardize:
        standardize = True
    else:
        # keep your original default behavior (True)
        standardize = True

    cfg = ZOFOGDA_AUCCfg(
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

        eta_k=args.eta_k,
        eta_m=args.eta_m,
        c1=args.c1,
        c2=args.c2,
        gamma=args.gamma,
        lam=args.lam,

        zo_delta=args.zo_delta,
        zo_q=args.zo_q,
        x_momentum=not args.no_x_momentum,
        y_momentum=not args.no_y_momentum,

        v_clip=args.v_clip,
        save_all_theta=args.save_all_theta,
        seed=args.seed,
        device=args.device,
    )

    model, dual_v, t, log_path = train_zofoGDA_auc_libsvm(cfg)
    print("Returned x-hat from step:", t)
    print("Final dual(v):", dual_v)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
