import torch
from utils import set_seed
from utils import IndexedTabularDataset, project_to_simplex, load_libsvm_dataset, hinge_each_from_logits
from torch.utils.data import DataLoader
from dataclasses import dataclass
from model import SimpleMLP
import torch.nn.functional as F
import random
import csv
import time


# -------------------------
# config: ZOFOGDA
# -------------------------
@dataclass
class ZOFOGDACfg:
    # data
    train_path: str = "./data/svmguide1"
    test_path: str = "./data/svmguide1.t"
    standardize: bool = True
    log_path: str = "zofoGDA_svmguide1"

    # model
    hidden: int = 128

    # training
    epochs: int = 200
    batch_size: int = 100

    # ===== schedules =====
    # eta_t = k / sqrt(m + t)
    eta_k: float = 1.0
    eta_m: float = 100.0   # m>0 to avoid huge step at t=1

    # alpha_{t+1} = c1 * eta_t, beta_{t+1} = c2 * eta_t
    c1: float = 0.1
    c2: float = 0.1

    # Algorithm 1 hyperparams
    # x update: x̃_{t+1} = x_t - γ w_t; x_{t+1} = x_t + η_t (x̃_{t+1} - x_t)
    gamma: float = 1e-2        # γ

    # y prox update:
    # ỹ_{t+1} = argmin_{y∈Y} <v_t, y> + (1/2λ)||y - y_t||^2  => Π_Y(y_t - λ v_t)
    # y_{t+1} = y_t + η_t (ỹ_{t+1} - y_t)
    lam: float = 1.0           # λ

    # Zeroth-order gradient settings for x
    zo_delta: float = 1e-3     # δ smoothing radius
    zo_q: int = 1              # number of random directions per step (q=1 is common)

    # y regularization (mu in your formula):  (mu/2)||y - 1/N||^2
    mu: float = 0.001

    # saving x-hat
    save_all_theta: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42




# -------------------------
# helpers: flatten / set parameters (for ZO gradient)
# -------------------------
@torch.no_grad()
def params_to_vec(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p for p in model.parameters()]).detach()

@torch.no_grad()
def vec_to_params_(model: torch.nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, [p for p in model.parameters()])

@torch.no_grad()
def sample_unit_vec(d: int, device: torch.device) -> torch.Tensor:
    u = torch.randn(d, device=device)
    u = u / (u.norm() + 1e-12)
    return u


# -------------------------
# Main: ZOFOGDA
# -------------------------
def train_zofoGDA_libsvm(cfg: ZOFOGDACfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        cfg.train_path, cfg.test_path, standardize=cfg.standardize, seed=cfg.seed
    )
    if K != 2:
        raise ValueError(
            f"This implementation uses binary hinge; need K=2 but got K={K}. "
            f"If you want multiclass, switch to Crammer-Singer hinge."
        )

    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    N = len(train_ds)

    model = SimpleMLP(in_dim=D, hidden=cfg.hidden, num_classes=K).to(device)

    # y in simplex
    y = torch.full((N,), 1.0 / N, device=device, dtype=torch.float32, requires_grad=True)

    def reg_value(yw: torch.Tensor) -> torch.Tensor:
        if cfg.mu <= 0:
            return torch.tensor(0.0, device=yw.device, dtype=yw.dtype)
        return 0.5 * cfg.mu * torch.sum((yw - (1.0 / N)) ** 2)

    # f(x,y;B): minibatch stochastic objective for min_x max_y
    # f = mean( y_i * hinge_i ) - reg(y)
    def f_batch(xb: torch.Tensor, yb: torch.Tensor, idx: torch.Tensor, y_full: torch.Tensor) -> torch.Tensor:
        logits = model(xb)
        hinge_each = hinge_each_from_logits(logits, yb)     # (B,)
        y_batch = y_full[idx]                               # (B,)
        return torch.mean(y_batch * hinge_each) - reg_value(y_full)

    # g = -f  (important!)
    def g_batch(xb: torch.Tensor, yb: torch.Tensor, idx: torch.Tensor, y_full: torch.Tensor) -> torch.Tensor:
        return -f_batch(xb, yb, idx, y_full)

    @torch.no_grad()
    def eval_acc(loader):
        model.eval()
        correct, total = 0, 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def full_train_obj():
        """
        Full objective for reporting:
          F(x,y) = (1/N) Σ_i y_i * hinge_i(x) - reg(y)
        """
        model.eval()
        loss_all = torch.empty(N, device=device)
        full_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        for xb, yb, idx in full_loader:
            xb, yb, idx = xb.to(device), yb.to(device), idx.to(device)
            logits = model(xb)
            hinge = hinge_each_from_logits(logits, yb)
            loss_all[idx] = hinge
        term1 = torch.sum(y.detach() * loss_all) / N
        term2 = reg_value(y.detach())
        return (term1 - term2).item()

    
    def eta_t(t: int) -> float:
        # eta_t = k / sqrt(m + t)
        return float(cfg.eta_k) / float((cfg.eta_m + t) ** 0.5)

    def alpha_t(t: int) -> float:
        # alpha_{t+1} = c1 * eta_t  (we use eta_t(t) for the step at iteration t)
        return float(cfg.c1) * eta_t(t)

    def beta_t(t: int) -> float:
        return float(cfg.c2) * eta_t(t)
    
    # logging
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"{cfg.log_path}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_obj", "test_acc", "y_min", "y_max"])

    # save thetas for x-hat
    thetas = []
    if cfg.save_all_theta:
        thetas.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # -------------------------
    # Initialize v1, w1 using first batch B1
    # -------------------------
    train_iter = iter(train_loader)
    try:
        xb, yb, idx = next(train_iter)
    except StopIteration:
        raise RuntimeError("Empty train loader.")
    xb, yb, idx = xb.to(device), yb.to(device), idx.to(device)

    # v1 = ∇_y g(x1,y1;B1)  (first-order)
    # w1 = ∇_x f(x1,y1;B1)  (zeroth-order estimate)
    def grad_y_g(xb, yb, idx):
        # compute v_hat = ∇_y g, where g = -f
        if y.grad is not None:
            y.grad.zero_()
        g = g_batch(xb, yb, idx, y)    # scalar
        g.backward()
        v_hat = y.grad.detach().clone()
        y.requires_grad_(True)
        return v_hat

    @torch.no_grad()
    def zo_grad_x_f(xb, yb, idx, q=1):
        """
        Zeroth-order gradient estimator for ∇_x f (x is model params).
        Two-point random direction estimator:
          grad ≈ (d / (2δ)) * (f(x+δu)-f(x-δu)) * u
        """
        x_vec = params_to_vec(model)
        d = x_vec.numel()
        grad_est = torch.zeros_like(x_vec)

        for _ in range(q):
            u = sample_unit_vec(d, device=device)
            xp = x_vec + cfg.zo_delta * u
            xm = x_vec - cfg.zo_delta * u

            vec_to_params_(model, xp)
            fp = f_batch(xb, yb, idx, y.detach())  # detach y for x-gradient estimation
            vec_to_params_(model, xm)
            fm = f_batch(xb, yb, idx, y.detach())

            # restore original
            vec_to_params_(model, x_vec)

            # scalar -> grad vector
            coeff = (d / (2.0 * cfg.zo_delta)) * (fp - fm)
            grad_est += coeff * u

        grad_est /= float(q)
        return grad_est

    v = grad_y_g(xb, yb, idx)                    # v1
    w = zo_grad_x_f(xb, yb, idx, q=cfg.zo_q)     # w1

    # -------------------------
    # Main loop: over epochs/batches as iterations t
    t = 1
    # -------------------------
    for ep in range(1, cfg.epochs + 1):
        model.train()

        # we already consumed one batch for initialization if ep==1,
        # so we continue from the current iterator; on later epochs reset iterator.
        if ep != 1:
            train_iter = iter(train_loader)

        while True:
            # current batch (B_t) already loaded in xb,yb,idx for ep==1 first iter;
            # for other iters load from iterator
            if ep != 1 or (ep == 1 and xb is None):
                try:
                    xb, yb, idx = next(train_iter)
                    xb, yb, idx = xb.to(device), yb.to(device), idx.to(device)
                except StopIteration:
                    break
            
            # current steps
            et = eta_t(t)
            at = alpha_t(t)
            bt = beta_t(t)
            
            # ---- Step 4 (x update):  x̃_{t+1} = x_t - γ w_t;  x_{t+1} = x_t + η( x̃_{t+1} - x_t )
            with torch.no_grad():
                x_vec = params_to_vec(model)
                x_tilde = x_vec - cfg.gamma * w
                x_next = x_vec + et * (x_tilde - x_vec)   # = x_vec - eta*gamma*w
                vec_to_params_(model, x_next)

            # ---- Step 5 (y update): ỹ_{t+1} = Π_Y(y_t - λ v_t), then relaxation
            # Note: v_t = ∇_y g, g=-f => y_t - λ v_t = y_t + λ ∇_y f  (ascent on f)
            with torch.no_grad():
                y_t = y.detach()
                y_tilde = project_to_simplex(y_t - cfg.lam * v, z=1.0)
                y_next = y_t + et * (y_tilde - y_t)
                y.copy_(y_next)
            y.requires_grad_(True)

            if cfg.save_all_theta:
                thetas.append({k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()})

            # ---- Step 6/7: compute w_{t+1}, v_{t+1} with new batch B_{t+1}
            try:
                xb2, yb2, idx2 = next(train_iter)
                xb2, yb2, idx2 = xb2.to(device), yb2.to(device), idx2.to(device)
            except StopIteration:
                # no next batch -> end epoch
                xb = None
                break

            # w_hat = ∇_x f_δ (x_{t+1}, y_{t+1}; B_{t+1})  (ZO estimator)
            w_hat = zo_grad_x_f(xb2, yb2, idx2, q=cfg.zo_q)

            # v_hat = ∇_y g (x_{t+1}, y_{t+1}; B_{t+1})  (first-order)
            v_hat = grad_y_g(xb2, yb2, idx2)

            # momentum updates:
            with torch.no_grad():
                w = (1.0 - bt) * w + bt * w_hat
                v = (1.0 - at) * v + at * v_hat

            # advance current batch pointer
            xb, yb, idx = xb2, yb2, idx2

        # epoch-end logging
        tr_obj = full_train_obj()
        te_acc = eval_acc(test_loader)
        with torch.no_grad():
            y_min, y_max = y.min().item(), y.max().item()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, tr_obj, te_acc, y_min, y_max])

        print(
            f"Epoch {ep:03d} | TrainObj={tr_obj:.6f} | TestAcc={te_acc*100:.2f}% "
            f"| y[min,max]=({y_min:.3e},{y_max:.3e})"
        )

    print("Log saved to:", log_path)

    # return x-hat uniformly from stored iterates
    if cfg.save_all_theta and len(thetas) > 0:
        t = random.randrange(len(thetas))
        model.load_state_dict(thetas[t])
        return model, y.detach().cpu(), t, log_path
    else:
        return model, y.detach().cpu(), None, log_path


if __name__ == "__main__":
    cfg = ZOFOGDACfg(
        train_path="./data/svmguide1",
        test_path="./data/svmguide1.t",
        log_path="./result/zofoGDA_svmguide1",

        # train_path="./data/splice",
        # test_path="./data/splice.t",
        # log_path="./result/zofoGDA_splice",

        standardize=True,
        hidden=128,

        epochs=500,
        batch_size=128,

        # schedules
        eta_k=10.0,
        eta_m=1000.0,
        c1=3.,
        c2=3.,
        gamma=1.0,
        lam=1.0,

        zo_delta=1e-4,
        zo_q=5,

        mu=0.001,
        save_all_theta=True,
    )

    model, y, t, log_path = train_zofoGDA_libsvm(cfg)
    print("Returned x-hat from step:", t)
    print("Log:", log_path)
