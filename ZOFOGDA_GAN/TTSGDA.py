import torch
from utils import set_seed
from utils import load_svmguide1, IndexedTabularDataset, project_to_simplex, load_libsvm_dataset, hinge_each_from_logits
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from model import SimpleMLP
import torch.nn.functional as F 
import random
import csv
import time
# -------------------------
# TT-SGDA config
# -------------------------
@dataclass
class TTSCfg:
    # data
    train_path: str = "./data/svmguide1"
    test_path: str  = "./data/svmguide1.t"
    standardize: bool = True

    log_path: str = "ttsgda_svmguide1"
    # model
    hidden: int = 64

    # TTSGDA hyperparams (ηx << ηy)
    epochs: int = 50
    batch_size: int = 256
    lr_x: float = 1e-3     # η_x
    lr_y: float = 5e-2     # η_y  (should be much larger)
    y_steps_per_batch: int = 1  # can be >1

    # optional y regularization (your earlier code had this)
    theta1: float = 0.0  # set 0.0 if you want EXACT simplex-only minimax

    # returning x-hat
    save_all_theta: bool = True  # if False, return final theta (not strictly per pseudo-code)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -------------------------
# TT-SGDA core (matches Algorithm 2)
# -------------------------
def train_ttsgda_libsvm(cfg: TTSCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    X_tr, y_tr, X_te, y_te, K, D = load_libsvm_dataset(
        cfg.train_path, cfg.test_path, standardize=cfg.standardize, seed=cfg.seed
    )

    train_ds = IndexedTabularDataset(X_tr, y_tr)
    test_ds = IndexedTabularDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    N = len(train_ds)

    model = SimpleMLP(in_dim=D, hidden=cfg.hidden, num_classes=K).to(device)
    opt_x = torch.optim.SGD(model.parameters(), lr=cfg.lr_x)

    # y \in simplex, uniform init
    y = torch.full((N,), 1.0 / N, device=device, dtype=torch.float32, requires_grad=True)

    def reg_value(yw: torch.Tensor) -> torch.Tensor:
        if cfg.theta1 <= 0:
            return torch.tensor(0.0, device=yw.device, dtype=yw.dtype)
        # (theta1/2) || N y - 1 ||^2
        return 0.5 * cfg.theta1 * torch.sum((yw - (1.0 / N)) ** 2)
    
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
        F(x,y) = (1/N) Σ_i y_i * CE_i(x) - reg(y)
        """
        model.eval()
        ce_all = torch.empty(N, device=device)
        full_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        for xb, yb, idx in full_loader:
            xb, yb, idx = xb.to(device), yb.to(device), idx.to(device)
            # ce = F.cross_entropy(model(xb), yb, reduction="none")
            logits = model(xb)
            ce = hinge_each_from_logits(logits, yb)
            ce_all[idx] = ce
        term1 = torch.sum(y * ce_all) / N
        term2 = reg_value(y)
        return (term1 - term2).item()

    # # store θ_t to sample x-hat uniformly from {x_t}_{t=0}^T
    thetas = []
    if cfg.save_all_theta:
        thetas.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    # -------------------------
    # Logging to CSV
    # -------------------------
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log = cfg.log_path
    log_path = f"{log}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_obj", "test_acc", "y_min", "y_max"])
    # -------------------------
    # TT-SGDA main loop
    # -------------------------
    # Pseudocode:
    # (g_x, g_y) = SG(G, {ξ_i}^M, x_{t-1}, y_{t-1})
    # x_t = x_{t-1} - η_x g_x
    # y_t = P_Y( y_{t-1} + η_y g_y )
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, idx in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            idx = idx.to(device)
            B = xb.size(0)
            if y.grad is not None: 
                y.grad.zero_()
            opt_x.zero_grad()
            logits = model(xb)
            # ce_each = F.cross_entropy(logits, yb, reduction="none")  # (B,)
            hinge_each = hinge_each_from_logits(logits, yb)
            y_batch = y[idx]
            loss = torch.mean(y_batch * hinge_each) - reg_value(y)
            loss.backward()
            opt_x.step()
            with torch.no_grad():
                y += cfg.lr_y * y.grad 
                y.copy_(project_to_simplex(y, z=1.0))
            y.requires_grad_(True)

            if cfg.save_all_theta:
                thetas.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        # epoch report
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

    # -------------------------
    # Return x-hat: randomly draw from {x_t}_{t=0}^T uniformly
    # -------------------------
    if cfg.save_all_theta and len(thetas) > 0:
        t = random.randrange(len(thetas))
        model.load_state_dict(thetas[t])
        # you也可以返回 t 方便复现实验
        return model, y.detach().cpu(), t
    else:
        return model, y.detach().cpu(), None


if __name__ == "__main__":
    cfg = TTSCfg(
        # 换成你自己的 libsvm 文件即可（支持二分类/多分类）
        train_path="./data/svmguide1",
        test_path="./data/svmguide1.t",
        log_path="./result/ttsgda_svmguide1",
        
        # train_path="./data/splice",
        # test_path="./data/splice.t",
        # log_path="./result/ttsgda_splice",
        
        standardize=True,
        hidden=128,

        # ηx << ηy
        lr_x=1e-1,
        lr_y=1e-0,

        epochs=500,
        batch_size=128,
        y_steps_per_batch=1,

        # EXACT simplex-only minimax: theta1=0
        theta1=0.001,

        # 伪代码要求从 {x_t} 均匀随机抽一个返回
        save_all_theta=True,
    )

    model, y, t = train_ttsgda_libsvm(cfg)
    print("Returned x-hat from step:", t)