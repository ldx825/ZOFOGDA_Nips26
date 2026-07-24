import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@torch.no_grad()
def project_to_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """
    Euclidean projection onto simplex: {v>=0, sum v = z}
    """
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not torch.any(cond):
        return torch.full_like(v, z / v.numel())
    rho = torch.nonzero(cond, as_tuple=False).max().item()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)

def load_svmguide1(train_path: str, test_path: str, standardize: bool = True, seed: int = 42):
    """
    Read LIBSVM format; map labels {+1,-1} -> {1,0}
    """
    X_tr, y_tr = load_svmlight_file(train_path)
    X_tr = X_tr.toarray().astype(np.float32)
    y_tr = (np.asarray(y_tr) > 0).astype(np.int64)

    if test_path is not None and os.path.exists(test_path):
        X_te, y_te = load_svmlight_file(test_path)
        X_te = X_te.toarray().astype(np.float32)
        y_te = (np.asarray(y_te) > 0).astype(np.int64)
    else:
        # fallback split 80/20
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X_tr.shape[0])
        split = int(0.8 * len(idx))
        tr_idx, te_idx = idx[:split], idx[split:]
        X_te, y_te = X_tr[te_idx], y_tr[te_idx]
        X_tr, y_tr = X_tr[tr_idx], y_tr[tr_idx]

    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

    return X_tr, y_tr, X_te, y_te

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
# -------------------------
# LIBSVM loader (supports binary/multiclass, train/test with consistent dims)
# -------------------------
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

def hinge_each_from_logits(logits: torch.Tensor, yb01: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,2)
    yb01:   (B,) in {0,1}
    return: hinge loss per sample (B,)
      hinge_i = relu(1 - b_i * score_i),
      score = logit_pos - logit_neg,
      b in {-1,+1} (class1 -> +1, class0 -> -1)
    """
    score = logits[:, 1] - logits[:, 0]          # (B,)
    b = yb01.float() * 2.0 - 1.0                 # 0->-1, 1->+1
    return F.relu(1.0 - b * score)
# -------------------------
# Dataset with indices
# -------------------------
class IndexedTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx