# dcgan_multi_dataset.py
# DCGAN-style GAN on MNIST/FashionMNIST/CIFAR10/CIFAR100 + visualization + FID/IS evaluation
#
# Run:
#   pip install torch torchvision matplotlib numpy
#   python dcgan_multi_dataset.py
#
# Select dataset by editing CFG.dataset below.

import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from typing import Callable, Iterable, Optional, Tuple, Union, Dict

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import math


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    seed: int = 42

    # NOTE:
    # Better practice: set CUDA_VISIBLE_DEVICES externally and use "cuda:0" here.
    # Example: CUDA_VISIBLE_DEVICES=5 python dcgan_multi_dataset.py
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # choose: "mnist", "fashionmnist", "cifar10", "cifar100"
    dataset: str = "mnist"

    tts: bool = False
    eta_y: float = 1e-3 # y --> D 和adam_的区别是，eta_y 必须大于 eta_x
    eta_x: float = 1e-5 # x --> G

    adam_: bool = False
    lr_y: float = 1e-3  # y --> D
    beta_y: float = (0.5, 0.999) 
    lr_x: float = 1e-3  # x --> G
    beta_x: float = (0.5, 0.999)



    zofo: bool = True
    _gamma: float = 1e-3 # gamma -- x -- G
    _lambda: float = 1e-3 # lambda -- y -- D
    zo_delta: float = 1e-3
    zo_q: int = 1  # 零阶估计的扰动方向数量



    batch_size: int = 128
    num_workers: int = 2

    # GAN
    z_dim: int = 128
    g_feat: int = 64
    d_feat: int = 64
    epochs: int = 30

    # logging/vis
    if adam_ == True:
        out_root: str = "./outputs_dcgan/adam"
    elif zofo == True:
        out_root: str = "./outputs_dcgan/zofo"
    else:
        out_root: str = "./outputs_dcgan/tts"

    sample_every: int = 1

    # metrics
    n_eval_gen: int = 5000
    n_eval_real: int = 5000


cfg = CFG()


def dataset_specs(name: str):
    name = name.lower()
    if name == "mnist":
        return dict(name="MNIST", channels=1, img_size=28)
    if name in ["fashionmnist", "fmnist", "fashion"]:
        return dict(name="FashionMNIST", channels=1, img_size=28)
    if name == "cifar10":
        return dict(name="CIFAR10", channels=3, img_size=32)
    if name == "cifar100":
        return dict(name="CIFAR100", channels=3, img_size=32)
    raise ValueError("dataset must be one of: mnist, fashionmnist, cifar10, cifar100")


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)


def _to_01_from_tanh(x):
    # x in [-1,1] -> [0,1]
    return (x + 1.0) / 2.0


# ----------------------------
# Models (support 1ch/3ch, 28/32)
# Strategy:
# - G always generates 32x32, then interpolate to target (28 or 32)
# - D downsamples 3 times and ends at 4x4 for both 28 and 32
# ----------------------------
class Generator(nn.Module):
    """
    z -> (C, 32, 32) then (optional) resize to target img_size.
    """
    def __init__(self, z_dim=128, feat=64, out_channels=1, img_size=28):
        super().__init__()
        self.img_size = img_size

        # 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(feat),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat, feat // 2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(feat // 2),
            nn.ReLU(True),

            nn.Conv2d(feat // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)  # (N,C,32,32)
        if x.shape[-1] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            )
        return x


class Discriminator(nn.Module):
    """
    x -> prob(real)
    Downsample:
      conv k4 s2 p1 : 32->16, 28->14
      conv k4 s2 p1 : 16->8,  14->7
      conv k3 s2 p1 : 8->4,   7->4  (this is the trick to unify)
    """
    def __init__(self, feat=64, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat * 2, feat * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),                 # -> (N, feat*4*4*4)
            nn.Linear((feat * 4) * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.features(x)
        return self.head(h)


# ----------------------------
# Data
# ----------------------------
def get_loaders(dataset_name: str, img_size: int, channels: int):
    if channels == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    tfm = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),  # -> [-1,1]
    ])

    root = "./data"
    name = dataset_name.lower()

    if name == "mnist":
        train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    elif name in ["fashionmnist", "fmnist", "fashion"]:
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)
    elif name == "cifar10":
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    elif name == "cifar100":
        train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
        test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# Visualization helpers
# ----------------------------
@torch.no_grad()
def sample_grid(G, epoch, fixed_z, out_dir, up_size=128):
    G.eval()
    fake = G(fixed_z).cpu()
    # 仅用于可视化：把 32x32 放大到 up_size x up_size
    fake_up = torch.nn.functional.interpolate(
        fake, size=(up_size, up_size),
        mode="nearest"  # 更像素风/更清晰；想平滑可改 bilinear
    ).cpu()

    grid = make_grid(fake_up, nrow=8, normalize=True, value_range=(-1, 1), padding=2)
    path = os.path.join(out_dir, f"samples_epoch_{epoch:03d}_up{up_size}.png")
    save_image(grid, path)
    return path


def plot_losses(d_losses, g_losses, out_dir):
    plt.figure()
    plt.plot(d_losses, label="D loss")
    plt.plot(g_losses, label="G loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ----------------------------
# Metrics: FID + Inception Score (InceptionV3)
# For 1-channel datasets we repeat to 3 channels.
# ----------------------------
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        self.preprocess = weights.transforms()

        # torchvision 0.23 + weights: aux_logits must be True
        self.inception = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        self.inception.eval().to(device)

        if hasattr(self.inception, "AuxLogits") and self.inception.AuxLogits is not None:
            self.inception.AuxLogits = None

        self.device = device

    @torch.no_grad()
    def features(self, x01):
        """
        x01: (N,C,H,W) in [0,1], C can be 1 or 3
        return: (N,2048) pool3 features
        """
        if x01.size(1) == 1:
            x01 = x01.repeat(1, 3, 1, 1)
        x3 = self.preprocess(x01).to(self.device)

        m = self.inception
        x = m.Conv2d_1a_3x3(x3)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    @torch.no_grad()
    def class_probs(self, x01):
        """
        Return softmax probabilities for Inception Score.
        """
        if x01.size(1) == 1:
            x01 = x01.repeat(1, 3, 1, 1)
        x3 = self.preprocess(x01).to(self.device)
        out = self.inception(x3)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        return torch.softmax(out, dim=1)


def compute_stats(feats: torch.Tensor):
    f = feats.detach().cpu().double().numpy()
    mu = np.mean(f, axis=0)
    sigma = np.cov(f, rowvar=False)
    return mu, sigma


def sqrtm_psd(mat: np.ndarray, eps=1e-10):
    mat = (mat + mat.T) / 2.0
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, a_min=eps, a_max=None)
    return (v * np.sqrt(w)) @ v.T


def fid_from_stats(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    diff_sq = diff @ diff
    prod = (s1 @ s2)
    prod = (prod + prod.T) / 2.0
    sqrt_prod = sqrtm_psd(prod)
    fid = diff_sq + np.trace(s1 + s2 - 2.0 * sqrt_prod)
    return float(np.real(fid))


@torch.no_grad()
def compute_fid(G, real_loader, device, n_real=5000, n_gen=5000):
    extractor = InceptionFeatureExtractor(device)

    # Real
    real_feats = []
    count = 0
    for x, _ in real_loader:
        x01 = _to_01_from_tanh(x)
        feats = extractor.features(x01.to(device))
        real_feats.append(feats.cpu())
        count += x.size(0)
        if count >= n_real:
            break
    real_feats = torch.cat(real_feats, dim=0)[:n_real]
    mu_r, sig_r = compute_stats(real_feats)

    # Gen
    gen_feats = []
    remaining = n_gen
    while remaining > 0:
        b = min(cfg.batch_size, remaining)
        z = torch.randn(b, cfg.z_dim, 1, 1, device=device)
        fake = G(z)
        fake01 = _to_01_from_tanh(fake)
        feats = extractor.features(fake01)
        gen_feats.append(feats.cpu())
        remaining -= b
    gen_feats = torch.cat(gen_feats, dim=0)[:n_gen]
    mu_g, sig_g = compute_stats(gen_feats)

    return fid_from_stats(mu_r, sig_r, mu_g, sig_g)


@torch.no_grad()
def compute_inception_score(G, device, n_gen=5000, splits=10):
    extractor = InceptionFeatureExtractor(device)

    probs_list = []
    remaining = n_gen
    while remaining > 0:
        b = min(cfg.batch_size, remaining)
        z = torch.randn(b, cfg.z_dim, 1, 1, device=device)
        fake = G(z)
        fake01 = _to_01_from_tanh(fake)
        p = extractor.class_probs(fake01).cpu()
        probs_list.append(p)
        remaining -= b

    probs = torch.cat(probs_list, dim=0)[:n_gen].numpy()

    split_scores = []
    N = probs.shape[0]
    split_size = N // splits
    for i in range(splits):
        part = probs[i * split_size: (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        kl = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl)))
    return float(np.mean(split_scores)), float(np.std(split_scores))


class AdamW_x(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        gamma: float = 1e-3,
        eta_t: float = 1e-3, 
        beta: float = 0.7,
        eps: float = 1e-6,
    ):
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma} - should be >= 0.0")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"gamma": gamma, "eta_t": eta_t, "beta": beta, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        # print(f"use_x")
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad ** 2

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                denom = torch.tensor(1.0, device=p.device, dtype=p.dtype)

                step_size = group["gamma"] * group["eta_t"]

                p.addcdiv_(exp_avg, denom, value=-step_size)

                beta = group["beta"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(1 - beta).add_(grad, alpha=(beta))
                # exp_avg_sq.mul_(1.0 - mu).addcmul_(grad, grad, value=mu)
                # exp_avg_sq = 1
                # denom = exp_avg_sq.sqrt().add(group["eps"])
                # denom = 1
                # denom = torch.tensor(1.0, device=p.device, dtype=p.dtype)

                # step_size = group["gamma"] * group["eta_t"]

                # p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
class AdamW_y(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        _lambda: float = 1e-3,
        eta_t: float = 1e-3, 
        alpha: float = 0.7,
        eps: float = 1e-6,
    ):
        if eta_t < 0.0:
            raise ValueError(f"Invalid eta_t: {eta_t} - should be >= 0.0")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"_lambda": _lambda, "eta_t": eta_t, "alpha": alpha, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        # print(f"use_y")
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.norm(grad, p=2)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                denom = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                
                p.addcdiv_(exp_avg, denom, value=-group["_lambda"] * group["eta_t"])

                alpha = group["alpha"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(1 - alpha).add_(grad, alpha=(alpha))
                # exp_avg_sq.mul_(1.0 - mu).add_(torch.norm(grad, p=2), alpha=(mu))
                # denom = exp_avg_sq.add_(group["eps"])
                # denom = 1
                # denom = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                
                # p.addcdiv_(exp_avg, denom, value=-group["_lambda"] * group["eta_t"])

        return loss
    

def zero_order_gradient_estimate(
    G: nn.Module,
    D: nn.Module,
    z: torch.Tensor,
    batch_size: int,
    criterion: nn.Module,
    delta: float = 1e-3,
    q: int = 1,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    使用零阶方法估计生成器G的梯度
    
    参数:
        G: 生成器模型
        D: 判别器模型
        z: 输入噪声 (batch_size, z_dim, 1, 1)
        criterion: 损失函数 (BCELoss)
        delta: 扰动大小
        q: 扰动方向数量
        device: 设备
    
    返回:
        grad_est_dict: 参数名到梯度估计的字典
    """
    G.eval()
    D.eval()
    
    # 保存原始参数
    original_params = {}
    for name, param in G.named_parameters():
        original_params[name] = param.data.clone()
        param.requires_grad = True
    
    # 生成真实标签
    # batch_size = z.size(0)
    label_real = torch.ones(batch_size, 1, device=device)
    
    # 计算原始损失 f(x, y)
    with torch.no_grad():
        fake = G(z)
        out = D(fake)
        f_original = criterion(out, label_real)
    
    # 初始化梯度估计字典
    grad_est_dict = {}
    
    # 对于每个参数，计算零阶梯度估计
    for name, param in G.named_parameters():
        if not param.requires_grad:
            continue
            
        param_dim = param.numel()  # 参数维度 d1
        shape = param.shape
        
        # 初始化梯度估计
        grad_est = torch.zeros_like(param)
        
        # 对每个扰动方向
        for _ in range(q):
            # 采样随机向量 u_i，从单位L2球面上均匀采样
            # 生成标准正态分布然后归一化到单位球面
            u = torch.randn_like(param)
            u_norm = torch.norm(u.view(-1), p=2)
            if u_norm > 0:
                u = u / u_norm  # 归一化到单位球面
            
            # 保存随机种子，确保正负扰动使用相同的u
            random_seed = torch.seed()
            
            # 计算 f(x + δu, y)
            torch.manual_seed(random_seed)
            param.data = original_params[name] + delta * u
            with torch.no_grad():
                fake_plus = G(z)
                out_plus = D(fake_plus)
                f_plus = criterion(out_plus, label_real)
            
            # 计算 f(x - δu, y)
            torch.manual_seed(random_seed)
            param.data = original_params[name] - delta * u
            with torch.no_grad():
                fake_minus = G(z)
                out_minus = D(fake_minus)
                f_minus = criterion(out_minus, label_real)
            
            # 计算梯度估计：d1/(2δ) * (f(x+δu) - f(x-δu)) * u
            grad_est += (param_dim / (2 * delta)) * (f_plus - f_minus) * u
        
        # 平均多个扰动方向的估计
        grad_est = grad_est / q
        grad_est_dict[name] = grad_est
    
    # 恢复原始参数
    for name, param in G.named_parameters():
        if name in original_params:
            param.data = original_params[name]
    
    return grad_est_dict, f_original


def set_adamw_hyper_x(optim, eta_t=None, beta=None):
    for g in optim.param_groups:
        if beta is not None:
            g['beta'] = float(beta)
        if eta_t is not None:
            g['eta_t'] = float(eta_t)


def set_adamw_hyper_y(optim, eta_t=None, alpha=None):
    for g in optim.param_groups:
        if alpha is not None:
            g['alpha'] = float(alpha)
        if eta_t is not None:
            g['eta_t'] = float(eta_t)
# ----------------------------
# Train
# ----------------------------
def train():
    set_seed(cfg.seed)

    specs = dataset_specs(cfg.dataset)
    out_dir = os.path.join(cfg.out_root, cfg.dataset.lower())
    if cfg.tts == True:
        out_dir = os.path.join(out_dir, f"eta_y:{cfg.eta_y}-eta_x:{cfg.eta_x}-epochs:{cfg.epochs}")
    elif cfg.adam_ == True:
        out_dir = os.path.join(out_dir, f"lr_y:{cfg.lr_y}-beta_y:{cfg.beta_y}-lr_x:{cfg.lr_x}-beta_x:{cfg.beta_x}-epochs:{cfg.epochs}")
    else:
        out_dir = os.path.join(out_dir, f"_gamma:{cfg._gamma}-_lambda:{cfg._lambda}-zo_delta:{cfg.zo_delta}-zo_q:{cfg.zo_q}-epochs:{cfg.epochs}")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, test_loader = get_loaders(cfg.dataset, specs["img_size"], specs["channels"])

    G = Generator(cfg.z_dim, cfg.g_feat, out_channels=specs["channels"], img_size=specs["img_size"]).to(cfg.device)
    D = Discriminator(cfg.d_feat, in_channels=specs["channels"]).to(cfg.device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    if cfg.adam_ == True:
        optG = optim.Adam(G.parameters(), lr=cfg.lr_x, betas=cfg.beta_x)
        optD = optim.Adam(D.parameters(), lr=cfg.lr_y, betas=cfg.beta_y)
    elif cfg.tts == True:
        optG = optim.SGD(G.parameters(), lr=cfg.eta_x, momentum=0.0) # x
        optD = optim.SGD(D.parameters(), lr=cfg.eta_y, momentum=0.0) # y
    elif cfg.zofo == True:
        optG = AdamW_x(G.parameters(), gamma=cfg._gamma) # x
        optD = AdamW_y(D.parameters(), _lambda=cfg._lambda) # y

    fixed_z = torch.randn(64, cfg.z_dim, 1, 1, device=cfg.device)

    d_losses, g_losses = [], []

    t = 0
    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        t0 = time.time()

        for real, _ in train_loader:
            real = real.to(cfg.device)
            b = real.size(0)

            # Train D
            D.zero_grad(set_to_none=True)
            label_real = torch.ones(b, 1, device=cfg.device)
            label_fake = torch.zeros(b, 1, device=cfg.device)

            out_real = D(real)
            loss_d_real = criterion(out_real, label_real)

            z = torch.randn(b, cfg.z_dim, 1, 1, device=cfg.device)
            fake = G(z).detach()
            out_fake = D(fake)
            loss_d_fake = criterion(out_fake, label_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            if cfg.zofo == False:
                optD.step()
            else:
                eta_t = 10 / (math.sqrt(1000 + t))
                set_adamw_hyper_y(optD, eta_t=eta_t, alpha=3 * eta_t)
                optD.step()

            if cfg.zofo == False:
                # Train G
                G.zero_grad(set_to_none=True)
                z = torch.randn(b, cfg.z_dim, 1, 1, device=cfg.device)
                fake = G(z)
                out = D(fake)
                loss_g = criterion(out, label_real)
                loss_g.backward()
                optG.step()
            else:
                # 采用零阶方法估计对x（即G）的梯度
                # pass
                G.zero_grad(set_to_none=True)
                z = torch.randn(b, cfg.z_dim, 1, 1, device=cfg.device)
                grad_est_dict, loss_g = zero_order_gradient_estimate(
                    G, D, z, b, criterion, 
                    delta=cfg.zo_delta, 
                    q=cfg.zo_q,
                    device=cfg.device
                )
                with torch.no_grad():
                    for name, param in G.named_parameters():
                        param.grad = grad_est_dict[name]
                eta_t = 10 / (math.sqrt(1000 + t))
                set_adamw_hyper_x(optG, eta_t=eta_t, beta=3 * eta_t)
                optG.step()

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            t += 1

        if epoch % cfg.sample_every == 0:
            sample_path = sample_grid(G, epoch, fixed_z, out_dir)
            print(f"[{cfg.dataset}] [Epoch {epoch}/{cfg.epochs}] saved samples: {sample_path}")

        print(f"[{cfg.dataset}] [Epoch {epoch}/{cfg.epochs}] time={time.time()-t0:.1f}s "
              f"D_loss={np.mean(d_losses[-len(train_loader):]):.4f} "
              f"G_loss={np.mean(g_losses[-len(train_loader):]):.4f}")

    loss_path = plot_losses(d_losses, g_losses, out_dir)
    print(f"Saved loss curve: {loss_path}")

    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pt"))
    print("Saved models.")

    # Evaluation
    print("\nComputing metrics (this may take a bit)...")
    G.eval()
    fid = compute_fid(G, test_loader, cfg.device, n_real=cfg.n_eval_real, n_gen=cfg.n_eval_gen)
    is_mean, is_std = compute_inception_score(G, cfg.device, n_gen=cfg.n_eval_gen, splits=10)

    print(f"FID (real test vs generated) = {fid:.4f}   (lower is better)")
    print(f"Inception Score = {is_mean:.4f} ± {is_std:.4f}   (higher is better; MNIST上参考意义有限)")

    # Final visualization
    with torch.no_grad():
        z = torch.randn(64, cfg.z_dim, 1, 1, device=cfg.device)
        fake = G(z)

        # CIFAR(32x32) 建议放大再拼图；MNIST(28x28)也可以放大
        up_size = 128 if specs["img_size"] <= 32 else specs["img_size"]
        fake_up = torch.nn.functional.interpolate(fake, size=(up_size, up_size), mode="nearest").cpu()

        grid = make_grid(fake_up, nrow=8, normalize=True, value_range=(-1, 1), padding=2)

    # 注意：彩色图 (H,W,3) OK；单通道 (H,W,1) 建议特殊处理
    img = np.transpose(grid.numpy(), (1, 2, 0))
    plt.figure(figsize=(12, 12))   # 原来 6x6 太小
    plt.title(f"Generated samples (final) - {cfg.dataset}")
    plt.axis("off")
    if img.shape[2] == 1:
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img)

    final_vis_path = os.path.join(out_dir, f"final_grid_matplotlib_up{up_size}.png")
    plt.tight_layout()
    plt.savefig(final_vis_path, dpi=300)  # 原来 150 太低
    plt.show()
    print(f"Saved final visualization: {final_vis_path}")



if __name__ == "__main__":
    train()
