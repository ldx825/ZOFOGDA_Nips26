import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import random
import time
import os
from tqdm import tqdm
import csv
from pathlib import Path
from model import Generator, Discriminator
# Save history to CSV
import pandas as pd


@dataclass
class ZOFOGDA_GAN_Cfg:
    # Data configuration
    dataset: str = "mnist"  # "mnist", "fashionmnist", "cifar10", "cifar100"
    data_dir: str = "./data"
    img_size: int = 32
    num_channels: int = 1  # Will be updated based on dataset
    
    # Model configuration
    latent_dim: int = 100
    g_hidden_dim: int = 128
    d_hidden_dim: int = 128
    use_batch_norm: bool = True
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    
    # ZOFOGDA algorithm hyperparameters
    # η_t = k / sqrt(m + t)
    eta_k: float = 1.0
    eta_m: float = 100.0
    
    # α_{t+1} = c1 * η_t, β_{t+1} = c2 * η_t
    c1: float = 0.1
    c2: float = 0.1
    
    # Algorithm hyperparameters
    gamma: float = 1e-2  # γ for x update
    lam: float = 1.0     # λ for y update (projection parameter)
    
    # Zeroth-order gradient settings
    zo_delta: float = 1e-3  # δ smoothing radius
    zo_q: int = 1           # number of random directions
    
    # GAN specific
    n_critic: int = 1  # Number of discriminator updates per generator update
    
    # Evaluation
    eval_freq: int = 5
    sample_freq: int = 10
    save_model: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Helper functions for ZO gradient
# -------------------------
@torch.no_grad()
def params_to_vec(model: nn.Module) -> torch.Tensor:
    """Convert model parameters to a single vector"""
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


@torch.no_grad()
def vec_to_params(vec: torch.Tensor, model: nn.Module):
    """Convert vector back to model parameters"""
    params = [p for p in model.parameters() if p.requires_grad]
    pointer = 0
    for p in params:
        num_elements = p.numel()
        p.data.copy_(vec[pointer:pointer + num_elements].view_as(p))
        pointer += num_elements


@torch.no_grad()
def sample_unit_vec(d: int, device: torch.device) -> torch.Tensor:
    """Sample a random unit vector"""
    u = torch.randn(d, device=device)
    u = u / (u.norm() + 1e-12)
    return u


@torch.no_grad()
def project_to_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Project vector onto simplex: Δ = {x | x ≥ 0, Σx = z}"""
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    rho = torch.nonzero(u * torch.arange(1, n_features + 1, device=v.device) > (cssv - z))[-1]
    theta = (cssv[rho] - z) / (rho.float() + 1.0)
    w = torch.clamp(v - theta, min=0)
    return w


# -------------------------
# ZOFOGDA Algorithm Implementation
# -------------------------
class ZOFOGDA_GAN:
    """Implement ZOFOGDA algorithm for GAN training"""
    
    def __init__(self, generator: nn.Module, discriminator: nn.Module, cfg: ZOFOGDA_GAN_Cfg):
        self.generator = generator
        self.discriminator = discriminator
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize momentum vectors
        self.w = None  # For generator (x)
        self.v = None  # For discriminator (y)
        
        # Tracking
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'grad_norm': [],
            'real_score': [],
            'fake_score': []
        }
        
    def eta_t(self, t: int) -> float:
        """Compute η_t = k / sqrt(m + t)"""
        return float(self.cfg.eta_k) / float((self.cfg.eta_m + t) ** 0.5)
    
    def alpha_t(self, t: int) -> float:
        """Compute α_{t+1} = c1 * η_t"""
        return float(self.cfg.c1) * self.eta_t(t)
    
    def beta_t(self, t: int) -> float:
        """Compute β_{t+1} = c2 * η_t"""
        return float(self.cfg.c2) * self.eta_t(t)
    
    @torch.no_grad()
    def compute_zo_gradient(self, real_imgs: torch.Tensor, z: torch.Tensor, 
                           generator: nn.Module, discriminator: nn.Module, 
                           q: int = 1) -> torch.Tensor:
        """
        Compute zeroth-order gradient estimate for generator parameters
        
        Args:
            real_imgs: Real images batch
            z: Noise batch
            generator: Generator model
            discriminator: Discriminator model
            q: Number of random directions
            
        Returns:
            Estimated gradient vector
        """
        # Get current parameters
        x_vec = params_to_vec(generator)
        d = x_vec.numel()
        grad_est = torch.zeros_like(x_vec)
        
        # Compute current loss for reference
        fake_imgs = generator(z)
        d_fake = discriminator(fake_imgs)
        
        # GAN objective: min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
        # For generator: minimize log(1 - D(G(z))) or maximize log(D(G(z)))
        # We'll use the non-saturating loss: maximize log(D(G(z)))
        current_loss = -torch.log(d_fake + 1e-8).mean()
        
        for _ in range(q):
            # Sample random direction
            u = sample_unit_vec(d, self.device)
            
            # Perturb parameters in +δ and -δ directions
            xp = x_vec + self.cfg.zo_delta * u
            xm = x_vec - self.cfg.zo_delta * u
            
            # Compute loss at x + δu
            vec_to_params(xp, generator)
            fake_imgs_p = generator(z)
            d_fake_p = discriminator(fake_imgs_p)
            loss_p = -torch.log(d_fake_p + 1e-8).mean()
            
            # Compute loss at x - δu
            vec_to_params(xm, generator)
            fake_imgs_m = generator(z)
            d_fake_m = discriminator(fake_imgs_m)
            loss_m = -torch.log(d_fake_m + 1e-8).mean()
            
            # Two-point gradient estimate
            grad_est += (d / (2.0 * self.cfg.zo_delta)) * (loss_p - loss_m) * u
        
        # Restore original parameters
        vec_to_params(x_vec, generator)
        
        grad_est /= float(q)
        return grad_est
    
    def compute_first_order_gradient(self, real_imgs: torch.Tensor, z: torch.Tensor, 
                                    generator: nn.Module, discriminator: nn.Module) -> torch.Tensor:
        """
        Compute first-order gradient for discriminator parameters
        
        Args:
            real_imgs: Real images batch
            z: Noise batch
            generator: Generator model
            discriminator: Discriminator model
            
        Returns:
            Gradient vector for discriminator parameters
        """
        # Clear gradients
        discriminator.zero_grad()
        
        # Compute discriminator loss
        # Real images
        real_validity = discriminator(real_imgs)
        d_real_loss = -torch.log(real_validity + 1e-8).mean()
        
        # Fake images
        fake_imgs = generator(z).detach()
        fake_validity = discriminator(fake_imgs)
        d_fake_loss = -torch.log(1 - fake_validity + 1e-8).mean()
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        
        # Compute gradients
        d_loss.backward()
        
        # Extract gradient vector
        grad_vec = torch.cat([p.grad.view(-1) for p in discriminator.parameters() if p.requires_grad])
        
        return grad_vec, d_loss.item()
    
    def initialize_momentum(self, real_imgs: torch.Tensor, z: torch.Tensor):
        """Initialize momentum vectors v1 and w1 using first batch"""
        # Initialize v1 (first-order gradient for discriminator)
        self.v, d_loss = self.compute_first_order_gradient(real_imgs, z, self.generator, self.discriminator)
        
        # Initialize w1 (zeroth-order gradient for generator)
        self.w = self.compute_zo_gradient(real_imgs, z, self.generator, self.discriminator, self.cfg.zo_q)
        
        return d_loss
    
    def update_generator(self, t: int) -> float:
        """Update generator parameters using ZO gradient"""
        # Get current parameters
        x_vec = params_to_vec(self.generator)
        
        # Compute η_t
        eta = self.eta_t(t)
        
        # Update: x̃_{t+1} = x_t - γ w_t
        x_tilde = x_vec - self.cfg.gamma * self.w
        
        # Update: x_{t+1} = x_t + η_t (x̃_{t+1} - x_t)
        x_next = x_vec + eta * (x_tilde - x_vec)
        
        # Apply update
        vec_to_params(x_next, self.generator)
        
        # Compute generator loss for logging
        z = torch.randn(self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        d_fake = self.discriminator(fake_imgs)
        g_loss = -torch.log(d_fake + 1e-8).mean().item()
        
        return g_loss
    
    def update_discriminator(self, real_imgs: torch.Tensor, z: torch.Tensor, t: int) -> Tuple[float, float]:
        """Update discriminator parameters using first-order gradient"""
        # Get current parameters (as vector)
        y_vec = params_to_vec(self.discriminator)
        
        # Compute η_t and λ
        eta = self.eta_t(t)
        lam = self.cfg.lam
        
        # Update: ỹ_{t+1} = argmin_y {⟨v_t, y⟩ + (1/2λ)‖y - y_t‖^2}
        # This is equivalent to projection: ỹ_{t+1} = Π_Y(y_t - λ v_t)
        # For unconstrained case (Y = R^n), simply: ỹ_{t+1} = y_t - λ v_t
        y_tilde = y_vec - lam * self.v
        
        # Apply projection to simplex if needed (for constrained case)
        # y_tilde = project_to_simplex(y_tilde, z=1.0)
        
        # Update: y_{t+1} = y_t + η_t (ỹ_{t+1} - y_t)
        y_next = y_vec + eta * (y_tilde - y_vec)
        
        # Apply update
        vec_to_params(y_next, self.discriminator)
        
        # Compute discriminator loss for logging
        with torch.no_grad():
            real_validity = self.discriminator(real_imgs)
            fake_imgs = self.generator(z)
            fake_validity = self.discriminator(fake_imgs)
            
            d_real_loss = -torch.log(real_validity + 1e-8).mean().item()
            d_fake_loss = -torch.log(1 - fake_validity + 1e-8).mean().item()
            d_loss = d_real_loss + d_fake_loss
            
            real_score = real_validity.mean().item()
            fake_score = fake_validity.mean().item()
        
        return d_loss, real_score, fake_score
    
    def update_momentum(self, real_imgs: torch.Tensor, z: torch.Tensor, t: int):
        """Update momentum vectors w_{t+1} and v_{t+1}"""
        # Compute β_{t+1}
        beta = self.beta_t(t)
        
        # Update w_{t+1}: (1 - β_{t+1}) w_t + β_{t+1} ∇_x f_δ(x_{t+1}, y_{t+1})
        w_hat = self.compute_zo_gradient(real_imgs, z, self.generator, self.discriminator, self.cfg.zo_q)
        self.w = (1.0 - beta) * self.w + beta * w_hat
        
        # Compute α_{t+1}
        alpha = self.alpha_t(t)
        
        # Update v_{t+1}: (1 - α_{t+1}) v_t + α_{t+1} ∇_y g(x_{t+1}, y_{t+1})
        v_hat, _ = self.compute_first_order_gradient(real_imgs, z, self.generator, self.discriminator)
        self.v = (1.0 - alpha) * self.v + alpha * v_hat
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        num_batches = 0
        
        # Initialize iterator
        data_iter = iter(dataloader)
        
        # Get first batch for initialization if needed
        if self.w is None or self.v is None:
            try:
                real_imgs, _ = next(data_iter)
                real_imgs = real_imgs.to(self.device)
                z = torch.randn(real_imgs.size(0), self.cfg.latent_dim, device=self.device)
                d_loss = self.initialize_momentum(real_imgs, z)
                epoch_d_loss += d_loss
                num_batches += 1
            except StopIteration:
                pass
        
        # Training loop
        for batch_idx, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)
            
            # Generate noise
            z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
            
            # Compute current iteration number
            t = (epoch - 1) * len(dataloader) + batch_idx + 1
            
            # Update discriminator (y)
            d_loss, real_score, fake_score = self.update_discriminator(real_imgs, z, t)
            
            # Update generator (x) every n_critic steps
            if batch_idx % self.cfg.n_critic == 0:
                g_loss = self.update_generator(t)
                epoch_g_loss += g_loss
            
            # Update momentum vectors
            self.update_momentum(real_imgs, z, t)
            
            # Accumulate statistics
            epoch_d_loss += d_loss
            epoch_real_score += real_score
            epoch_fake_score += fake_score
            num_batches += 1
            
            # Logging
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}, "
                      f"Real_score={real_score:.4f}, Fake_score={fake_score:.4f}")
        
        # Return epoch statistics
        return {
            'd_loss': epoch_d_loss / max(num_batches, 1),
            'g_loss': epoch_g_loss / max(num_batches // max(self.cfg.n_critic, 1), 1),
            'real_score': epoch_real_score / max(num_batches, 1),
            'fake_score': epoch_fake_score / max(num_batches, 1)
        }


# -------------------------
# Data Loading and Utilities
# -------------------------
def get_dataloader(cfg: ZOFOGDA_GAN_Cfg) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for the specified dataset"""
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    if cfg.dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=cfg.data_dir, train=False, download=True, transform=transform
        )
        cfg.num_channels = 1
        
    elif cfg.dataset == "fashionmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=cfg.data_dir, train=False, download=True, transform=transform
        )
        cfg.num_channels = 1
        
    elif cfg.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir, train=False, download=True, transform=transform
        )
        cfg.num_channels = 3
        
    elif cfg.dataset == "cifar100":
        transform = transforms.Compose([
            transforms.Resize(cfg.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir, train=False, download=True, transform=transform
        )
        cfg.num_channels = 3
        
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def evaluate_gan(generator: nn.Module, dataloader: DataLoader, 
                latent_dim: int, device: torch.device, num_samples: int = 1000) -> dict:
    """Evaluate GAN performance"""
    generator.eval()
    
    metrics = {}
    
    with torch.no_grad():
        # Generate samples
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_imgs = generator(z)
        
        # Compute statistics
        metrics['fake_mean'] = fake_imgs.mean().item()
        metrics['fake_std'] = fake_imgs.std().item()
        
        # Compute inception score (simplified)
        # Note: For proper inception score, you'd need a pre-trained Inception network
        fake_imgs_flattened = fake_imgs.view(fake_imgs.size(0), -1)
        metrics['diversity'] = fake_imgs_flattened.std(dim=0).mean().item()
    
    return metrics


def save_samples(generator: nn.Module, epoch: int, latent_dim: int, 
                device: torch.device, save_dir: str = "./samples"):
    """Save generated samples"""
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Generate samples
        z = torch.randn(64, latent_dim, device=device)
        samples = generator(z)
        
        # Save grid of samples
        torchvision.utils.save_image(
            samples.data,
            os.path.join(save_dir, f"epoch_{epoch:03d}.png"),
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )


# -------------------------
# Main Training Function
# -------------------------
def train_zofogda_gan(cfg: ZOFOGDA_GAN_Cfg):
    """Main training function for ZOFOGDA GAN"""
    
    # Set seed
    set_seed(cfg.seed)
    
    # Create output directories
    os.makedirs("./gan_results", exist_ok=True)
    os.makedirs("./gan_samples", exist_ok=True)
    os.makedirs("./gan_models", exist_ok=True)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloader(cfg)
    print(f"Dataset: {cfg.dataset}, Channels: {cfg.num_channels}, "
          f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    
    # Create models
    generator = Generator(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.g_hidden_dim,
        output_channels=cfg.num_channels,
        img_size=cfg.img_size,
        use_bn=cfg.use_batch_norm
    )
    
    discriminator = Discriminator(
        input_channels=cfg.num_channels,
        hidden_dim=cfg.d_hidden_dim,
        img_size=cfg.img_size,
        use_bn=cfg.use_batch_norm
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create optimizer
    optimizer = ZOFOGDA_GAN(generator, discriminator, cfg)
    
    # Training history
    history = {
        'epoch': [],
        'd_loss': [],
        'g_loss': [],
        'real_score': [],
        'fake_score': [],
        'diversity': []
    }
    
    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"{'='*50}")
        
        # Train for one epoch
        epoch_stats = optimizer.train_epoch(train_loader, epoch)
        
        # Update history
        history['epoch'].append(epoch)
        history['d_loss'].append(epoch_stats['d_loss'])
        history['g_loss'].append(epoch_stats['g_loss'])
        history['real_score'].append(epoch_stats['real_score'])
        history['fake_score'].append(epoch_stats['fake_score'])
        
        # Evaluate
        if epoch % cfg.eval_freq == 0:
            eval_metrics = evaluate_gan(
                generator, test_loader, cfg.latent_dim, torch.device(cfg.device)
            )
            history['diversity'].append(eval_metrics['diversity'])
            
            print(f"\nEvaluation at epoch {epoch}:")
            print(f"  D_loss: {epoch_stats['d_loss']:.4f}")
            print(f"  G_loss: {epoch_stats['g_loss']:.4f}")
            print(f"  Real score: {epoch_stats['real_score']:.4f}")
            print(f"  Fake score: {epoch_stats['fake_score']:.4f}")
            print(f"  Diversity: {eval_metrics['diversity']:.4f}")
        
        # Save samples
        if epoch % cfg.sample_freq == 0:
            save_samples(generator, epoch, cfg.latent_dim, torch.device(cfg.device))
            print(f"Saved samples for epoch {epoch}")
        
        # Save model
        if cfg.save_model and epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'config': cfg,
                'history': history
            }, f"./models/zofogda_{cfg.dataset}_epoch_{epoch}.pth")
    
    # Save final model and history
    torch.save({
        'epoch': cfg.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': cfg,
        'history': history
    }, f"./gan_models/zofogda_{cfg.dataset}_final.pth")
    
    df = pd.DataFrame(history)
    df.to_csv(f"./gan_results/zofogda_{cfg.dataset}_history.csv", index=False)
    
    print(f"\nTraining completed!")
    print(f"Results saved to ./gan_results/")
    print(f"Models saved to ./gan_models/")
    print(f"Samples saved to ./gan_samples/")
    
    return generator, discriminator, history


# -------------------------
# Configuration and Main
# -------------------------
if __name__ == "__main__":
    # Example configuration for MNIST
    cfg = ZOFOGDA_GAN_Cfg(
        dataset="cifar100",  # Change to "fashionmnist", "cifar10", or "cifar100" as needed
        data_dir="./data",
        img_size=32,
        
        # Model
        latent_dim=100,
        g_hidden_dim=128,
        d_hidden_dim=128,
        use_batch_norm=True,
        
        # Training
        epochs=1,
        batch_size=128,
        num_workers=0,
        
        # ZOFOGDA hyperparameters
        eta_k=10.0,
        eta_m=1000.0,
        c1=3.0,
        c2=3.0,
        gamma=1e-2,  # Smaller gamma for more stable training for G
        lam=1e-4, # for D
        
        # Zeroth-order
        zo_delta=1e-3,
        zo_q=1,
        
        # GAN
        n_critic=1,  # More discriminator updates per generator update
        
        # Evaluation
        eval_freq=5,
        sample_freq=10,
        save_model=True,
        
        # Device
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        seed=42
    )
    
    # Train
    generator, discriminator, history = train_zofogda_gan(cfg)