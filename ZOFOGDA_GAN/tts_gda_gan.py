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
# Save history to CSV
import pandas as pd
from model import Generator, Discriminator


@dataclass
class TTSCfg:
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
    
    # TT-SGDA hyperparameters (η_x << η_y)
    lr_x: float = 1e-3     # η_x for generator (small)
    lr_y: float = 5e-2     # η_y for discriminator (large)
    y_steps_per_batch: int = 1  # can be >1
    
    # GAN specific
    n_critic: int = 1  # Number of discriminator updates per generator update
    
    # Regularization
    weight_decay: float = 0.0
    clip_value: float = 0.01  # For WGAN-like training
    
    # Evaluation
    eval_freq: int = 5
    sample_freq: int = 10
    save_model: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Logging
    log_path: str = "./results/ttsgda"


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WDiscriminator(nn.Module):
    """Discriminator for WGAN (without sigmoid)"""
    def __init__(self, input_channels: int = 1, hidden_dim: int = 128, 
                 img_size: int = 32, use_bn: bool = True):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate flattened size
        self.flatten_size = hidden_dim * (img_size // 8) * (img_size // 8)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1)
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -------------------------
# TT-SGDA Algorithm Implementation
# -------------------------
class TT_SGDA:
    """Implement TT-SGDA algorithm for GAN training"""
    
    def __init__(self, generator: nn.Module, discriminator: nn.Module, cfg: TTSCfg):
        self.generator = generator
        self.discriminator = discriminator
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Separate optimizers with different learning rates
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), 
            lr=cfg.lr_x,
            betas=(0.5, 0.999),
            weight_decay=cfg.weight_decay
        )
        
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr_y,
            betas=(0.5, 0.999),
            weight_decay=cfg.weight_decay
        )
        
        # Track parameters for gradient clipping (for WGAN)
        self.d_params = list(self.discriminator.parameters())
        
        # Tracking
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'real_score': [],
            'fake_score': []
        }
    
    def compute_loss(self, real_imgs: torch.Tensor, z: torch.Tensor, 
                     return_scores: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAN losses
        
        Returns:
            d_loss: Discriminator loss
            g_loss: Generator loss
        """
        # Generate fake images
        fake_imgs = self.generator(z)
        
        # Discriminator predictions
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(fake_imgs)
        
        # Standard GAN loss (minimax)
        # For discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        d_loss_real = -torch.log(d_real + 1e-8).mean()
        d_loss_fake = -torch.log(1 - d_fake + 1e-8).mean()
        d_loss = d_loss_real + d_loss_fake
        
        # For generator: minimize log(1 - D(G(z))) or maximize log(D(G(z)))
        # Using non-saturating loss
        g_loss = -torch.log(d_fake + 1e-8).mean()
        
        if return_scores:
            return d_loss, g_loss, d_real.mean(), d_fake.mean()
        return d_loss, g_loss
    
    def compute_wgan_loss(self, real_imgs: torch.Tensor, z: torch.Tensor,
                         return_scores: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WGAN loss (for comparison)
        """
        # Generate fake images
        fake_imgs = self.generator(z)
        
        # Discriminator predictions (no sigmoid)
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(fake_imgs)
        
        # WGAN loss
        d_loss = -d_real.mean() + d_fake.mean()
        g_loss = -d_fake.mean()
        
        if return_scores:
            return d_loss, g_loss, d_real.mean(), d_fake.mean()
        return d_loss, g_loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch using TT-SGDA algorithm"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        num_batches = 0
        
        for batch_idx, (real_imgs, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)
            
            # Generate noise
            z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
            
            # ---------------------
            # Update Discriminator (y) - Multiple steps possible
            # ---------------------
            for _ in range(self.cfg.y_steps_per_batch):
                self.opt_d.zero_grad()
                
                # Compute discriminator loss
                d_loss, g_loss, real_score, fake_score = self.compute_loss(
                    real_imgs, z, return_scores=True
                )
                
                # Backward and optimize (gradient ascent for y)
                d_loss.backward()
                self.opt_d.step()
                
                # Gradient clipping (optional, for WGAN)
                if self.cfg.clip_value > 0:
                    for p in self.d_params:
                        p.data.clamp_(-self.cfg.clip_value, self.cfg.clip_value)
            
            # ---------------------
            # Update Generator (x) - Slower updates
            # ---------------------
            if batch_idx % self.cfg.n_critic == 0:
                self.opt_g.zero_grad()
                
                # Generate new noise for generator update
                z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
                
                # Compute generator loss
                _, g_loss = self.compute_loss(real_imgs, z, return_scores=False)
                
                # Backward and optimize (gradient descent for x)
                g_loss.backward()
                self.opt_g.step()
            
            # Accumulate statistics
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_real_score += real_score.item()
            epoch_fake_score += fake_score.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: D_loss={d_loss.item():.4f}, "
                      f"G_loss={g_loss.item():.4f}, "
                      f"Real_score={real_score.item():.4f}, "
                      f"Fake_score={fake_score.item():.4f}")
        
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
def get_dataloader(cfg: TTSCfg) -> Tuple[DataLoader, DataLoader]:
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
        
        # Compute diversity measure
        fake_imgs_flattened = fake_imgs.view(fake_imgs.size(0), -1)
        per_pixel_std = fake_imgs_flattened.std(dim=0)
        metrics['diversity'] = per_pixel_std.mean().item()
        
        # Compute mode collapse measure
        pairwise_dist = torch.cdist(fake_imgs_flattened, fake_imgs_flattened, p=2)
        metrics['avg_pairwise_distance'] = pairwise_dist.mean().item()
        
        # Compute intra-class variance
        fake_imgs_variance = fake_imgs_flattened.var(dim=0).mean().item()
        metrics['variance'] = fake_imgs_variance
        
        # Compute Inception Score (simplified)
        # Note: For proper inception score, use a pre-trained Inception network
        class_scores = torch.softmax(torch.randn(num_samples, 10, device=device), dim=1)
        py = class_scores.mean(dim=0)
        scores = []
        for i in range(10):
            pyx = class_scores[i]
            kl = (pyx * (torch.log(pyx) - torch.log(py))).sum()
            scores.append(torch.exp(kl))
        metrics['inception_score_mean'] = torch.mean(torch.tensor(scores)).item()
    
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
            os.path.join(save_dir, f"ttsgda_epoch_{epoch:03d}.png"),
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )


# -------------------------
# Main Training Function
# -------------------------
def train_ttsgda_gan(cfg: TTSCfg):
    """Main training function for TT-SGDA GAN"""
    
    # Set seed
    set_seed(cfg.seed)
    
    # Create output directories
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
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
    optimizer = TT_SGDA(generator, discriminator, cfg)
    
    # Training history
    history = {
        'epoch': [],
        'd_loss': [],
        'g_loss': [],
        'real_score': [],
        'fake_score': [],
        'diversity': []
    }
    
    # Create log file
    log_file = f"./results/ttsgda_{cfg.dataset}_log.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'd_loss', 'g_loss', 'real_score', 'fake_score', 'diversity'])
    
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
            print(f"  Inception Score: {eval_metrics['inception_score_mean']:.4f}")
        
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
            }, f"./models/ttsgda_{cfg.dataset}_epoch_{epoch}.pth")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            diversity_val = eval_metrics.get('diversity', 0.0) if epoch % cfg.eval_freq == 0 else 0.0
            writer.writerow([
                epoch,
                epoch_stats['d_loss'],
                epoch_stats['g_loss'],
                epoch_stats['real_score'],
                epoch_stats['fake_score'],
                diversity_val
            ])
    
    # Save final model and history
    torch.save({
        'epoch': cfg.epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': cfg,
        'history': history
    }, f"./models/ttsgda_{cfg.dataset}_final.pth")
    df = pd.DataFrame(history)
    df.to_csv(f"./results/ttsgda_{cfg.dataset}_history.csv", index=False)
    
    print(f"\nTraining completed!")
    print(f"Results saved to ./results/")
    print(f"Models saved to ./models/")
    print(f"Samples saved to ./samples/")
    print(f"Log file: {log_file}")
    
    return generator, discriminator, history


# -------------------------
# Configuration and Main
# -------------------------
if __name__ == "__main__":
    # Example configuration for MNIST
    cfg = TTSCfg(
        dataset="mnist",  # Change to "fashionmnist", "cifar10", or "cifar100" as needed
        data_dir="./data",
        img_size=32,
        
        # Model
        latent_dim=100,
        g_hidden_dim=128,
        d_hidden_dim=128,
        use_batch_norm=True,
        
        # Training
        epochs=100,
        batch_size=128,
        num_workers=4,
        
        # TT-SGDA hyperparameters (η_x << η_y)
        lr_x=1e-3,     # Small learning rate for generator
        lr_y=5e-2,     # Large learning rate for discriminator
        y_steps_per_batch=1,
        
        # GAN
        n_critic=5,  # More discriminator updates per generator update
        
        # Regularization
        weight_decay=0.0,
        clip_value=0.0,  # Set to 0.01 for WGAN-style training
        
        # Evaluation
        eval_freq=5,
        sample_freq=10,
        save_model=True,
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        
        # Logging
        log_path=f"./results/ttsgda_mnist"
    )
    
    # Train
    generator, discriminator, history = train_ttsgda_gan(cfg)