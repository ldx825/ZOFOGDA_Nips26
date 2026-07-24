import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from model import Generator, Discriminator
from zofo_gda_gan import ZOFOGDA_GAN_Cfg


def compute_fid(real_features, fake_features):
    """
    Compute Frechet Inception Distance (simplified version)
    For full FID, use torch-fid or clean-fid library
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sigma1 @ sigma2
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_inception_score(fake_imgs, classifier, device, n_split=10):
    """
    Compute Inception Score
    """
    classifier.eval()
    preds = []
    
    with torch.no_grad():
        # Forward pass through classifier
        outputs = classifier(fake_imgs)
        preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    scores = []
    
    # Split into groups
    split_size = preds.shape[0] // n_split
    for i in range(n_split):
        part = preds[i * split_size:(i + 1) * split_size, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)


def evaluate_model(model_path, dataset="mnist", num_samples=10000):
    """
    Evaluate trained GAN model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg = checkpoint['config']
    
    # Create model
    generator = Generator(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.g_hidden_dim,
        output_channels=cfg.num_channels,
        img_size=cfg.img_size,
        use_bn=cfg.use_batch_norm
    )
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Generate samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    
    all_fake_imgs = []
    with torch.no_grad():
        for _ in range(num_samples // 100):
            z = torch.randn(100, cfg.latent_dim, device=device)
            fake_imgs = generator(z)
            all_fake_imgs.append(fake_imgs.cpu())
    
    fake_imgs = torch.cat(all_fake_imgs, dim=0)[:num_samples]
    
    # Compute statistics
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = fake_imgs.mean().item()
    metrics['std'] = fake_imgs.std().item()
    metrics['min'] = fake_imgs.min().item()
    metrics['max'] = fake_imgs.max().item()
    
    # Diversity measure
    fake_flattened = fake_imgs.view(fake_imgs.size(0), -1)
    per_pixel_std = fake_flattened.std(dim=0)
    metrics['diversity'] = per_pixel_std.mean().item()
    
    # Mode collapse detection
    pairwise_dist = torch.cdist(fake_flattened, fake_flattened, p=2)
    metrics['avg_pairwise_distance'] = pairwise_dist.mean().item()
    
    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    with torch.no_grad():
        z = torch.randn(10, cfg.latent_dim, device=device)
        samples = generator(z).cpu()
    
    for i, ax in enumerate(axes):
        if cfg.num_channels == 1:
            ax.imshow(samples[i].squeeze(), cmap='gray')
        else:
            ax.imshow(samples[i].permute(1, 2, 0) * 0.5 + 0.5)  # Unnormalize
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_samples_{dataset}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot losses
        axes[0, 0].plot(history['epoch'], history['d_loss'], label='D Loss')
        axes[0, 0].plot(history['epoch'], history['g_loss'], label='G Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot scores
        axes[0, 1].plot(history['epoch'], history['real_score'], label='Real Score')
        axes[0, 1].plot(history['epoch'], history['fake_score'], label='Fake Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Discriminator Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot score difference
        score_diff = np.array(history['real_score']) - np.array(history['fake_score'])
        axes[1, 0].plot(history['epoch'], score_diff)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Real - Fake Score')
        axes[1, 0].set_title('Score Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot diversity if available
        if 'diversity' in history and len(history['diversity']) > 0:
            axes[1, 1].plot(history['epoch'][::cfg.eval_freq], history['diversity'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Diversity')
            axes[1, 1].set_title('Sample Diversity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{dataset}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Print metrics
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {dataset}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")
    
    # Save metrics
    with open(f'evaluation_metrics_{dataset}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    model_path = "./models/zofogda_mnist_final.pth"
    metrics = evaluate_model(model_path, dataset="mnist")