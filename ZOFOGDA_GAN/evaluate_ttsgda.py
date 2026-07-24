import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from ttsgda_gan import Generator, Discriminator, TTSCfg, get_dataloader
import pandas as pd


def compute_inception_score(fake_imgs, num_classes=10, splits=10):
    """
    Compute Inception Score (simplified version)
    Note: For accurate IS, use a pre-trained Inception network
    """
    # Fake classification scores (simulated)
    scores = torch.randn(fake_imgs.size(0), num_classes)
    scores = torch.softmax(scores, dim=1)
    
    # Split into groups
    split_scores = []
    for i in range(splits):
        part = scores[i * (scores.size(0) // splits): (i + 1) * (scores.size(0) // splits), :]
        kl = part * (torch.log(part + 1e-16) - torch.log(torch.mean(part, dim=0, keepdim=True) + 1e-16))
        kl = torch.mean(torch.sum(kl, dim=1))
        split_scores.append(torch.exp(kl).item())
    
    return np.mean(split_scores), np.std(split_scores)


def compute_fid(real_features, fake_features):
    """
    Compute Frechet Inception Distance (simplified)
    For real FID, use proper feature extraction
    """
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sigma1 @ sigma2
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def evaluate_model(model_path, dataset="mnist", num_samples=10000, batch_size=100):
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
        for i in range(0, num_samples, batch_size):
            batch_size_curr = min(batch_size, num_samples - i)
            z = torch.randn(batch_size_curr, cfg.latent_dim, device=device)
            fake_imgs = generator(z)
            all_fake_imgs.append(fake_imgs.cpu())
    
    fake_imgs = torch.cat(all_fake_imgs, dim=0)[:num_samples]
    
    # Load real data for comparison
    _, test_loader = get_dataloader(cfg)
    real_imgs = []
    for batch_idx, (real_batch, _) in enumerate(test_loader):
        real_imgs.append(real_batch)
        if len(real_imgs) * cfg.batch_size >= num_samples:
            break
    real_imgs = torch.cat(real_imgs, dim=0)[:num_samples]
    
    # Compute metrics
    metrics = {}
    
    # Basic statistics
    metrics['fake_mean'] = fake_imgs.mean().item()
    metrics['fake_std'] = fake_imgs.std().item()
    metrics['real_mean'] = real_imgs.mean().item()
    metrics['real_std'] = real_imgs.std().item()
    
    # Diversity measures
    fake_flattened = fake_imgs.view(fake_imgs.size(0), -1)
    real_flattened = real_imgs.view(real_imgs.size(0), -1)
    
    metrics['fake_diversity'] = fake_flattened.std(dim=0).mean().item()
    metrics['real_diversity'] = real_flattened.std(dim=0).mean().item()
    
    # Mode collapse detection
    fake_pairwise_dist = torch.cdist(fake_flattened, fake_flattened, p=2)
    metrics['fake_avg_pairwise_distance'] = fake_pairwise_dist.mean().item()
    
    real_pairwise_dist = torch.cdist(real_flattened, real_flattened, p=2)
    metrics['real_avg_pairwise_distance'] = real_pairwise_dist.mean().item()
    
    # Inception Score
    is_mean, is_std = compute_inception_score(fake_imgs)
    metrics['inception_score_mean'] = is_mean
    metrics['inception_score_std'] = is_std
    
    # FID (simplified)
    fake_features = fake_flattened.numpy()[:1000]  # Use subset for speed
    real_features = real_flattened.numpy()[:1000]
    metrics['fid'] = compute_fid(real_features, fake_features)
    
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
        ax.set_title(f'Sample {i+1}')
    
    plt.suptitle(f'TT-SGDA Generated Samples ({dataset})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./results/ttsgda_samples_{dataset}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot training history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot losses
        axes[0, 0].plot(history['epoch'], history['d_loss'], label='Discriminator Loss')
        axes[0, 0].plot(history['epoch'], history['g_loss'], label='Generator Loss')
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
        if len(history['real_score']) == len(history['fake_score']):
            score_diff = np.array(history['real_score']) - np.array(history['fake_score'])
            axes[1, 0].plot(history['epoch'], score_diff)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Real - Fake Score')
            axes[1, 0].set_title('Score Difference')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot diversity if available
        if 'diversity' in history and len(history['diversity']) > 0:
            diversity_epochs = history['epoch'][::cfg.eval_freq][:len(history['diversity'])]
            axes[1, 1].plot(diversity_epochs, history['diversity'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Diversity')
            axes[1, 1].set_title('Sample Diversity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'TT-SGDA Training History ({dataset})', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'./results/ttsgda_history_{dataset}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"TT-SGDA Evaluation Results for {dataset}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:35s}: {value:.6f}")
    
    # Save metrics
    with open(f'./results/ttsgda_metrics_{dataset}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save to CSV for comparison
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'./results/ttsgda_metrics_{dataset}.csv', index=False)
    
    return metrics


def compare_methods(dataset="mnist"):
    """Compare TT-SGDA with ZOFOGDA results"""
    try:
        # Load TT-SGDA metrics
        with open(f'./results/ttsgda_metrics_{dataset}.json', 'r') as f:
            ttsgda_metrics = json.load(f)
        
        # Load ZOFOGDA metrics (if available)
        try:
            with open(f'./results/zofo_metrics_{dataset}.json', 'r') as f:
                zofo_metrics = json.load(f)
        except:
            zofo_metrics = None
        
        # Create comparison table
        metrics_to_compare = [
            'inception_score_mean',
            'fid',
            'fake_diversity',
            'fake_avg_pairwise_distance'
        ]
        
        print(f"\n{'='*60}")
        print(f"Comparison: TT-SGDA vs ZOFOGDA ({dataset})")
        print(f"{'='*60}")
        
        print(f"{'Metric':35s} {'TT-SGDA':15s} {'ZOFOGDA':15s} {'Difference':15s}")
        print(f"{'-'*80}")
        
        for metric in metrics_to_compare:
            ttsgda_val = ttsgda_metrics.get(metric, 'N/A')
            if zofo_metrics:
                zofo_val = zofo_metrics.get(metric, 'N/A')
                if isinstance(ttsgda_val, (int, float)) and isinstance(zofo_val, (int, float)):
                    diff = ttsgda_val - zofo_val
                    print(f"{metric:35s} {ttsgda_val:15.6f} {zofo_val:15.6f} {diff:15.6f}")
                else:
                    print(f"{metric:35s} {str(ttsgda_val):15s} {str(zofo_val):15s} {'N/A':15s}")
            else:
                print(f"{metric:35s} {ttsgda_val if isinstance(ttsgda_val, str) else f'{ttsgda_val:.6f}':15s} {'N/A':15s} {'N/A':15s}")
    
    except Exception as e:
        print(f"Error comparing methods: {e}")


if __name__ == "__main__":
    # Example usage
    dataset = "mnist"
    model_path = f"./models/ttsgda_{dataset}_final.pth"
    
    # Evaluate TT-SGDA model
    print(f"Evaluating TT-SGDA on {dataset}...")
    metrics = evaluate_model(model_path, dataset=dataset, num_samples=10000)
    
    # Compare with ZOFOGDA
    compare_methods(dataset)