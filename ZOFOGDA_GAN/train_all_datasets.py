import torch
from zofo_gda import ZOFOGDACfg, train_zofogda_gan
import json
from pathlib import Path


def train_all_datasets():
    """Train on all datasets with different configurations"""
    
    datasets = ["mnist", "fashionmnist", "cifar10"]  # Add cifar100 if desired
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Training on {dataset}")
        print(f"{'='*60}")
        
        # Adjust configuration based on dataset
        if dataset == "cifar10" or dataset == "cifar100":
            img_size = 32
            channels = 3
            n_critic = 5
            gamma = 5e-4
        else:
            img_size = 32
            channels = 1
            n_critic = 3
            gamma = 1e-3
        
        cfg = ZOFOGDACfg(
            dataset=dataset,
            data_dir="./data",
            img_size=img_size,
            num_channels=channels,
            
            # Model
            latent_dim=100,
            g_hidden_dim=128,
            d_hidden_dim=128,
            use_batch_norm=True,
            
            # Training
            epochs=100 if dataset in ["mnist", "fashionmnist"] else 200,
            batch_size=128,
            num_workers=4,
            
            # ZOFOGDA hyperparameters
            eta_k=1.0,
            eta_m=100.0,
            c1=0.1,
            c2=0.1,
            gamma=gamma,
            lam=1.0,
            
            # Zeroth-order
            zo_delta=1e-3,
            zo_q=1,
            
            # GAN
            n_critic=n_critic,
            
            # Evaluation
            eval_freq=5,
            sample_freq=10,
            save_model=True,
            
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=42
        )
        
        # Train
        try:
            generator, discriminator, history = train_zofogda_gan(cfg)
            
            # Store results
            results[dataset] = {
                'final_d_loss': history['d_loss'][-1],
                'final_g_loss': history['g_loss'][-1],
                'final_real_score': history['real_score'][-1],
                'final_fake_score': history['fake_score'][-1],
                'epochs_trained': len(history['epoch'])
            }
            
        except Exception as e:
            print(f"Error training on {dataset}: {e}")
            results[dataset] = {'error': str(e)}
    
    # Save overall results
    with open('./results/all_datasets_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed for all datasets!")
    print(f"{'='*60}")
    
    for dataset, result in results.items():
        if 'error' not in result:
            print(f"\n{dataset}:")
            print(f"  Final D Loss: {result['final_d_loss']:.4f}")
            print(f"  Final G Loss: {result['final_g_loss']:.4f}")
            print(f"  Real Score: {result['final_real_score']:.4f}")
            print(f"  Fake Score: {result['final_fake_score']:.4f}")


if __name__ == "__main__":
    train_all_datasets()