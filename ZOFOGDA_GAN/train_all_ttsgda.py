import torch
from ttsgda_gan import TTSCfg, train_ttsgda_gan
import json
import time
from pathlib import Path


def train_all_datasets():
    """Train TT-SGDA on all datasets with different configurations"""
    
    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Training TT-SGDA on {dataset}")
        print(f"{'='*60}")
        
        # Adjust configuration based on dataset
        if dataset in ["cifar10", "cifar100"]:
            img_size = 32
            channels = 3
            lr_x = 2e-4  # Smaller learning rate for complex datasets
            lr_y = 1e-2  # Still larger than lr_x
            epochs = 200
            n_critic = 5
        else:
            img_size = 32
            channels = 1
            lr_x = 1e-3
            lr_y = 5e-2
            epochs = 100
            n_critic = 3
        
        cfg = TTSCfg(
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
            epochs=epochs,
            batch_size=128,
            num_workers=4,
            
            # TT-SGDA hyperparameters (η_x << η_y)
            lr_x=lr_x,
            lr_y=lr_y,
            y_steps_per_batch=1,
            
            # GAN
            n_critic=n_critic,
            
            # Regularization
            weight_decay=1e-5,
            clip_value=0.0,
            
            # Evaluation
            eval_freq=5,
            sample_freq=10,
            save_model=True,
            
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=42,
            
            # Logging
            log_path=f"./results/ttsgda_{dataset}"
        )
        
        # Train
        try:
            start_time = time.time()
            generator, discriminator, history = train_ttsgda_gan(cfg)
            training_time = time.time() - start_time
            
            # Store results
            results[dataset] = {
                'final_d_loss': history['d_loss'][-1] if history['d_loss'] else None,
                'final_g_loss': history['g_loss'][-1] if history['g_loss'] else None,
                'final_real_score': history['real_score'][-1] if history['real_score'] else None,
                'final_fake_score': history['fake_score'][-1] if history['fake_score'] else None,
                'epochs_trained': len(history['epoch']),
                'training_time': training_time,
                'history_file': f"./results/ttsgda_{dataset}_history.csv"
            }
            
        except Exception as e:
            print(f"Error training TT-SGDA on {dataset}: {e}")
            results[dataset] = {'error': str(e)}
    
    # Save overall results
    with open('./results/ttsgda_all_datasets_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TT-SGDA Training completed for all datasets!")
    print(f"{'='*60}")
    
    for dataset, result in results.items():
        if 'error' not in result:
            print(f"\n{dataset}:")
            print(f"  Training Time: {result['training_time']:.2f} seconds")
            print(f"  Epochs Trained: {result['epochs_trained']}")
            print(f"  Final D Loss: {result['final_d_loss']:.4f}")
            print(f"  Final G Loss: {result['final_g_loss']:.4f}")
            print(f"  Real Score: {result['final_real_score']:.4f}")
            print(f"  Fake Score: {result['final_fake_score']:.4f}")


def hyperparameter_search(dataset="mnist"):
    """Perform hyperparameter search for TT-SGDA"""
    
    learning_rates = [
        (1e-3, 5e-2),  # Base
        (5e-4, 2e-2),  # Smaller
        (2e-3, 1e-1),  # Larger
        (1e-4, 1e-3),  # Very small
    ]
    
    n_critic_values = [1, 3, 5]
    
    results = []
    
    for lr_x, lr_y in learning_rates:
        for n_critic in n_critic_values:
            print(f"\n{'='*60}")
            print(f"Testing: lr_x={lr_x}, lr_y={lr_y}, n_critic={n_critic}")
            print(f"{'='*60}")
            
            cfg = TTSCfg(
                dataset=dataset,
                data_dir="./data",
                img_size=32,
                
                # Model
                latent_dim=100,
                g_hidden_dim=128,
                d_hidden_dim=128,
                use_batch_norm=True,
                
                # Training
                epochs=50,  # Shorter for search
                batch_size=128,
                num_workers=4,
                
                # TT-SGDA hyperparameters
                lr_x=lr_x,
                lr_y=lr_y,
                y_steps_per_batch=1,
                n_critic=n_critic,
                
                # Regularization
                weight_decay=0.0,
                clip_value=0.0,
                
                # Evaluation
                eval_freq=5,
                sample_freq=10,
                save_model=False,  # Don't save during search
                
                device="cuda" if torch.cuda.is_available() else "cpu",
                seed=42,
            )
            
            try:
                start_time = time.time()
                generator, discriminator, history = train_ttsgda_gan(cfg)
                training_time = time.time() - start_time
                
                # Evaluate final performance
                if history['d_loss'] and history['g_loss']:
                    final_score = (history['real_score'][-1] + history['fake_score'][-1]) / 2
                    
                    results.append({
                        'lr_x': lr_x,
                        'lr_y': lr_y,
                        'n_critic': n_critic,
                        'final_d_loss': history['d_loss'][-1],
                        'final_g_loss': history['g_loss'][-1],
                        'final_score': final_score,
                        'training_time': training_time
                    })
                    
                    print(f"Result: D_loss={history['d_loss'][-1]:.4f}, "
                          f"G_loss={history['g_loss'][-1]:.4f}, "
                          f"Score={final_score:.4f}")
            
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'lr_x': lr_x,
                    'lr_y': lr_y,
                    'n_critic': n_critic,
                    'error': str(e)
                })
    
    # Save search results
    with open(f'./results/ttsgda_hparam_search_{dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print best configuration
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['final_d_loss'])
        print(f"\n{'='*60}")
        print(f"Best Configuration for {dataset}:")
        print(f"  lr_x: {best_result['lr_x']}")
        print(f"  lr_y: {best_result['lr_y']}")
        print(f"  n_critic: {best_result['n_critic']}")
        print(f"  Final D Loss: {best_result['final_d_loss']:.4f}")
        print(f"  Final Score: {best_result['final_score']:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Option 1: Train on all datasets
    train_all_datasets()
    
    # Option 2: Hyperparameter search on a specific dataset
    # hyperparameter_search("mnist")