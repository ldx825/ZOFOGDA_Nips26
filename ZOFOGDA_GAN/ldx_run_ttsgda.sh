#!/bin/bash

# Run TT-SGDA GAN training on different datasets

echo "Starting TT-SGDA GAN experiments..."

# Create directories
mkdir -p data
mkdir -p results
mkdir -p models
mkdir -p samples
mkdir -p logs

# Install dependencies if needed
# pip install torch torchvision tqdm matplotlib pandas

# Train on MNIST
echo "Training TT-SGDA on MNIST..."
python ttsgda_gan.py --dataset mnist --epochs 100 --lr_x 1e-3 --lr_y 5e-2

# Train on FashionMNIST
echo "Training TT-SGDA on FashionMNIST..."
python ttsgda_gan.py --dataset fashionmnist --epochs 120 --lr_x 1e-3 --lr_y 1e-1

# Train on CIFAR-10
echo "Training TT-SGDA on CIFAR-10..."
python ttsgda_gan.py --dataset cifar10 --epochs 200 --lr_x 2e-4 --lr_y 1e-2 --n_critic 5

# Train on CIFAR-100
echo "Training TT-SGDA on CIFAR-100..."
python ttsgda_gan.py --dataset cifar100 --epochs 250 --lr_x 1e-4 --lr_y 5e-3 --n_critic 5

# Evaluate all models
echo "Evaluating TT-SGDA models..."
python evaluate_ttsgda.py --model_path ./models/ttsgda_mnist_final.pth --dataset mnist
python evaluate_ttsgda.py --model_path ./models/ttsgda_fashionmnist_final.pth --dataset fashionmnist
python evaluate_ttsgda.py --model_path ./models/ttsgda_cifar10_final.pth --dataset cifar10
python evaluate_ttsgda.py --model_path ./models/ttsgda_cifar100_final.pth --dataset cifar100

# Generate comparison report
echo "Generating comparison report..."
python -c "
import json
import pandas as pd

datasets = ['mnist', 'fashionmnist', 'cifar10', 'cifar100']
all_results = []

for dataset in datasets:
    try:
        with open(f'./results/ttsgda_metrics_{dataset}.json', 'r') as f:
            metrics = json.load(f)
            metrics['dataset'] = dataset
            metrics['method'] = 'TT-SGDA'
            all_results.append(metrics)
    except:
        pass

if all_results:
    df = pd.DataFrame(all_results)
    # Select key metrics
    key_columns = ['dataset', 'method', 'inception_score_mean', 'fid', 'fake_diversity', 'fake_avg_pairwise_distance']
    available_columns = [col for col in key_columns if col in df.columns]
    
    report_df = df[available_columns]
    report_df.to_csv('./results/ttsgda_comparison_report.csv', index=False)
    
    print('TT-SGDA Results Summary:')
    print(report_df.to_string(index=False))
"

echo "TT-SGDA experiments completed!"