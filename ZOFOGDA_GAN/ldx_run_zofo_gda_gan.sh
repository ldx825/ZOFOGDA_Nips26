#!/bin/bash

# Run ZOFOGDA GAN training on different datasets

echo "Starting ZOFOGDA GAN experiments..."

# Create directories
mkdir -p data
mkdir -p results
mkdir -p models
mkdir -p samples
mkdir -p logs

# Install dependencies if needed
# pip install torch torchvision tqdm matplotlib pandas

# Train on MNIST
echo "Training on MNIST..."
python zofo_gda_gan.py --dataset mnist --epochs 100 --batch_size 128

# Train on FashionMNIST
echo "Training on FashionMNIST..."
python zofo_gda_gan.py --dataset fashionmnist --epochs 120 --batch_size 128

# Train on CIFAR-10
echo "Training on CIFAR-10..."
python zofo_gda_gan.py --dataset cifar10 --epochs 200 --batch_size 128 --n_critic 5

# Train on CIFAR-100 (optional)
# echo "Training on CIFAR-100..."
# python zofo_gda.py --dataset cifar100 --epochs 250 --batch_size 128 --n_critic 5

# Evaluate all models
echo "Evaluating models..."
python evaluate.py --model_path ./models/zofogda_mnist_final.pth --dataset mnist
python evaluate.py --model_path ./models/zofogda_fashionmnist_final.pth --dataset fashionmnist
python evaluate.py --model_path ./models/zofogda_cifar10_final.pth --dataset cifar10

# Generate comparison report
python -c "
import json
import pandas as pd

datasets = ['mnist', 'fashionmnist', 'cifar10']
results = []

for dataset in datasets:
    try:
        with open(f'evaluation_metrics_{dataset}.json', 'r') as f:
            metrics = json.load(f)
            metrics['dataset'] = dataset
            results.append(metrics)
    except:
        pass

if results:
    df = pd.DataFrame(results)
    df.to_csv('./results/comparison_report.csv', index=False)
    print(df.to_string())
"

echo "Experiments completed!"