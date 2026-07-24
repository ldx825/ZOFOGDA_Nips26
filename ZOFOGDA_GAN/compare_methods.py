import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(method, dataset):
    """Load results for a specific method and dataset"""
    try:
        with open(f'./results/{method}_metrics_{dataset}.json', 'r') as f:
            return json.load(f)
    except:
        return None


def compare_all_datasets():
    """Compare TT-SGDA and ZOFOGDA across all datasets"""
    
    datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]
    methods = ["ttsgda", "zofogda"]  # Assuming ZOFOGDA files are named zofogda_metrics_*.json
    
    comparison_data = []
    
    for dataset in datasets:
        for method in methods:
            results = load_results(method, dataset)
            if results:
                results['dataset'] = dataset
                results['method'] = method.upper()
                comparison_data.append(results)
    
    if not comparison_data:
        print("No results found for comparison")
        return
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv('./results/method_comparison.csv', index=False)
    
    # Create visualizations
    metrics_to_plot = ['inception_score_mean', 'fid', 'fake_diversity']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            # Create pivot table
            pivot_df = df.pivot(index='dataset', columns='method', values=metric)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric)
            ax.legend(title='Method')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./results/comparison_{metric}.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Print summary table
    print("\nMethod Comparison Summary:")
    print("=" * 80)
    
    summary_metrics = ['inception_score_mean', 'fid', 'fake_diversity']
    for metric in summary_metrics:
        if metric in df.columns:
            print(f"\n{metric}:")
            print("-" * 40)
            
            # Group by method
            grouped = df.groupby('method')[metric].agg(['mean', 'std', 'min', 'max'])
            print(grouped.to_string())
    
    # Create radar chart for comparison
    create_radar_chart(df)


def create_radar_chart(df):
    """Create radar chart comparing methods"""
    
    # Normalize metrics (higher is better for some, lower for others)
    # We'll use min-max normalization for each metric separately
    normalized_data = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # For each metric, normalize
        norm_row = {'method': method}
        
        # Inception Score (higher is better)
        if 'inception_score_mean' in df.columns:
            max_is = df['inception_score_mean'].max()
            min_is = df['inception_score_mean'].min()
            if max_is > min_is:
                norm_is = (method_data['inception_score_mean'].mean() - min_is) / (max_is - min_is)
                norm_row['IS'] = norm_is
        
        # FID (lower is better, so invert)
        if 'fid' in df.columns:
            max_fid = df['fid'].max()
            min_fid = df['fid'].min()
            if max_fid > min_fid:
                norm_fid = 1 - ((method_data['fid'].mean() - min_fid) / (max_fid - min_fid))
                norm_row['FID'] = norm_fid
        
        # Diversity (higher is better)
        if 'fake_diversity' in df.columns:
            max_div = df['fake_diversity'].max()
            min_div = df['fake_diversity'].min()
            if max_div > min_div:
                norm_div = (method_data['fake_diversity'].mean() - min_div) / (max_div - min_div)
                norm_row['Diversity'] = norm_div
        
        normalized_data.append(norm_row)
    
    norm_df = pd.DataFrame(normalized_data)
    
    # Create radar chart
    if len(norm_df) > 1 and len(norm_df.columns) > 2:
        metrics = [col for col in norm_df.columns if col != 'method']
        N = len(metrics)
        
        # Create angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for idx, row in norm_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['method'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Method Comparison (Normalized)', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('./results/method_comparison_radar.png', dpi=150, bbox_inches='tight')
        plt.show()


def analyze_training_dynamics():
    """Analyze training dynamics from history files"""
    
    datasets = ["mnist", "fashionmnist", "cifar10"]
    methods = ["ttsgda", "zofogda"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets[:4]):  # Limit to 4 datasets
        ax = axes[idx]
        
        for method in methods:
            history_file = f'./results/{method}_{dataset}_history.csv'
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                
                if 'd_loss' in history_df.columns:
                    ax.plot(history_df.index[:100], history_df['d_loss'][:100], 
                           label=f'{method.upper()}', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator Loss')
        ax.set_title(f'{dataset.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Dynamics: Discriminator Loss', fontsize=16)
    plt.tight_layout()
    plt.savefig('./results/training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Compare methods
    compare_all_datasets()
    
    # Analyze training dynamics
    analyze_training_dynamics()
    
    # Print summary
    print("\n" + "="*80)
    print("Comparison completed!")
    print("Results saved to ./results/")
    print("="*80)