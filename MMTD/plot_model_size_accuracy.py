#!/usr/bin/env python3
"""
Plot model size vs accuracy with properly positioned labels to avoid overlapping
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

def load_and_process_data(csv_path):
    """Load CSV data and calculate average accuracy per model"""
    df = pd.read_csv(csv_path)
    
    # Filter out incomplete data (models with missing values)
    df = df.dropna(subset=['best_accuracy', 'model_size_mb'])
    
    # Group by model name and calculate average accuracy across folds
    model_stats = df.groupby('model_name').agg({
        'best_accuracy': 'mean',
        'model_size_mb': 'first',  # Model size should be same across folds
        'fold': 'count'  # Count number of folds
    }).reset_index()
    
    # Only include models with 5 folds
    model_stats = model_stats[model_stats['fold'] == 5]
    
    return model_stats

def create_plot(model_stats):
    """Create scatter plot with properly positioned labels"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter plot
    scatter = ax.scatter(model_stats['model_size_mb'], 
                        model_stats['best_accuracy'], 
                        s=100, 
                        alpha=0.7,
                        c='steelblue',
                        edgecolors='black',
                        linewidth=1)
    
    # Prepare text labels
    texts = []
    for idx, row in model_stats.iterrows():
        text = ax.text(row['model_size_mb'], 
                      row['best_accuracy'], 
                      row['model_name'],
                      fontsize=9,
                      ha='center',
                      va='center',
                      bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', 
                               alpha=0.8,
                               edgecolor='gray'))
        texts.append(text)
    
    # Adjust text positions to avoid overlapping
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
                expand_points=(1.2, 1.2),
                expand_text=(1.1, 1.1),
                force_points=0.5,
                force_text=0.5)
    
    # Customize plot
    ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (5-fold)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Average Accuracy\n(5-fold Cross Validation)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set axis limits with some padding
    x_margin = (model_stats['model_size_mb'].max() - model_stats['model_size_mb'].min()) * 0.1
    y_margin = (model_stats['best_accuracy'].max() - model_stats['best_accuracy'].min()) * 0.05
    
    ax.set_xlim(model_stats['model_size_mb'].min() - x_margin, 
                model_stats['model_size_mb'].max() + x_margin)
    ax.set_ylim(model_stats['best_accuracy'].min() - y_margin, 
                model_stats['best_accuracy'].max() + y_margin)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Add statistics text box
    stats_text = f"Total Models: {len(model_stats)}\n"
    stats_text += f"Size Range: {model_stats['model_size_mb'].min():.1f} - {model_stats['model_size_mb'].max():.1f} MB\n"
    stats_text += f"Accuracy Range: {model_stats['best_accuracy'].min():.3f} - {model_stats['best_accuracy'].max():.3f}"
    
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def save_results_table(model_stats, output_path):
    """Save results as a formatted table"""
    # Sort by accuracy (descending)
    sorted_stats = model_stats.sort_values('best_accuracy', ascending=False).copy()
    sorted_stats['rank'] = range(1, len(sorted_stats) + 1)
    
    # Format for display
    sorted_stats['accuracy_pct'] = sorted_stats['best_accuracy'].apply(lambda x: f"{x:.3f}")
    sorted_stats['size_mb'] = sorted_stats['model_size_mb'].apply(lambda x: f"{x:.1f}")
    
    # Select columns for output
    output_df = sorted_stats[['rank', 'model_name', 'accuracy_pct', 'size_mb']].copy()
    output_df.columns = ['Rank', 'Model Name', 'Avg Accuracy', 'Size (MB)']
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Results table saved to: {output_path}")
    
    return output_df

def main():
    # Load and process data
    csv_path = 'total_results.csv'
    model_stats = load_and_process_data(csv_path)
    
    print(f"Loaded {len(model_stats)} models with complete 5-fold results")
    print("\nModel Statistics:")
    print(model_stats[['model_name', 'best_accuracy', 'model_size_mb']].to_string(index=False))
    
    # Create plot
    fig, ax = create_plot(model_stats)
    
    # Save plot
    plot_path = 'model_size_vs_accuracy.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save results table
    table_path = 'model_ranking_by_accuracy.csv'
    results_table = save_results_table(model_stats, table_path)
    
    print(f"\nTop 5 Models by Accuracy:")
    print(results_table.head().to_string(index=False))
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
