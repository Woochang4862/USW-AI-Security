#!/usr/bin/env python3
"""
Enhanced plot for model size vs accuracy with better label positioning
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

def create_enhanced_plot(model_stats):
    """Create enhanced scatter plot with optimized label positioning"""
    # Set up the plot with larger figure size
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors based on model performance tiers
    def get_color(accuracy):
        if accuracy >= 0.95:
            return '#1f77b4'  # Blue for excellent
        elif accuracy >= 0.85:
            return '#ff7f0e'  # Orange for good
        elif accuracy >= 0.70:
            return '#2ca02c'  # Green for moderate
        else:
            return '#d62728'  # Red for poor
    
    colors = [get_color(acc) for acc in model_stats['best_accuracy']]
    
    # Create scatter plot with different sizes based on accuracy
    sizes = 80 + (model_stats['best_accuracy'] - model_stats['best_accuracy'].min()) * 200
    
    scatter = ax.scatter(model_stats['model_size_mb'], 
                        model_stats['best_accuracy'], 
                        s=sizes,
                        c=colors,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1.5)
    
    # Create simplified model names for better readability
    def simplify_name(name):
        # Remove common words and make more compact
        name = name.replace('Bert + ', '').replace('bert + ', '')
        name = name.replace('Distil', 'D').replace('Mobile', 'M').replace('Tiny', 'T')
        return name
    
    model_stats['simple_name'] = model_stats['model_name'].apply(simplify_name)
    
    # Prepare text labels with better positioning strategy
    texts = []
    for idx, row in model_stats.iterrows():
        # Use simplified names for less clutter
        label = row['simple_name']
        
        text = ax.annotate(label,
                          xy=(row['model_size_mb'], row['best_accuracy']),
                          xytext=(5, 5),  # Small offset
                          textcoords='offset points',
                          fontsize=10,
                          fontweight='bold',
                          ha='left',
                          va='bottom',
                          bbox=dict(boxstyle='round,pad=0.4', 
                                   facecolor='white', 
                                   alpha=0.9,
                                   edgecolor='gray',
                                   linewidth=0.5),
                          arrowprops=dict(arrowstyle='->', 
                                        connectionstyle='arc3,rad=0.1',
                                        color='gray', 
                                        alpha=0.7,
                                        lw=1))
        texts.append(text)
    
    # Advanced text adjustment with more control
    adjust_text(texts,
                x=model_stats['model_size_mb'].values,
                y=model_stats['best_accuracy'].values,
                expand_points=(1.5, 1.5),
                expand_text=(1.3, 1.3),
                force_points=0.3,
                force_text=0.8,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.8),
                ax=ax)
    
    # Customize plot appearance
    ax.set_xlabel('Model Size (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy (5-fold CV)', fontsize=14, fontweight='bold')
    ax.set_title('Model Size vs Average Accuracy\nMultimodal Text Detection Models', 
                fontsize=16, fontweight='bold', pad=25)
    
    # Add sophisticated grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set axis limits with appropriate padding
    x_range = model_stats['model_size_mb'].max() - model_stats['model_size_mb'].min()
    y_range = model_stats['best_accuracy'].max() - model_stats['best_accuracy'].min()
    
    x_margin = x_range * 0.15
    y_margin = y_range * 0.08
    
    ax.set_xlim(model_stats['model_size_mb'].min() - x_margin, 
                model_stats['model_size_mb'].max() + x_margin)
    ax.set_ylim(model_stats['best_accuracy'].min() - y_margin, 
                model_stats['best_accuracy'].max() + y_margin)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add performance legend
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='Excellent (≥95%)', alpha=0.7, edgecolors='black'),
        plt.scatter([], [], c='#ff7f0e', s=100, label='Good (85-95%)', alpha=0.7, edgecolors='black'),
        plt.scatter([], [], c='#2ca02c', s=100, label='Moderate (70-85%)', alpha=0.7, edgecolors='black'),
        plt.scatter([], [], c='#d62728', s=100, label='Poor (<70%)', alpha=0.7, edgecolors='black')
    ]
    
    legend = ax.legend(handles=legend_elements, 
                      title='Performance Tier',
                      title_fontsize=12,
                      fontsize=10,
                      loc='upper right',
                      frameon=True,
                      fancybox=True,
                      shadow=True)
    legend.get_frame().set_alpha(0.9)
    
    # Add comprehensive statistics box
    best_model = model_stats.loc[model_stats['best_accuracy'].idxmax()]
    smallest_model = model_stats.loc[model_stats['model_size_mb'].idxmin()]
    
    stats_text = f"Dataset: {len(model_stats)} Models (5-fold CV)\n"
    stats_text += f"Best Performance: {best_model['model_name']} ({best_model['best_accuracy']:.3f})\n"
    stats_text += f"Smallest Model: {smallest_model['model_name']} ({smallest_model['model_size_mb']:.1f}MB)\n"
    stats_text += f"Size Range: {model_stats['model_size_mb'].min():.1f} - {model_stats['model_size_mb'].max():.1f} MB\n"
    stats_text += f"Accuracy Range: {model_stats['best_accuracy'].min():.3f} - {model_stats['best_accuracy'].max():.3f}"
    
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, 
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', 
                     facecolor='lightblue', 
                     alpha=0.9,
                     edgecolor='navy',
                     linewidth=1))
    
    # Add efficiency frontier line (Pareto front)
    sorted_models = model_stats.sort_values('model_size_mb')
    pareto_front = []
    max_acc_so_far = 0
    
    for _, row in sorted_models.iterrows():
        if row['best_accuracy'] > max_acc_so_far:
            pareto_front.append((row['model_size_mb'], row['best_accuracy']))
            max_acc_so_far = row['best_accuracy']
    
    if len(pareto_front) > 1:
        pareto_x, pareto_y = zip(*pareto_front)
        ax.plot(pareto_x, pareto_y, 'r--', alpha=0.6, linewidth=2, 
                label='Efficiency Frontier')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig, ax

def save_detailed_results(model_stats, output_path):
    """Save detailed results with efficiency metrics"""
    # Calculate efficiency score (accuracy per MB)
    model_stats['efficiency'] = model_stats['best_accuracy'] / model_stats['model_size_mb']
    
    # Sort by accuracy (descending)
    sorted_stats = model_stats.sort_values('best_accuracy', ascending=False).copy()
    sorted_stats['accuracy_rank'] = range(1, len(sorted_stats) + 1)
    
    # Sort by efficiency and add efficiency rank
    efficiency_sorted = model_stats.sort_values('efficiency', ascending=False).copy()
    efficiency_rank = {name: rank for rank, name in enumerate(efficiency_sorted['model_name'], 1)}
    sorted_stats['efficiency_rank'] = sorted_stats['model_name'].map(efficiency_rank)
    
    # Format for display
    sorted_stats['accuracy_pct'] = sorted_stats['best_accuracy'].apply(lambda x: f"{x:.3f}")
    sorted_stats['size_mb'] = sorted_stats['model_size_mb'].apply(lambda x: f"{x:.1f}")
    sorted_stats['efficiency_score'] = sorted_stats['efficiency'].apply(lambda x: f"{x:.6f}")
    
    # Select columns for output
    output_df = sorted_stats[['accuracy_rank', 'model_name', 'accuracy_pct', 'size_mb', 
                             'efficiency_score', 'efficiency_rank']].copy()
    output_df.columns = ['Acc_Rank', 'Model_Name', 'Avg_Accuracy', 'Size_MB', 
                        'Efficiency_Score', 'Eff_Rank']
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")
    
    return output_df

def main():
    # Load and process data
    csv_path = 'total_results.csv'
    model_stats = load_and_process_data(csv_path)
    
    print(f"Loaded {len(model_stats)} models with complete 5-fold results")
    
    # Create enhanced plot
    fig, ax = create_enhanced_plot(model_stats)
    
    # Save high-quality plot
    plot_path = 'model_size_vs_accuracy_enhanced.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Enhanced plot saved to: {plot_path}")
    
    # Save detailed results
    table_path = 'detailed_model_analysis.csv'
    results_table = save_detailed_results(model_stats, table_path)
    
    print(f"\nTop 5 Models by Accuracy:")
    print(results_table.head().to_string(index=False))
    
    print(f"\nTop 5 Most Efficient Models:")
    efficient_models = results_table.sort_values('Eff_Rank').head()
    print(efficient_models[['Model_Name', 'Avg_Accuracy', 'Size_MB', 'Efficiency_Score']].to_string(index=False))

if __name__ == "__main__":
    main()
