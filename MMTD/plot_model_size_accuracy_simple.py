#!/usr/bin/env python3
"""
Simple and clean plot for model size vs accuracy without overlapping labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def manual_label_positioning(model_stats):
    """Manually position labels to avoid overlapping"""
    # Create a copy for positioning
    positioning = model_stats.copy()
    
    # Define manual offsets for each model to avoid overlapping
    label_offsets = {
        'Tinybert + Vit-Tiny': (10, 15),
        'Tinybert + Mobilevit': (10, -20),
        'Tinybert + Deit': (15, 10),
        'Tinybert + Beit': (-80, -15),
        'Distilbert + Vit-Tiny': (15, 10),
        'Distilbert + Deit': (15, 10),
        'Distilbert + Mobilevit': (15, -20),
        'Distilbert + Beit': (15, 10),
        'Bert + Vit-Tiny': (15, 10),
        'Bert + Deit': (15, -20),
        'Bert + Mobilevit': (15, 10),
        'Mobilebert + Vit-Tiny': (15, 10),
        'Mobilebert + Deit': (15, -20),
        'Mobilebert + Mobilevit': (15, 10),
        'Mobilebert + Beit': (15, 10)
    }
    
    return label_offsets

def create_simple_plot(model_stats):
    """Create simple and clean scatter plot"""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
    
    # Create scatter plot
    scatter = ax.scatter(model_stats['model_size_mb'], 
                        model_stats['best_accuracy'], 
                        s=80,
                        c=colors,
                        alpha=0.8,
                        edgecolors='black',
                        linewidth=1)
    
    # Get manual label positions
    label_offsets = manual_label_positioning(model_stats)
    
    # Add labels with manual positioning
    for idx, row in model_stats.iterrows():
        model_name = row['model_name']
        x, y = row['model_size_mb'], row['best_accuracy']
        
        # Get offset for this model
        offset_x, offset_y = label_offsets.get(model_name, (10, 10))
        
        # Create simplified label
        label = model_name.replace('bert + ', '').replace('Bert + ', '')
        
        # Add text with manual offset
        ax.annotate(label,
                   xy=(x, y),
                   xytext=(x + offset_x, y + offset_y * 0.001),  # Convert y offset to data units
                   fontsize=9,
                   fontweight='normal',
                   ha='left' if offset_x > 0 else 'right',
                   va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            alpha=0.8,
                            edgecolor='gray',
                            linewidth=0.5))
    
    # Customize plot appearance
    ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Average Accuracy', fontsize=14, fontweight='bold', pad=20)
    
    # Add light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with padding
    x_range = model_stats['model_size_mb'].max() - model_stats['model_size_mb'].min()
    y_range = model_stats['best_accuracy'].max() - model_stats['best_accuracy'].min()
    
    x_margin = x_range * 0.1
    y_margin = y_range * 0.05
    
    ax.set_xlim(model_stats['model_size_mb'].min() - x_margin, 
                model_stats['model_size_mb'].max() + x_margin)
    ax.set_ylim(model_stats['best_accuracy'].min() - y_margin, 
                model_stats['best_accuracy'].max() + y_margin)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add simple legend for colors only
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=60, label='Excellent (≥95%)', alpha=0.8, edgecolors='black'),
        plt.scatter([], [], c='#ff7f0e', s=60, label='Good (85-95%)', alpha=0.8, edgecolors='black'),
        plt.scatter([], [], c='#2ca02c', s=60, label='Moderate (70-85%)', alpha=0.8, edgecolors='black'),
        plt.scatter([], [], c='#d62728', s=60, label='Poor (<70%)', alpha=0.8, edgecolors='black')
    ]
    
    ax.legend(handles=legend_elements, 
              loc='lower right',
              fontsize=9,
              frameon=True,
              fancybox=True,
              shadow=False)
    
    plt.tight_layout()
    return fig, ax

def main():
    # Load and process data
    csv_path = 'total_results.csv'
    model_stats = load_and_process_data(csv_path)
    
    print(f"Loaded {len(model_stats)} models with complete 5-fold results")
    
    # Create simple plot
    fig, ax = create_simple_plot(model_stats)
    
    # Save plot
    plot_path = 'model_size_vs_accuracy_simple.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simple plot saved to: {plot_path}")
    
    # Show statistics
    best_model = model_stats.loc[model_stats['best_accuracy'].idxmax()]
    smallest_model = model_stats.loc[model_stats['model_size_mb'].idxmin()]
    
    print(f"\nBest Performance: {best_model['model_name']} ({best_model['best_accuracy']:.3f})")
    print(f"Smallest Model: {smallest_model['model_name']} ({smallest_model['model_size_mb']:.1f}MB)")

if __name__ == "__main__":
    main()
