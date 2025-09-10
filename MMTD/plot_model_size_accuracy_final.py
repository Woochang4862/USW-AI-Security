#!/usr/bin/env python3
"""
Final clean plot for model size vs accuracy with perfect label positioning
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data(csv_path):
    """Load CSV data and calculate average accuracy per model"""
    df = pd.read_csv(csv_path)
    
    # Filter out incomplete data (models with missing values)
    df = df.dropna(subset=['accuracy', 'model_size_mb'])
    
    # Group by model name and calculate average accuracy across folds
    model_stats = df.groupby('model_name').agg({
        'accuracy': 'mean',
        'model_size_mb': 'first',  # Model size should be same across folds
        'fold': 'count'  # Count number of folds
    }).reset_index()
    
    # Only include models with 5 folds
    model_stats = model_stats[model_stats['fold'] == 5]
    
    return model_stats

def create_final_plot(model_stats):
    """Create final clean scatter plot with perfect positioning"""
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
    
    colors = [get_color(acc) for acc in model_stats['accuracy']]
    
    # Create scatter plot
    scatter = ax.scatter(model_stats['model_size_mb'], 
                        model_stats['accuracy'], 
                        s=100,
                        c=colors,
                        alpha=0.8,
                        edgecolors='black',
                        linewidth=1)
    
    # Define specific text positions for each model to avoid overlapping
    # Format: (x_coordinate, y_coordinate, horizontal_alignment, vertical_alignment)
    text_positions = {
        'Tinybert + Vit-Tiny': (90, 0.98, 'left', 'center'),
        'Tinybert + Mobilevit': (90, 0.95, 'left', 'center'),
        'Tinybert + Deit': (420, 0.92, 'left', 'center'),
        'Tinybert + Beit': (350, 0.75, 'left', 'center'),
        'Distilbert + Vit-Tiny': (580, 0.92, 'left', 'center'),
        'Distilbert + Deit': (880, 0.90, 'left', 'center'),
        'Distilbert + Mobilevit': (580, 0.87, 'left', 'center'),
        'Distilbert + Beit': (900, 0.75, 'left', 'center'),
        'Bert + Mobilevit': (420, 0.86, 'right', 'center'),
        'Bert + Vit-Tiny': (480, 0.73, 'left', 'center'),
        'Bert + Deit': (720, 0.62, 'left', 'center'),
        'MMTD': (1000, 0.99, 'right', 'center'),
        'Mobilebert + Beit': (480, 0.53, 'left', 'center'),
        'Mobilebert + Vit-Tiny': (130, 0.52, 'left', 'center'),
        'Mobilebert + Mobilevit': (130, 0.50, 'left', 'center'),
        'Mobilebert + Deit': (450, 0.51, 'left', 'center')
    }
    
    # Add labels using text with absolute positioning
    for idx, row in model_stats.iterrows():
        model_name = row['model_name']
        
        # Get positioning for this model
        if model_name in text_positions:
            text_x, text_y, ha, va = text_positions[model_name]
        else:
            # Default position if not specified
            text_x = row['model_size_mb'] + 20
            text_y = row['accuracy'] + 0.01
            ha, va = 'left', 'center'
        
        # Create simplified label (remove redundant words)
        label = model_name
        
        # Add text with absolute coordinates
        ax.text(text_x, text_y, label,
                fontsize=9,
                fontweight='normal',
                ha=ha,
                va=va,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         alpha=0.9,
                         edgecolor='gray',
                         linewidth=0.5))
    
    # Customize plot appearance
    ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Average Accuracy', fontsize=14, fontweight='bold', pad=20)
    
    # Add light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with appropriate padding
    x_range = model_stats['model_size_mb'].max() - model_stats['model_size_mb'].min()
    y_range = model_stats['accuracy'].max() - model_stats['accuracy'].min()
    
    x_margin = x_range * 0.08
    y_margin = y_range * 0.05
    
    ax.set_xlim(model_stats['model_size_mb'].min() - x_margin, 
                model_stats['model_size_mb'].max() + x_margin)
    ax.set_ylim(model_stats['accuracy'].min() - y_margin, 
                model_stats['accuracy'].max() + y_margin)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add simple legend for colors only (positioned at bottom right)
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
              shadow=False,
              framealpha=0.9)
    
    plt.tight_layout()
    return fig, ax

def main():
    # Load and process data
    csv_path = 'total_results.csv'
    model_stats = load_and_process_data(csv_path)
    
    print(f"Loaded {len(model_stats)} models with complete 5-fold results")
    
    # Create final plot
    fig, ax = create_final_plot(model_stats)
    
    # Save plot
    plot_path = 'model_size_vs_accuracy_final.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Final plot saved to: {plot_path}")
    
    # Show top models
    top_models = model_stats.nlargest(5, 'accuracy')
    print(f"\nTop 5 Models:")
    for idx, row in model_stats.iterrows():
        print(f"{row['model_name']}: {row['accuracy']:.3f} ({row['model_size_mb']:.1f}MB)")

if __name__ == "__main__":
    main()
