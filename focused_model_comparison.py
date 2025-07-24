#!/usr/bin/env python3
"""
Focused bar chart generator for individual model comparisons.
Creates separate bar graphs for each model comparing:
1. Eager mode vs Graph mode inference time
2. Guard time analysis per model

This script generates only the specific charts requested by the user.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style for better-looking charts
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Experimental configuration constants
WARMUP_RUNS = 10
TIMING_RUNS = 50
GUARD_RUNS = 20
DEFAULT_BATCH_SIZE = 32
INPUT_SHAPES = {
    'ResNet50': '(32, 3, 224, 224)',
    'ViT-Base': '(32, 3, 224, 224)', 
    'BERT-Base': '(32, 512)',
    'GPT2-Medium': '(32, 1024)',
    'DenseNet121': '(32, 3, 224, 224)',
    'DistilBERT': '(32, 128)'
}

def add_clean_experimental_description(fig, chart_type, model_name=None):
    """Add clean, minimal experimental methodology description"""
    
    if chart_type == 'inference':
        desc = f"""Experimental Setup: {WARMUP_RUNS} warmup + {TIMING_RUNS} timing runs | Hardware: GPU | Compilation: torch.compile()"""
        
    elif chart_type == 'guard':
        desc = f"""Guard Time Setup: {GUARD_RUNS} runs per test | Shape variations tested | Recompilation triggers measured"""
    
    else:
        desc = "Standard benchmarking methodology"
    
    # Add as clean footer text
    fig.text(0.5, 0.02, desc, ha='center', va='bottom', fontsize=8, 
             style='italic', color='gray')

def create_individual_model_inference_comparison(model_data, model_name):
    """
    Create a clean bar chart for a single model comparing eager vs graph mode inference time
    
    Args:
        model_data: DataFrame row or dict with inference time data for one model
        model_name: Name of the model
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract times (handle both dict and DataFrame formats)
    if isinstance(model_data, dict):
        eager_time = model_data.get('baseline_time', model_data.get('eager_time', 0))
        graph_time = model_data.get('compiled_time', model_data.get('graph_time', 0))
    else:
        eager_time = getattr(model_data, 'baseline_time', getattr(model_data, 'eager_time', 0))
        graph_time = getattr(model_data, 'compiled_time', getattr(model_data, 'graph_time', 0))
    
    # Convert to milliseconds for better readability
    eager_ms = eager_time * 1000
    graph_ms = graph_time * 1000
    
    # Calculate speedup
    speedup = eager_time / graph_time if graph_time > 0 else 0
    
    # Clean color scheme
    colors = ['#3498db', '#2ecc71' if speedup > 1.0 else '#e74c3c']
    
    # Create bars
    categories = ['Eager Mode', 'Compiled Mode']
    times = [eager_ms, graph_ms]
    
    bars = ax.bar(categories, times, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                f'{time_val:.1f}ms', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    if speedup > 0:
        speedup_color = '#27ae60' if speedup > 1.0 else '#c0392b'
        speedup_text = f'{speedup:.1f}× speedup' if speedup > 1.0 else f'{speedup:.1f}× slower'
        ax.text(0.5, 0.85, speedup_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold', 
                color=speedup_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=speedup_color, linewidth=2))
    
    # Clean formatting
    ax.set_title(f'{model_name} Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(times) * 1.2)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add input shape info
    input_shape = INPUT_SHAPES.get(model_name, 'Variable')
    ax.text(0.98, 0.02, f'Input: {input_shape}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, style='italic', color='gray')
    
    # Add experimental description
    add_clean_experimental_description(fig, 'inference', model_name)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig

def create_individual_model_guard_analysis(model_data, model_name):
    """
    Create clean guard time analysis bar chart for a single model
    
    Args:
        model_data: DataFrame row or dict with guard time data for one model
        model_name: Name of the model
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract guard data
    if isinstance(model_data, dict):
        guard_time = model_data.get('guard_check_time', 0)
        recompile_time = model_data.get('recompile_time', 0)
        guard_failure_rate = model_data.get('guard_failure_rate', 0)
        baseline_time = model_data.get('baseline_time', model_data.get('eager_time', 0))
    else:
        guard_time = getattr(model_data, 'guard_check_time', 0)
        recompile_time = getattr(model_data, 'recompile_time', 0)
        guard_failure_rate = getattr(model_data, 'guard_failure_rate', 0)
        baseline_time = getattr(model_data, 'baseline_time', getattr(model_data, 'eager_time', 0))
    
    # Chart 1: Guard Check Time
    guard_ms = guard_time * 1000
    bar1 = ax1.bar(['Guard Check Time'], [guard_ms], color='#3498db', 
                   alpha=0.8, edgecolor='white', linewidth=2, width=0.5)
    
    if guard_ms > 0:
        ax1.text(0, guard_ms + guard_ms * 0.05, f'{guard_ms:.2f}ms', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Calculate overhead percentage
        if baseline_time > 0:
            overhead_pct = (guard_time / baseline_time) * 100
            ax1.text(0, guard_ms * 0.5, f'{overhead_pct:.1f}%\noverhead', 
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='#f39c12', 
                              alpha=0.8, edgecolor='white'))
    
    ax1.set_title(f'{model_name} - Guard Overhead', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, guard_ms * 1.3 if guard_ms > 0 else 1)
    
    # Chart 2: Recompilation Analysis
    metrics = ['Recompile Time (s)', 'Failure Rate (%)']
    values = [recompile_time, guard_failure_rate * 100]
    colors = ['#e67e22', '#e74c3c' if guard_failure_rate > 0.2 else '#f39c12' if guard_failure_rate > 0.1 else '#27ae60']
    
    bars2 = ax2.bar(range(len(metrics)), values, color=colors, alpha=0.8, 
                    edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels
    for i, (bar, value, metric) in enumerate(zip(bars2, values, metrics)):
        if 'Time' in metric:
            label = f'{value:.1f}s'
        else:
            label = f'{value:.1f}%'
        
        max_val = max(values) if max(values) > 0 else 1
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title(f'{model_name} - Recompilation Stats', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time / Rate', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Fix the tick labels
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(['Recompile\nTime (s)', 'Failure\nRate (%)'], fontsize=9)
    
    # Clean up spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    plt.suptitle(f'{model_name} - Guard Time Analysis', fontsize=14, fontweight='bold', y=0.95)
    
    # Add experimental description
    add_clean_experimental_description(fig, 'guard', model_name)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.88)
    return fig

def generate_all_model_charts(data, output_dir="./model_charts"):
    """
    Generate separate charts for each model in the dataset
    
    Args:
        data: DataFrame with model performance data
        output_dir: Directory to save charts
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating clean individual model comparison charts...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Handle different model name columns
    model_col = 'model_name' if 'model_name' in data.columns else 'model'
    models = data[model_col].unique()
    
    inference_charts = []
    guard_charts = []
    
    for model in models:
        print(f"\nProcessing {model}...")
        
        # Get model data (use first row if multiple entries per model)
        model_data = data[data[model_col] == model].iloc[0]
        
        # 1. Create inference time comparison chart
        try:
            fig1 = create_individual_model_inference_comparison(model_data, model)
            filename1 = f"{output_dir}/{model.replace(' ', '_').replace('-', '_')}_inference_comparison.png"
            fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig1)
            inference_charts.append(filename1)
            print(f"  ✓ Inference comparison: {filename1}")
        except Exception as e:
            print(f"  ✗ Error creating inference chart: {e}")
        
        # 2. Create guard time analysis chart
        try:
            fig2 = create_individual_model_guard_analysis(model_data, model)
            filename2 = f"{output_dir}/{model.replace(' ', '_').replace('-', '_')}_guard_analysis.png"
            fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            guard_charts.append(filename2)
            print(f"  ✓ Guard analysis: {filename2}")
        except Exception as e:
            print(f"  ✗ Error creating guard chart: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Chart generation complete!")
    print(f"  • {len(inference_charts)} inference comparison charts")
    print(f"  • {len(guard_charts)} guard analysis charts")
    print(f"  • Total: {len(inference_charts) + len(guard_charts)} charts")
    
    return inference_charts, guard_charts

def load_sample_data():
    """Load or create sample data for testing"""
    models = ['ResNet50', 'ViT-Base', 'DenseNet121', 'DistilBERT']
    
    np.random.seed(42)
    data = {
        'model_name': models,
        'baseline_time': np.random.uniform(0.02, 0.15, len(models)),  # 20-150ms
        'compiled_time': [],
        'guard_check_time': np.random.uniform(0.0002, 0.003, len(models)),  # 0.2-3ms
        'recompile_time': np.random.uniform(0.5, 4.0, len(models)),  # 0.5-4 seconds
        'guard_failure_rate': np.random.uniform(0.05, 0.25, len(models)),  # 5-25%
    }
    
    # Generate realistic compiled times
    for i, baseline in enumerate(data['baseline_time']):
        if i % 3 == 0:  # Some models slower with compilation
            speedup = np.random.uniform(0.8, 0.95)
        else:  # Most models faster
            speedup = np.random.uniform(1.2, 2.5)
        data['compiled_time'].append(baseline / speedup)
    
    return pd.DataFrame(data)

def main():
    """Main function to generate all charts"""
    print("Clean Model Comparison Chart Generator")
    print("=====================================")
    print("Generating clean, professional bar charts for each model:")
    print("1. Eager vs Graph Mode Inference Time")
    print("2. Guard Time Analysis")
    print()
    
    # Load data (replace this with your actual data loading)
    data = load_sample_data()
    
    # Display data info
    print("Data Overview:")
    print(f"  • Models: {len(data)} models")
    print(f"  • Columns: {list(data.columns)}")
    print()
    
    # Generate all charts
    inference_charts, guard_charts = generate_all_model_charts(data)
    
    print(f"\nAll clean charts have been generated!")
    print(f"Check the './model_charts' directory for your enhanced bar graphs.")

if __name__ == "__main__":
    main() 