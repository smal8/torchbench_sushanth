#!/usr/bin/env python3
"""
Demo script to show the new comprehensive visualizations for GAT/GCN benchmarks
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_sample_gat_data():
    """Create comprehensive sample GAT data for demonstration"""
    sample_results = []
    
    # GAT-Small data across all batch sizes
    gat_small_data = [
        {'model_name': 'GAT-Small', 'batch_size': 1, 'eager_time_ms': 1.06, 'graph_time_ms': 0.39, 'speedup': 2.71, 'compilation_time_s': 10.69, 'eager_memory_mb': 245, 'graph_memory_mb': 245, 'memory_overhead_mb': 0},
        {'model_name': 'GAT-Small', 'batch_size': 2, 'eager_time_ms': 1.98, 'graph_time_ms': 0.60, 'speedup': 3.33, 'compilation_time_s': 3.00, 'eager_memory_mb': 290, 'graph_memory_mb': 292, 'memory_overhead_mb': 2},
        {'model_name': 'GAT-Small', 'batch_size': 4, 'eager_time_ms': 3.81, 'graph_time_ms': 1.00, 'speedup': 3.82, 'compilation_time_s': 5.02, 'eager_memory_mb': 380, 'graph_memory_mb': 385, 'memory_overhead_mb': 5},
        {'model_name': 'GAT-Small', 'batch_size': 8, 'eager_time_ms': 7.73, 'graph_time_ms': 1.76, 'speedup': 4.39, 'compilation_time_s': 9.09, 'eager_memory_mb': 560, 'graph_memory_mb': 570, 'memory_overhead_mb': 10},
        {'model_name': 'GAT-Small', 'batch_size': 16, 'eager_time_ms': 15.02, 'graph_time_ms': 3.23, 'speedup': 4.66, 'compilation_time_s': 18.23, 'eager_memory_mb': 920, 'graph_memory_mb': 940, 'memory_overhead_mb': 20},
        {'model_name': 'GAT-Small', 'batch_size': 32, 'eager_time_ms': 29.22, 'graph_time_ms': 5.76, 'speedup': 5.07, 'compilation_time_s': 37.01, 'eager_memory_mb': 1640, 'graph_memory_mb': 1680, 'memory_overhead_mb': 40},
        {'model_name': 'GAT-Small', 'batch_size': 64, 'eager_time_ms': 59.98, 'graph_time_ms': 11.79, 'speedup': 5.09, 'compilation_time_s': 79.64, 'eager_memory_mb': 3080, 'graph_memory_mb': 3160, 'memory_overhead_mb': 80},
    ]
    
    # GAT-Medium data (partial)
    gat_medium_data = [
        {'model_name': 'GAT-Medium', 'batch_size': 1, 'eager_time_ms': 2.45, 'graph_time_ms': 0.58, 'speedup': 4.22, 'compilation_time_s': 15.32, 'eager_memory_mb': 512, 'graph_memory_mb': 515, 'memory_overhead_mb': 3},
        {'model_name': 'GAT-Medium', 'batch_size': 8, 'eager_time_ms': 18.45, 'graph_time_ms': 3.22, 'speedup': 5.73, 'compilation_time_s': 45.67, 'eager_memory_mb': 1280, 'graph_memory_mb': 1320, 'memory_overhead_mb': 40},
        {'model_name': 'GAT-Medium', 'batch_size': 32, 'eager_time_ms': 72.18, 'graph_time_ms': 10.45, 'speedup': 6.91, 'compilation_time_s': 89.23, 'eager_memory_mb': 4200, 'graph_memory_mb': 4320, 'memory_overhead_mb': 120},
    ]
    
    # GAT-Large data (partial)
    gat_large_data = [
        {'model_name': 'GAT-Large', 'batch_size': 1, 'eager_time_ms': 4.32, 'graph_time_ms': 0.89, 'speedup': 4.85, 'compilation_time_s': 22.14, 'eager_memory_mb': 1024, 'graph_memory_mb': 1030, 'memory_overhead_mb': 6},
        {'model_name': 'GAT-Large', 'batch_size': 8, 'eager_time_ms': 35.67, 'graph_time_ms': 5.12, 'speedup': 6.97, 'compilation_time_s': 67.89, 'eager_memory_mb': 2560, 'graph_memory_mb': 2640, 'memory_overhead_mb': 80},
        {'model_name': 'GAT-Large', 'batch_size': 32, 'eager_time_ms': 145.23, 'graph_time_ms': 18.76, 'speedup': 7.74, 'compilation_time_s': 134.56, 'eager_memory_mb': 8400, 'graph_memory_mb': 8680, 'memory_overhead_mb': 280},
    ]
    
    sample_results.extend(gat_small_data)
    sample_results.extend(gat_medium_data)
    sample_results.extend(gat_large_data)
    
    return sample_results

def create_batch_size_plots_demo(df, output_prefix="demo"):
    """Create batch size visualization demo"""
    models = sorted(df['model_name'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Inference Time vs Batch Size - Eager Mode
    for model in models:
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        ax1.plot(model_data['batch_size'], model_data['eager_time_ms'], 
                marker='o', linewidth=2, markersize=6, label=f'{model} (Eager)')
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Eager Mode: Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)
    
    # 2. Inference Time vs Batch Size - Graph Mode  
    for model in models:
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        ax2.plot(model_data['batch_size'], model_data['graph_time_ms'],
                marker='s', linewidth=2, markersize=6, label=f'{model} (Graph)')
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Inference Time (ms)', fontsize=12)
    ax2.set_title('Graph Mode: Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    
    # 3. Speedup vs Batch Size
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, model in enumerate(models):
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        ax3.plot(model_data['batch_size'], model_data['speedup'],
                marker='D', linewidth=3, markersize=8, label=model, 
                color=colors[i % len(colors)])
    
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Speedup (x)', fontsize=12)
    ax3.set_title('Speedup vs Batch Size', fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels(batch_sizes)
    
    # 4. Throughput Analysis
    df['eager_throughput'] = df['batch_size'] / (df['eager_time_ms'] / 1000)
    df['graph_throughput'] = df['batch_size'] / (df['graph_time_ms'] / 1000)
    
    for i, model in enumerate(models):
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        ax4.plot(model_data['batch_size'], model_data['eager_throughput'],
                marker='o', linewidth=2, markersize=6, linestyle='--', 
                label=f'{model} (Eager)', color=colors[i % len(colors)], alpha=0.7)
        ax4.plot(model_data['batch_size'], model_data['graph_throughput'],
                marker='s', linewidth=2, markersize=6, 
                label=f'{model} (Graph)', color=colors[i % len(colors)])
    
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax4.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, ncol=2)
    ax4.set_xticks(batch_sizes)
    ax4.set_xticklabels(batch_sizes)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_batch_size_analysis_demo.png', dpi=300, bbox_inches='tight')
    print(f"📈 Demo batch size analysis saved to {output_prefix}_batch_size_analysis_demo.png")

def create_summary_plots_demo(df, output_prefix="demo"):
    """Create summary visualization demo"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average Speedup comparison
    model_speedups = df.groupby('model_name')['speedup'].mean().sort_values(ascending=True)
    bars = ax1.barh(range(len(model_speedups)), model_speedups.values, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(model_speedups)))
    ax1.set_yticklabels(model_speedups.index, fontsize=10)
    ax1.set_xlabel('Average Speedup (x)', fontsize=12)
    ax1.set_title('Average Graph Mode Speedup by Model', fontsize=14, fontweight='bold')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}x', ha='left', va='center', fontsize=9)
    
    # 2. Compilation Time vs Speedup
    model_stats = df.groupby('model_name').agg({
        'compilation_time_s': 'mean',
        'speedup': 'mean'
    }).reset_index()
    
    scatter = ax2.scatter(model_stats['compilation_time_s'], model_stats['speedup'], 
                         s=100, alpha=0.7, c=range(len(model_stats)), cmap='viridis')
    ax2.set_xlabel('Average Compilation Time (s)', fontsize=12)
    ax2.set_ylabel('Average Speedup (x)', fontsize=12)
    ax2.set_title('Compilation Time vs Speedup', fontsize=14, fontweight='bold')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    for i, row in model_stats.iterrows():
        ax2.annotate(row['model_name'], (row['compilation_time_s'], row['speedup']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. Memory Usage Comparison
    memory_data = df.groupby('model_name')[['eager_memory_mb', 'memory_overhead_mb']].mean()
    x = range(len(memory_data))
    
    ax3.bar(x, memory_data['eager_memory_mb'], label='Eager Mode', alpha=0.8, color='lightgreen')
    ax3.bar(x, memory_data['memory_overhead_mb'], bottom=memory_data['eager_memory_mb'],
           label='Graph Mode Overhead', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax3.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(memory_data.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Efficiency Heatmap (Speedup vs Batch Size)
    pivot_data = df.pivot(index='model_name', columns='batch_size', values='speedup')
    im = ax4.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_xticklabels(pivot_data.columns)
    ax4.set_yticks(range(len(pivot_data.index)))
    ax4.set_yticklabels(pivot_data.index)
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Models', fontsize=12)
    ax4.set_title('Speedup Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not pd.isna(value):
                ax4.text(j, i, f'{value:.1f}', ha="center", va="center", fontsize=9)
    
    plt.colorbar(im, ax=ax4, label='Speedup (x)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary_analysis_demo.png', dpi=300, bbox_inches='tight')
    print(f"📈 Demo summary analysis saved to {output_prefix}_summary_analysis_demo.png")

def main():
    """Demonstrate the new visualization capabilities"""
    print("🎨 Creating Visualization Demos")
    print("="*80)
    
    # Create sample data
    sample_data = create_sample_gat_data()
    df = pd.DataFrame(sample_data)
    
    print(f"📊 Sample data created: {len(df)} data points")
    print(f"📊 Models: {list(df['model_name'].unique())}")
    print(f"📊 Batch sizes: {sorted(df['batch_size'].unique())}")
    
    # Create visualizations
    print(f"\n🎯 Creating batch size analysis charts...")
    create_batch_size_plots_demo(df, output_prefix="gat_demo")
    
    print(f"\n🎯 Creating summary analysis charts...")
    create_summary_plots_demo(df, output_prefix="gat_demo")
    
    # Show what files will be generated
    print(f"\n📁 Generated Visualization Files:")
    print(f"  📈 Batch Size Analysis: gat_demo_batch_size_analysis_demo.png")
    print(f"  📈 Summary Analysis: gat_demo_summary_analysis_demo.png")
    
    print(f"\n🚀 Key Insights from Sample Data:")
    print(f"  ✅ GAT-Small shows increasing speedup with batch size (2.71x → 5.09x)")
    print(f"  ✅ GAT-Large achieves highest speedups at large batch sizes (7.74x)")
    print(f"  ✅ Compilation time increases with model complexity and batch size")
    print(f"  ✅ Memory overhead is minimal compared to base usage")
    
    print(f"\n🎉 Demo completed! Run your actual benchmarks to see real results.")

if __name__ == "__main__":
    main() 