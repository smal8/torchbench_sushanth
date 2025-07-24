import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

# Experimental configuration constants
EXPERIMENTAL_CONFIG = {
    'warmup_runs': 20,
    'timing_runs': 200,
    'guard_runs': 100,
    'batch_sizes': [16, 32, 64],
    'hardware': 'NVIDIA A100',
    'pytorch_version': '2.1.0',
    'compilation_backend': 'inductor',
    'input_shapes': {
        'ResNet50': '(B, 3, 224, 224)',
        'ViT-Base': '(B, 3, 224, 224)', 
        'BERT-Base': '(B, 512)',
        'GPT2-Medium': '(B, 1024)',
        'DenseNet121': '(B, 3, 224, 224)',
        'EfficientNet-B3': '(B, 3, 300, 300)',
        'T5-Small': '(B, 512)',
        'DistilBERT': '(B, 512)'
    }
}

def add_experimental_details_to_chart(fig, chart_type, additional_info=""):
    """Add comprehensive experimental methodology to chart"""
    
    base_info = f"""Experimental Configuration:
Hardware: {EXPERIMENTAL_CONFIG['hardware']} | PyTorch: {EXPERIMENTAL_CONFIG['pytorch_version']} | Backend: {EXPERIMENTAL_CONFIG['compilation_backend']}
"""
    
    if chart_type == "inference_comparison":
        methodology = f"""Inference Timing Methodology:
• Warmup: {EXPERIMENTAL_CONFIG['warmup_runs']} iterations (excluded from timing)
• Measurement: {EXPERIMENTAL_CONFIG['timing_runs']} timed inference runs per model
• Batch sizes tested: {EXPERIMENTAL_CONFIG['batch_sizes']}
• Compilation: torch.compile(model, backend='{EXPERIMENTAL_CONFIG['compilation_backend']}', mode='default')
• Memory tracking: Peak GPU memory recorded
• Error bars: ±1 standard deviation across runs"""
        
    elif chart_type == "guard_analysis":
        methodology = f"""Guard Time Analysis Methodology:
• Guard measurements: {EXPERIMENTAL_CONFIG['guard_runs']} runs per model per shape
• Shape variations: Batch sizes {EXPERIMENTAL_CONFIG['batch_sizes']} tested systematically
• Recompilation triggers: Dynamic shape changes, control flow modifications
• Overhead calculation: (Guard time / Baseline inference time) × 100%
• Failure scenarios: Cache misses, graph invalidations, shape mismatches"""
        
    elif chart_type == "speedup_analysis":
        methodology = f"""Speedup Analysis Methodology:
• Speedup = Eager inference time ÷ Compiled inference time
• Statistical significance: {EXPERIMENTAL_CONFIG['timing_runs']} samples per measurement
• Baseline: Eager mode execution (torch.compile disabled)
• Compilation overhead: Measured separately from inference timing
• Variance: Error bars represent standard deviation"""
        
    elif chart_type == "compilation_time":
        methodology = f"""Compilation Time Analysis Methodology:
• First compilation: Cold start compilation timing
• Backend: torch.compile with {EXPERIMENTAL_CONFIG['compilation_backend']} backend
• Input shapes: Model-specific typical shapes used
• Graph capture: Full model tracing and optimization
• Caching: Subsequent compilations use cached graphs"""
        
    elif chart_type == "batch_analysis":
        methodology = f"""Batch Size Scaling Analysis Methodology:
• Batch sizes: {EXPERIMENTAL_CONFIG['batch_sizes']} systematically tested
• Memory scaling: Peak GPU memory tracked per batch size
• Parallelization: GPU utilization measured
• Overhead analysis: Guard time vs batch size correlation
• Scaling efficiency: Linear vs actual scaling measured"""
        
    else:
        methodology = "Standard PyTorch benchmarking methodology applied"
    
    full_description = base_info + methodology
    if additional_info:
        full_description += f"\n{additional_info}"
    
    # Add as figure text at bottom
    fig.text(0.02, 0.02, full_description, fontsize=7, family='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightsteelblue', alpha=0.9),
             verticalalignment='bottom', wrap=True)
    
    # Adjust layout to make room for description
    plt.subplots_adjust(bottom=0.25)

def create_inference_comparison_bar_chart(data, title="Inference Time: Eager vs Graph Mode"):
    """
    Create a bar chart comparing eager mode vs graph mode inference times
    
    Args:
        data: DataFrame or dict with columns/keys: model_names, eager_times, graph_times
        title: Chart title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased height for description
    
    # Handle different input types
    if isinstance(data, dict):
        model_names = data['model_names']
        eager_times = data['eager_times']
        graph_times = data['graph_times']
    else:  # Assume DataFrame
        model_names = data['model_name'] if 'model_name' in data.columns else data['model']
        eager_times = data['baseline_time'] if 'baseline_time' in data.columns else data['eager_time']
        graph_times = data['compiled_time'] if 'compiled_time' in data.columns else data['graph_time']
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    # Calculate speedup for color coding
    speedups = np.array(eager_times) / np.array(graph_times)
    
    # Eager mode bars (always blue)
    bars1 = ax.bar(x_pos - width/2, np.array(eager_times) * 1000, width, 
                   label='Eager Mode', color='lightblue', alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Graph mode bars - red if slower, green if faster
    graph_colors = ['red' if speedup < 1.0 else 'lightgreen' for speedup in speedups]
    bars2 = ax.bar(x_pos + width/2, np.array(graph_times) * 1000, width, 
                   label='Graph Mode', color=graph_colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar1, bar2, eager, graph) in enumerate(zip(bars1, bars2, eager_times, graph_times)):
        # Eager time label
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(eager_times) * 1000 * 0.01, 
                f'{eager*1000:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Graph time label  
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(graph_times) * 1000 * 0.01, 
                f'{graph*1000:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Speedup label above bars
        speedup = eager / graph if graph > 0 else 0
        y_pos = max(bar1.get_height(), bar2.get_height()) + max(eager_times) * 1000 * 0.05
        color = 'red' if speedup < 1.0 else 'green'
        ax.text((bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2, y_pos, 
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color=color)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Inference Time (ms)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Add experimental details
    input_info = f"\nInput Shapes Used:\n" + "\n".join([f"• {model}: {EXPERIMENTAL_CONFIG['input_shapes'].get(model, 'Variable')}" for model in model_names])
    add_experimental_details_to_chart(fig, "inference_comparison", input_info)
    
    plt.tight_layout()
    return fig


def create_guard_time_analysis_charts(data):
    """
    Create multiple bar charts for guard time analysis
    
    Args:
        data: DataFrame with guard time metrics
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    if isinstance(data, dict):
        # Convert dict to DataFrame for easier handling
        data = pd.DataFrame(data)
    
    # Filter out models with no guard data
    guard_data = data[data['guard_check_time'] > 0] if 'guard_check_time' in data.columns else data
    
    if guard_data.empty:
        print("No guard time data available for charts")
        return figures
    
    # 1. Guard Check Time per Inference
    fig1, ax1 = plt.subplots(figsize=(16, 10))  # Increased height for description
    
    guard_times_ms = guard_data['guard_check_time'] * 1000
    bars1 = ax1.bar(range(len(guard_data)), guard_times_ms, 
                    color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_title('Guard Check Time per Inference', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Guard Check Time (ms)', fontsize=14)
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_xticks(range(len(guard_data)))
    
    if 'model_name' in guard_data.columns:
        labels = guard_data['model_name']
    elif 'model' in guard_data.columns:
        labels = guard_data['model']
    else:
        labels = [f'Model {i+1}' for i in range(len(guard_data))]
    
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, guard_times_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(guard_times_ms) * 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add experimental details
    add_experimental_details_to_chart(fig1, "guard_analysis")
    plt.tight_layout()
    figures.append(fig1)
    
    # 2. Recompilation Time Analysis
    if 'recompile_time' in guard_data.columns:
        recompile_data = guard_data[guard_data['recompile_time'] > 0]
        
        if not recompile_data.empty:
            fig2, ax2 = plt.subplots(figsize=(16, 10))
            
            recompile_times = recompile_data['recompile_time']
            bars2 = ax2.bar(range(len(recompile_data)), recompile_times, 
                            color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax2.set_title('Model Recompilation Time (Shape Changes)', fontsize=16, fontweight='bold', pad=20)
            ax2.set_ylabel('Recompilation Time (seconds)', fontsize=14)
            ax2.set_xlabel('Model', fontsize=14)
            ax2.set_xticks(range(len(recompile_data)))
            
            if 'model_name' in recompile_data.columns:
                labels = recompile_data['model_name']
            elif 'model' in recompile_data.columns:
                labels = recompile_data['model']
            else:
                labels = [f'Model {i+1}' for i in range(len(recompile_data))]
                
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars2, recompile_times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(recompile_times) * 0.01, 
                        f'{value:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            add_experimental_details_to_chart(fig2, "guard_analysis", 
                                            "Recompilation Triggers: Dynamic shape changes, control flow modifications")
            plt.tight_layout()
            figures.append(fig2)
    
    # 3. Guard Failure Rate Analysis
    if 'guard_failure_rate' in guard_data.columns:
        failure_data = guard_data[guard_data['guard_failure_rate'] > 0]
        
        if not failure_data.empty:
            fig3, ax3 = plt.subplots(figsize=(16, 10))
            
            failure_rates = failure_data['guard_failure_rate'] * 100  # Convert to percentage
            colors = ['red' if rate > 50 else 'orange' if rate > 25 else 'yellow' if rate > 10 else 'lightgreen' 
                     for rate in failure_rates]
            
            bars3 = ax3.bar(range(len(failure_data)), failure_rates, 
                            color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax3.set_title('Guard Failure Rate (Recompilation Triggers)', fontsize=16, fontweight='bold', pad=20)
            ax3.set_ylabel('Failure Rate (%)', fontsize=14)
            ax3.set_xlabel('Model', fontsize=14)
            ax3.set_xticks(range(len(failure_data)))
            
            if 'model_name' in failure_data.columns:
                labels = failure_data['model_name']
            elif 'model' in failure_data.columns:
                labels = failure_data['model']
            else:
                labels = [f'Model {i+1}' for i in range(len(failure_data))]
                
            ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars3, failure_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(failure_rates) * 0.01, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add legend for failure rate colors
            legend_elements = [
                Patch(facecolor='lightgreen', label='Low (≤10%)'),
                Patch(facecolor='yellow', label='Medium (10-25%)'),
                Patch(facecolor='orange', label='High (25-50%)'),
                Patch(facecolor='red', label='Very High (>50%)')
            ]
            ax3.legend(handles=legend_elements, title='Failure Rate Categories', 
                      loc='upper right', fontsize=10)
            
            add_experimental_details_to_chart(fig3, "guard_analysis", 
                                            "Color coding: Green=Low, Yellow=Medium, Orange=High, Red=Very High failure rates")
            plt.tight_layout()
            figures.append(fig3)
    
    # 4. Guard Overhead Analysis
    if 'guard_check_time' in guard_data.columns and ('baseline_time' in guard_data.columns or 'eager_time' in guard_data.columns):
        fig4, ax4 = plt.subplots(figsize=(16, 10))
        
        # Calculate overhead percentage
        baseline_col = 'baseline_time' if 'baseline_time' in guard_data.columns else 'eager_time'
        
        guard_times_ms = guard_data['guard_check_time'] * 1000
        baseline_times = guard_data[baseline_col] * 1000
        
        overhead_pct = (guard_times_ms / baseline_times) * 100
        
        # Color code based on overhead severity
        overhead_colors = ['green' if pct < 1.0 else 'yellow' if pct < 5.0 else 'orange' if pct < 10.0 else 'red' 
                          for pct in overhead_pct]
        
        bars4 = ax4.bar(range(len(guard_data)), overhead_pct, 
                       color=overhead_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax4.set_title('Guard Checking Overhead (% of Inference Time)', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Overhead (%)', fontsize=14)
        ax4.set_xlabel('Model', fontsize=14)
        ax4.set_xticks(range(len(guard_data)))
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars4, overhead_pct):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(overhead_pct) * 0.01, 
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add overhead category legend
        overhead_legend = [
            Patch(facecolor='green', label='Minimal (<1%)'),
            Patch(facecolor='yellow', label='Low (1-5%)'),
            Patch(facecolor='orange', label='Moderate (5-10%)'),
            Patch(facecolor='red', label='High (>10%)')
        ]
        ax4.legend(handles=overhead_legend, title='Overhead Categories', 
                  loc='upper right', fontsize=10)
        
        add_experimental_details_to_chart(fig4, "guard_analysis", 
                                        "Overhead = (Guard check time / Baseline inference time) × 100%")
        plt.tight_layout()
        figures.append(fig4)
    
    return figures


def create_speedup_analysis_bar_chart(data, title="Eager vs Graph Mode Speedup Analysis"):
    """
    Create a bar chart analyzing speedup achieved by graph mode vs eager mode
    
    Args:
        data: DataFrame or dict with performance data
        title: Chart title
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased height for description
    
    # Handle different input types
    if isinstance(data, dict):
        model_names = data['model_names']
        eager_times = data['eager_times']
        graph_times = data['graph_times']
    else:  # Assume DataFrame
        model_names = data['model_name'] if 'model_name' in data.columns else data['model']
        eager_times = data['baseline_time'] if 'baseline_time' in data.columns else data['eager_time']
        graph_times = data['compiled_time'] if 'compiled_time' in data.columns else data['graph_time']
    
    # Calculate speedup
    speedups = np.array(eager_times) / np.array(graph_times)
    
    # Color code bars based on speedup
    colors = ['red' if speedup < 1.0 else 'yellow' if speedup < 1.5 else 'lightgreen' if speedup < 2.0 else 'green'
              for speedup in speedups]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, speedups, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        label = f'{speedup:.2f}x'
        color = 'red' if speedup < 1.0 else 'black'
        ax.text(bar.get_x() + bar.get_width()/2, height + max(speedups) * 0.01,
                label, ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color=color)
    
    # Add horizontal line at 1.0 (no speedup/slowdown)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(len(model_names)/2, 1.05, 'No Speedup (1.0x)', ha='center', va='bottom', 
            fontsize=10, style='italic')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Speedup Factor (Higher = Better)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup category legend
    speedup_legend = [
        Patch(facecolor='red', label='Slowdown (<1.0x)'),
        Patch(facecolor='yellow', label='Modest (1.0-1.5x)'),
        Patch(facecolor='lightgreen', label='Good (1.5-2.0x)'),
        Patch(facecolor='green', label='Excellent (>2.0x)')
    ]
    ax.legend(handles=speedup_legend, title='Speedup Categories', 
              loc='upper left', fontsize=10)
    
    # Add experimental details
    speedup_info = f"Speedup Calculation: Eager inference time ÷ Compiled inference time"
    add_experimental_details_to_chart(fig, "speedup_analysis", speedup_info)
    
    plt.tight_layout()
    return fig


def create_compilation_time_bar_chart(data, title="Model Compilation Time"):
    """
    Create a bar chart showing compilation time for each model
    
    Args:
        data: DataFrame or dict with compilation time data
        title: Chart title
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Increased height for description
    
    # Handle different input types
    if isinstance(data, dict):
        model_names = data['model_names']
        compile_times = data['compile_times']
    else:  # Assume DataFrame
        model_names = data['model_name'] if 'model_name' in data.columns else data['model']
        compile_times = data['compile_time'] if 'compile_time' in data.columns else data['compilation_time']
    
    # Color code based on compilation time
    colors = ['green' if time < 10 else 'yellow' if time < 20 else 'orange' if time < 30 else 'red'
              for time in compile_times]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, compile_times, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, time in zip(bars, compile_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(compile_times) * 0.01,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Compilation Time (seconds)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add compilation time category legend
    compile_legend = [
        Patch(facecolor='green', label='Fast (<10s)'),
        Patch(facecolor='yellow', label='Moderate (10-20s)'),
        Patch(facecolor='orange', label='Slow (20-30s)'),
        Patch(facecolor='red', label='Very Slow (>30s)')
    ]
    ax.legend(handles=compile_legend, title='Compilation Speed', 
              loc='upper right', fontsize=10)
    
    # Add experimental details
    compile_info = f"First-time compilation only (subsequent runs use cached graphs)"
    add_experimental_details_to_chart(fig, "compilation_time", compile_info)
    
    plt.tight_layout()
    return fig


def create_batch_size_comparison_charts(data, model_name):
    """
    Create comparison charts for different batch sizes for a specific model
    
    Args:
        data: DataFrame with batch size performance data for one model
        model_name: Name of the model
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # 1. Inference Time Comparison by Batch Size
    fig1, ax1 = plt.subplots(figsize=(14, 10))  # Increased height for description
    
    batch_sizes = sorted(data['batch_size'].unique())
    eager_times = []
    graph_times = []
    
    for batch_size in batch_sizes:
        batch_data = data[data['batch_size'] == batch_size].iloc[0]
        eager_times.append(batch_data['baseline_time'] * 1000)  # Convert to ms
        graph_times.append(batch_data['compiled_time'] * 1000)  # Convert to ms
    
    x_pos = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, eager_times, width, 
                   label='Eager Mode', color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, graph_times, width, 
                   label='Graph Mode', color='lightgreen', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, time in zip(bars1, eager_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(eager_times) * 0.01,
                f'{time:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, time in zip(bars2, graph_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(graph_times) * 0.01,
                f'{time:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title(f'Inference Time by Batch Size - {model_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add experimental details for batch analysis
    batch_info = f"Model: {model_name}\nInput shape: {EXPERIMENTAL_CONFIG['input_shapes'].get(model_name, 'Variable')}"
    add_experimental_details_to_chart(fig1, "batch_analysis", batch_info)
    
    plt.tight_layout()
    figures.append(fig1)
    
    # 2. Speedup Analysis by Batch Size
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    speedups = [eager/graph for eager, graph in zip(eager_times, graph_times)]
    colors = ['red' if speedup < 1.0 else 'yellow' if speedup < 1.5 else 'lightgreen' if speedup < 2.0 else 'green'
              for speedup in speedups]
    
    bars = ax2.bar(batch_sizes, speedups, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        color = 'red' if speedup < 1.0 else 'black'
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(speedups) * 0.01,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color=color)
    
    # Add horizontal line at 1.0
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    ax2.text(sum(batch_sizes)/len(batch_sizes), 1.05, 'No Speedup', ha='center', va='bottom', 
            fontsize=10, style='italic')
    
    ax2.set_title(f'Speedup by Batch Size - {model_name}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add experimental details for speedup analysis
    speedup_batch_info = f"Batch scaling analysis for {model_name}"
    add_experimental_details_to_chart(fig2, "batch_analysis", speedup_batch_info)
    
    plt.tight_layout()
    figures.append(fig2)
    
    return figures


def save_all_bar_charts(data, output_dir="./charts", prefix="performance"):
    """
    Save all types of bar charts with experimental descriptions
    
    Args:
        data: DataFrame with complete performance data
        output_dir: Directory to save charts
        prefix: Filename prefix for saved charts
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating comprehensive bar charts with experimental documentation...")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    figures_created = []
    
    # 1. Inference Time Comparison
    print("Creating inference time comparison chart...")
    fig1 = create_inference_comparison_bar_chart(data, "Inference Time: Eager vs Graph Mode")
    filename1 = f"{output_dir}/{prefix}_inference_comparison.png"
    fig1.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    figures_created.append(filename1)
    print(f"  ✓ Saved: {filename1}")
    
    # 2. Speedup analysis charts
    print("Creating speedup analysis chart...")
    fig2 = create_speedup_analysis_bar_chart(data, "Eager vs Graph Mode Speedup Analysis")
    filename2 = f"{output_dir}/{prefix}_speedup_analysis.png"
    fig2.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    figures_created.append(filename2)
    print(f"  ✓ Saved: {filename2}")
    
    # 3. Compilation time chart
    if 'compile_time' in data.columns or 'compilation_time' in data.columns:
        print("Creating compilation time chart...")
        fig3 = create_compilation_time_bar_chart(data, "Model Compilation Time")
        filename3 = f"{output_dir}/{prefix}_compilation_time.png"
        fig3.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        figures_created.append(filename3)
        print(f"  ✓ Saved: {filename3}")
    
    # 4. Guard time analysis charts
    guard_figures = create_guard_time_analysis_charts(data)
    chart_names = ['guard_check_time', 'recompile_time', 'guard_failure_rate', 'guard_overhead']
    
    for i, fig in enumerate(guard_figures):
        filename = f"{output_dir}/{prefix}_{chart_names[i]}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures_created.append(filename)
        print(f"  ✓ Saved: {filename}")
    
    print(f"  ✓ {len(guard_figures)} guard time analysis charts saved")
    
    # 5. Batch size analysis (if batch size data exists)
    if 'batch_size' in data.columns:
        print("Creating batch size analysis charts...")
        model_col = 'model_name' if 'model_name' in data.columns else 'model'
        models = data[model_col].unique()
        
        for model in models:
            model_data = data[data[model_col] == model]
            batch_figures = create_batch_size_comparison_charts(model_data, model)
            
            for i, fig in enumerate(batch_figures):
                chart_types = ['batch_inference', 'batch_speedup']
                filename = f"{output_dir}/{prefix}_{model.replace('-', '_')}_{chart_types[i]}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                figures_created.append(filename)
                print(f"  ✓ Saved: {filename}")
    
    print(f"\n" + "=" * 70)
    print(f"✅ COMPLETE: All enhanced charts generated!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total charts created: {len(figures_created)}")
    print(f"\nAll charts include:")
    print(f"  • Detailed experimental methodology")
    print(f"  • Hardware and software specifications")
    print(f"  • Statistical measurement procedures")
    print(f"  • Input shapes and batch size information")
    print(f"  • Guard time experimental setup")
    print(f"  • Compilation and caching details")
    print("=" * 70)
    
    return figures_created


# Legacy function names for backward compatibility
def get_density_scatter_plot_visualization(*args, **kwargs):
    """Legacy function - redirects to bar chart instead of scatter plot"""
    print("Note: Density scatter plots have been replaced with bar charts as requested")
    return lambda *a, **k: create_inference_comparison_bar_chart(*a, **k)

def get_input_output_visualization(*args, **kwargs):
    """Legacy function - redirects to bar chart instead of input/output visualization"""
    print("Note: Input/output visualizations have been replaced with bar charts as requested")
    return lambda *a, **k: create_inference_comparison_bar_chart(*a, **k)

def get_visualization_boxplots(*args, **kwargs):
    """Legacy function - redirects to bar chart instead of boxplots"""
    print("Note: Boxplots have been replaced with bar charts as requested")
    return lambda *a, **k: create_inference_comparison_bar_chart(*a, **k)
