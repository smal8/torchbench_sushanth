"""
Model Testing: Tests 5 different Vision Transformer models:
vit_b_16 (Base model with 16x16 patches)
vit_b_32 (Base model with 32x32 patches)
vit_l_16 (Large model with 16x16 patches)
vit_l_32 (Large model with 32x32 patches)
vit_h_14 (Huge model with 14x14 patches)


Metrics:
Compile time: How long PyTorch 2's torch.compile() takes
Baseline inference time: Standard PyTorch model performance
Compiled inference time: Performance after compilation
Speedup: How much faster the compiled model is
Throughput: Images processed per second
"""

import torch
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import gc
import warnings
import os

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
BATCH_SIZES = [1, 4, 8, 16, 32]
NUM_WARMUP_RUNS = 100
NUM_TIMING_RUNS = 100
IMAGE_SIZE = 224

# Vision Transformer models to test - using modern weights parameter
VIT_MODELS = {
    'vit_b_16': lambda: models.vit_b_16(weights=None),
    'vit_b_32': lambda: models.vit_b_32(weights=None),
    'vit_l_16': lambda: models.vit_l_16(weights=None),
    'vit_l_32': lambda: models.vit_l_32(weights=None),
    'vit_h_14': lambda: models.vit_h_14(weights=None),
}

def measure_compile_time(model, dummy_input, compile_mode='default'):
    """Measure compilation time for a model with fallback options"""
    
    compile_modes = [compile_mode, 'reduce-overhead', 'max-autotune', 'default']
    
    for mode in compile_modes:
        try:
            print(f"      Trying compile mode: {mode}")
            start_time = time.perf_counter()
            
            if mode == 'default':
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False,
                    disable=False
                )
            elif mode == 'reduce-overhead':
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False,
                    dynamic=False
                )
            else:  # max-autotune
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False, 
                    dynamic=False
                )
            
            # First forward pass triggers compilation
            with torch.no_grad():
                _ = compiled_model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            compile_time = time.perf_counter() - start_time
            print(f"      Success with mode: {mode}")
            return compiled_model, compile_time, mode
            
        except Exception as e:
            error_msg = str(e)
            print(f"      Failed with mode {mode}: {error_msg[:100]}...")
            
            if 'compiled_model' in locals():
                del compiled_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
    
    # If all compilation modes fail, try TorchScript as fallback
    try:
        print(f"      Trying TorchScript fallback...")
        start_time = time.perf_counter()
        
        # TorchScript compilation
        model.eval()
        compiled_model = torch.jit.trace(model, dummy_input)
        compiled_model = torch.jit.optimize_for_inference(compiled_model)
        
        with torch.no_grad():
            _ = compiled_model(dummy_input)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        compile_time = time.perf_counter() - start_time
        print(f"      Success with TorchScript fallback")
        return compiled_model, compile_time, "torchscript_fallback"
        
    except Exception as e:
        print(f"      TorchScript fallback failed: {str(e)[:50]}...")
        if 'compiled_model' in locals():
            del compiled_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # If all compilation modes and fallbacks fail, return None
    print(f"      All compilation modes and fallbacks failed")
    return None, None, None

def measure_inference_time(model, dummy_input, num_runs=NUM_TIMING_RUNS, is_eager=False):
    """Measure inference time for a model"""
    model.eval()
    
    # Ensure eager mode for baseline measurements
    if is_eager:
        # Disable any compilation for baseline
        torch._dynamo.reset()
        
    # Warmup runs
    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Timing runs using perf_counter for better precision
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start_time)
    
    return np.mean(times), np.std(times)

def test_model_performance():
    """Test all models with different batch sizes"""
    results = defaultdict(list)
    successful_tests = 0
    total_tests = len(VIT_MODELS) * len(BATCH_SIZES)
    
    print(f"Total planned tests: {total_tests} ({len(VIT_MODELS)} models × {len(BATCH_SIZES)} batch sizes)")
    
    for model_idx, (model_name, model_fn) in enumerate(VIT_MODELS.items()):
        print(f"\nTesting {model_name} ({model_idx + 1}/{len(VIT_MODELS)})...")
        
        for batch_idx, batch_size in enumerate(BATCH_SIZES):
            test_num = model_idx * len(BATCH_SIZES) + batch_idx + 1
            print(f"  Test {test_num}/{total_tests} - Batch size: {batch_size}")
            
            baseline_model = None
            compiled_model = None
            dummy_input = None
            
            try:
                # Create dummy input
                dummy_input = torch.randn(
                    batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, 
                    device=device, 
                    dtype=torch.float32
                )
                
                # Load baseline model (eager mode)
                print(f"    Loading baseline model...")
                baseline_model = model_fn()
                baseline_model = baseline_model.to(device)
                baseline_model.eval()
                
                # Ensure we're in eager mode for baseline
                torch._dynamo.reset()
                
                # Measure baseline (eager mode) inference time
                print(f"    Measuring baseline (eager) performance...")
                baseline_time, baseline_std = measure_inference_time(
                    baseline_model, dummy_input, is_eager=True
                )
                
                # Create a fresh model copy for compilation
                print(f"    Loading model for compilation...")
                compiled_model_base = model_fn()
                compiled_model_base = compiled_model_base.to(device)
                compiled_model_base.eval()
                
                # Measure compile time and get compiled model
                print(f"    Compiling model...")
                compiled_model, compile_time, compile_mode = measure_compile_time(
                    compiled_model_base, dummy_input
                )
                
                if compiled_model is None:
                    # If compilation failed, record results with failed compilation
                    print(f"    Compilation failed - using baseline times")
                    compiled_time = baseline_time
                    compiled_std = baseline_std
                    compile_time = 0.0
                    compile_mode = "failed"
                    speedup = 1.0
                else:
                    # Measure compiled inference time
                    print(f"    Measuring compiled performance...")
                    compiled_time, compiled_std = measure_inference_time(compiled_model, dummy_input)
                    speedup = baseline_time / compiled_time if compiled_time > 0 else 1.0
                
                # Calculate throughput (images/second)
                baseline_throughput = batch_size / baseline_time if baseline_time > 0 else 0
                compiled_throughput = batch_size / compiled_time if compiled_time > 0 else 0
                
                # Store results
                results['model'].append(model_name)
                results['batch_size'].append(batch_size)
                results['compile_time'].append(compile_time)
                results['compile_mode'].append(compile_mode)
                results['baseline_time'].append(baseline_time)
                results['baseline_std'].append(baseline_std)
                results['compiled_time'].append(compiled_time)
                results['compiled_std'].append(compiled_std)
                results['speedup'].append(speedup)
                results['baseline_throughput'].append(baseline_throughput)
                results['compiled_throughput'].append(compiled_throughput)
                
                print(f"    ✓ Results:")
                print(f"      Compile time: {compile_time:.3f}s (mode: {compile_mode})")
                print(f"      Baseline (eager): {baseline_time:.3f}±{baseline_std:.3f}s")
                print(f"      Compiled: {compiled_time:.3f}±{compiled_std:.3f}s")
                print(f"      Speedup: {speedup:.2f}x")
                print(f"      Throughput - Baseline: {baseline_throughput:.1f} img/s, Compiled: {compiled_throughput:.1f} img/s")
                
                successful_tests += 1
                
            except Exception as e:
                print(f"    ✗ Error with {model_name} batch_size={batch_size}: {str(e)[:100]}...")
                # Still record a failed result to maintain the full matrix
                results['model'].append(model_name)
                results['batch_size'].append(batch_size)
                results['compile_time'].append(0.0)
                results['compile_mode'].append("error")
                results['baseline_time'].append(0.0)
                results['baseline_std'].append(0.0)
                results['compiled_time'].append(0.0)
                results['compiled_std'].append(0.0)
                results['speedup'].append(0.0)
                results['baseline_throughput'].append(0.0)
                results['compiled_throughput'].append(0.0)
                
            finally:
                # Clean up memory thoroughly
                if dummy_input is not None:
                    del dummy_input
                if baseline_model is not None:
                    del baseline_model
                if compiled_model is not None:
                    del compiled_model
                if 'compiled_model_base' in locals():
                    del compiled_model_base
                    
                # Reset dynamo state
                torch._dynamo.reset()
                
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
    
    print(f"\nCompleted: {successful_tests}/{total_tests} tests successful")
    
    # Verify we have all combinations
    df = pd.DataFrame(results)
    if not df.empty:
        print(f"Results matrix: {len(df)} total results")
        print(f"Models tested: {sorted(df['model'].unique())}")
        print(f"Batch sizes tested: {sorted(df['batch_size'].unique())}")
        
        # Check for missing combinations
        expected_combinations = len(VIT_MODELS) * len(BATCH_SIZES)
        actual_combinations = len(df)
        if actual_combinations == expected_combinations:
            print(f"✓ Complete matrix: all {expected_combinations} combinations tested")
        else:
            print(f"⚠ Incomplete matrix: {actual_combinations}/{expected_combinations} combinations")
    
    return df

def create_visualizations(df):
    """Create separate bar charts for each model"""
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Filter out error cases for visualization
    df_viz = df[df['baseline_time'] > 0].copy()
    
    if df_viz.empty:
        print("No valid data to visualize")
        return
        
    models = sorted(df_viz['model'].unique())
    
    if not models:
        print("No models found in valid data")
        return
    
    # Create separate figure for each model
    for model_name in models:
        model_data = df_viz[df_viz['model'] == model_name].sort_values('batch_size')
        
        if model_data.empty:
            print(f"No valid data for model {model_name}")
            continue
            
        batch_sizes = model_data['batch_size'].tolist()
        baseline_times = model_data['baseline_time'].tolist()
        compiled_times = model_data['compiled_time'].tolist()
        compile_times = model_data['compile_time'].tolist()
        speedups = model_data['speedup'].tolist()
        baseline_throughput = model_data['baseline_throughput'].tolist()
        compiled_throughput = model_data['compiled_throughput'].tolist()
        
        # Create figure with 4 subplots for this model
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Analysis: {model_name}', fontsize=16, fontweight='bold', y=0.98)
        
        x_pos = np.arange(len(batch_sizes))
        width = 0.35
        
        # 1. Inference Time Comparison
        bars1 = ax1.bar(x_pos - width/2, baseline_times, width, label='Baseline (Eager)', 
                        alpha=0.8, color='lightcoral')
        bars2 = ax1.bar(x_pos + width/2, compiled_times, width, label='Compiled', 
                        alpha=0.8, color='lightgreen')
        
        ax1.set_title('Inference Time by Batch Size', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Inference Time (seconds)')
        ax1.set_xlabel('Batch Size')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(batch_sizes)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Add value labels
        for i, (bar1, bar2, baseline, compiled) in enumerate(zip(bars1, bars2, baseline_times, compiled_times)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, baseline * 1.1, f'{baseline:.3f}s', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax1.text(bar2.get_x() + bar2.get_width()/2, compiled * 1.1, f'{compiled:.3f}s', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Compilation Time
        bars = ax2.bar(x_pos, compile_times, alpha=0.8, color='skyblue')
        ax2.set_title('Compilation Time by Batch Size', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Compilation Time (seconds)')
        ax2.set_xlabel('Batch Size')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(batch_sizes)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, compile_times)):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(compile_times)*0.01, 
                        f'{value:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Speedup Analysis
        colors = ['red' if x < 1.0 else 'orange' if x < 1.2 else 'lightgreen' if x < 1.5 else 'darkgreen' 
                  for x in speedups]
        bars = ax3.bar(x_pos, speedups, alpha=0.8, color=colors)
        ax3.set_title('Speedup by Batch Size', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Speedup (x times faster)')
        ax3.set_xlabel('Batch Size')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(batch_sizes)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add speedup value labels
        for i, (bar, value) in enumerate(zip(bars, speedups)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend for speedup colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.8, label='Slowdown (<1.0x)'),
                          Patch(facecolor='orange', alpha=0.8, label='Minimal (1.0-1.2x)'),
                          Patch(facecolor='lightgreen', alpha=0.8, label='Good (1.2-1.5x)'),
                          Patch(facecolor='darkgreen', alpha=0.8, label='Excellent (>1.5x)')]
        ax3.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # 4. Throughput Comparison
        bars1 = ax4.bar(x_pos - width/2, baseline_throughput, width, label='Baseline (Eager)', 
                        alpha=0.8, color='lightcoral')
        bars2 = ax4.bar(x_pos + width/2, compiled_throughput, width, label='Compiled', 
                        alpha=0.8, color='lightgreen')
        
        ax4.set_title('Throughput by Batch Size', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Throughput (images/second)')
        ax4.set_xlabel('Batch Size')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(batch_sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add throughput value labels
        for i, (bar1, bar2, baseline, compiled) in enumerate(zip(bars1, bars2, baseline_throughput, compiled_throughput)):
            ax4.text(bar1.get_x() + bar1.get_width()/2, baseline + max(baseline_throughput)*0.01, 
                    f'{baseline:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax4.text(bar2.get_x() + bar2.get_width()/2, compiled + max(compiled_throughput)*0.01, 
                    f'{compiled:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual model chart
        filename = f'vit_benchmark_{model_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved for {model_name}: {filename}")
        
        # Close the figure to free memory
        plt.close()
    
    # Also create a summary chart with all models
    create_summary_chart(df_viz)

def create_summary_chart(df):
    """Create a summary chart showing all models together"""
    
    if df.empty:
        return
        
    models = sorted(df['model'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Summary: All Models', fontsize=16, fontweight='bold', y=0.98)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 1. Inference Time Comparison
    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('batch_size')
        if not model_data.empty:
            ax1.plot(model_data['batch_size'], model_data['baseline_time'], 
                    'o-', label=f'{model} (eager)', alpha=0.7, color=colors[i])
            ax1.plot(model_data['batch_size'], model_data['compiled_time'], 
                    's--', label=f'{model} (compiled)', alpha=0.7, color=colors[i])
    
    ax1.set_title('Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Inference Time (seconds)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Speedup Comparison
    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('batch_size')
        if not model_data.empty:
            ax2.plot(model_data['batch_size'], model_data['speedup'], 
                    'o-', label=model, linewidth=2, color=colors[i])
    
    ax2.set_title('Speedup vs Batch Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (x times faster)')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    # 3. Compilation Time by Model
    compile_data = df.groupby('model')['compile_time'].first().sort_index()
    bars = ax3.bar(range(len(compile_data)), compile_data.values, 
                   alpha=0.8, color='skyblue')
    ax3.set_title('First Compilation Time by Model', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Compilation Time (seconds)')
    ax3.set_xlabel('Model')
    ax3.set_xticks(range(len(compile_data)))
    ax3.set_xticklabels(compile_data.index, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, compile_data.values):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(compile_data.values)*0.01, 
                    f'{value:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Average Throughput by Model
    throughput_data = df.groupby('model')[['baseline_throughput', 'compiled_throughput']].mean().sort_index()
    x_pos = np.arange(len(throughput_data))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, throughput_data['baseline_throughput'], width, 
                    label='Baseline (Eager)', alpha=0.8, color='lightcoral')
    bars2 = ax4.bar(x_pos + width/2, throughput_data['compiled_throughput'], width, 
                    label='Compiled', alpha=0.8, color='lightgreen')
    
    ax4.set_title('Average Throughput by Model', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Throughput (images/second)')
    ax4.set_xlabel('Model')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(throughput_data.index, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('vit_benchmark_summary.png', dpi=300, bbox_inches='tight')
    print("Summary chart saved: vit_benchmark_summary.png")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if df.empty:
        print("No successful tests to analyze")
        return
    
    # Filter out error cases for statistics
    df_valid = df[df['baseline_time'] > 0].copy()
    
    if df_valid.empty:
        print("No valid test results to analyze")
        return
    
    # Test completion matrix
    print(f"\nTest Completion Matrix:")
    completion_matrix = df.pivot_table(
        index='model', 
        columns='batch_size', 
        values='baseline_time', 
        aggfunc=lambda x: 'PASS' if x.iloc[0] > 0 else 'FAIL'
    )
    print(completion_matrix)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total planned tests: {len(VIT_MODELS) * len(BATCH_SIZES)}")
    print(f"Completed tests: {len(df)}")
    print(f"Successful tests: {len(df_valid)}")
    print(f"Success rate: {len(df_valid)/len(df)*100:.1f}%")
    
    if 'compile_mode' in df_valid.columns:
        successful_compiles = df_valid[df_valid['compile_mode'] != 'failed']
        print(f"Successful compilations: {len(successful_compiles)}/{len(df_valid)}")
        if not successful_compiles.empty:
            print(f"Average compile time: {successful_compiles['compile_time'].mean():.2f}±{successful_compiles['compile_time'].std():.2f}s")
    
    print(f"Average speedup: {df_valid['speedup'].mean():.2f}x")
    print(f"Best speedup: {df_valid['speedup'].max():.2f}x")
    print(f"Tests with >1.5x speedup: {(df_valid['speedup'] > 1.5).sum()}/{len(df_valid)} cases")
    
    # By model statistics
    print(f"\nBy Model:")
    model_stats = df_valid.groupby('model').agg({
        'compile_time': 'first',
        'speedup': ['mean', 'max'],
        'baseline_throughput': 'mean',
        'compiled_throughput': 'mean'
    }).round(2)
    print(model_stats)
    
    # By batch size statistics
    print(f"\nBy Batch Size:")
    batch_stats = df_valid.groupby('batch_size').agg({
        'speedup': ['mean', 'std'],
        'compile_time': 'mean',
        'baseline_throughput': 'mean',
        'compiled_throughput': 'mean'
    }).round(2)
    print(batch_stats)
    
    # Compilation mode statistics
    if 'compile_mode' in df_valid.columns:
        print(f"\nBy Compilation Mode:")
        mode_stats = df_valid.groupby('compile_mode').agg({
            'speedup': ['count', 'mean'],
            'compile_time': 'mean'
        }).round(2)
        print(mode_stats)

if __name__ == "__main__":
    print("Starting Vision Transformer Benchmark...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Models to test: {list(VIT_MODELS.keys())}")
    print(f"Warmup runs: {NUM_WARMUP_RUNS}, Timing runs: {NUM_TIMING_RUNS}")
    print(f"Expected total tests: {len(VIT_MODELS)} × {len(BATCH_SIZES)} = {len(VIT_MODELS) * len(BATCH_SIZES)}")
    
    # Run the benchmark
    results_df = test_model_performance()
    
    if not results_df.empty:
        # Save results to CSV
        results_df.to_csv('vit_benchmark_results.csv', index=False)
        print(f"\nResults saved to: vit_benchmark_results.csv")
        
        # Print summary statistics
        print_summary_statistics(results_df)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(results_df)
        print("\n📊 Charts Generated:")
        print("- Individual model charts: vit_benchmark_<model_name>.png")
        print("- Summary chart: vit_benchmark_summary.png")
        print(f"- Total files created: {len(results_df[results_df['baseline_time'] > 0]['model'].unique()) + 1}")
        
        print("\nBenchmark completed!")
    else:
        print("No results collected. All models failed to execute.")