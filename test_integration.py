#!/usr/bin/env python3
"""
Test script to verify the new bar chart visualizations work with existing data
"""

import pandas as pd
import numpy as np
from visualization import save_all_bar_charts, create_inference_comparison_bar_chart

def create_test_data_like_torchbench():
    """Create test data that matches the structure from torchbench scripts"""
    
    # Simulate data structure from test_torchbench_attention_models.py
    models = [
        'bert_base_uncased', 'distilbert_base_uncased', 'gpt2_medium',
        'resnet50', 'vit_base_patch16_224', 'densenet121',
        't5_small', 'wav2vec2_base'
    ]
    
    batch_sizes = [1, 4, 8, 16]
    model_types = ['nlp', 'nlp', 'nlp', 'vision', 'vision', 'vision', 'nlp', 'speech']
    
    data = []
    np.random.seed(42)
    
    for i, model in enumerate(models):
        for batch_size in batch_sizes:
            # Simulate realistic timing data
            base_time = np.random.uniform(0.01, 0.2)  # 10-200ms base
            batch_factor = batch_size ** 0.8
            
            baseline_time = base_time * batch_factor
            
            # Some models benefit more from compilation than others
            if 'bert' in model or 'gpt' in model:
                speedup = np.random.uniform(1.3, 2.1)  # NLP models often benefit more
            elif 'resnet' in model or 'densenet' in model:
                speedup = np.random.uniform(1.1, 1.6)  # Vision models vary
            else:
                speedup = np.random.uniform(0.9, 1.4)  # Some may be slower
            
            compiled_time = baseline_time / speedup
            
            # Guard metrics (simulated)
            guard_check_time = np.random.uniform(0.0001, 0.002)
            recompile_time = np.random.uniform(0.5, 3.0)
            guard_failure_rate = np.random.uniform(0.0, 0.3)
            recompile_count = np.random.randint(0, 3)
            
            data.append({
                'model_name': model,
                'model_type': model_types[i],
                'batch_size': batch_size,
                'baseline_time': baseline_time,
                'compiled_time': compiled_time,
                'compile_time': np.random.uniform(5.0, 30.0),
                'guard_check_time': guard_check_time,
                'recompile_time': recompile_time,
                'guard_failure_rate': guard_failure_rate,
                'recompile_count': recompile_count,
                'speedup': speedup,
                'baseline_throughput': batch_size / baseline_time,
                'compiled_throughput': batch_size / compiled_time
            })
    
    return pd.DataFrame(data)

def test_basic_functionality():
    """Test basic chart creation functionality"""
    print("Testing basic functionality...")
    
    # Create sample data
    data = create_test_data_like_torchbench()
    print(f"Created test data: {len(data)} rows, {len(data.columns)} columns")
    print(f"Models: {data['model_name'].unique()}")
    print(f"Batch sizes: {sorted(data['batch_size'].unique())}")
    
    # Test individual chart creation
    print("\nTesting individual chart creation...")
    
    # Get average data per model for main comparison
    model_avg = data.groupby('model_name').agg({
        'baseline_time': 'mean',
        'compiled_time': 'mean',
        'compile_time': 'first',
        'guard_check_time': 'mean',
        'recompile_time': 'mean',
        'guard_failure_rate': 'mean',
        'speedup': 'mean'
    }).reset_index()
    
    print(f"Model averages: {len(model_avg)} models")
    
    # Test main inference comparison chart
    try:
        fig = create_inference_comparison_bar_chart(
            model_avg, 
            "Test: Eager vs Graph Mode Inference"
        )
        fig.savefig('test_inference_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Inference comparison chart created successfully")
        fig.clf()
    except Exception as e:
        print(f"✗ Error creating inference comparison chart: {e}")
    
    # Test comprehensive chart generation
    print("\nTesting comprehensive chart generation...")
    try:
        save_all_bar_charts(
            model_avg,
            output_dir="./test_charts",
            prefix="integration_test"
        )
        print("✓ Comprehensive chart generation completed successfully")
    except Exception as e:
        print(f"✗ Error in comprehensive chart generation: {e}")
    
    return data, model_avg

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    # Test with minimal data
    minimal_data = pd.DataFrame({
        'model_name': ['test_model'],
        'baseline_time': [0.05],
        'compiled_time': [0.03]
    })
    
    try:
        fig = create_inference_comparison_bar_chart(minimal_data, "Minimal Data Test")
        fig.savefig('test_minimal.png', dpi=300, bbox_inches='tight')
        print("✓ Minimal data test passed")
        fig.clf()
    except Exception as e:
        print(f"✗ Minimal data test failed: {e}")
    
    # Test with no guard data
    no_guard_data = pd.DataFrame({
        'model_name': ['model1', 'model2'],
        'baseline_time': [0.05, 0.08],
        'compiled_time': [0.03, 0.06],
        'compile_time': [10.0, 15.0]
    })
    
    try:
        save_all_bar_charts(no_guard_data, output_dir="./test_no_guard", prefix="no_guard")
        print("✓ No guard data test passed")
    except Exception as e:
        print(f"✗ No guard data test failed: {e}")

def test_data_format_compatibility():
    """Test compatibility with different data formats"""
    print("\nTesting data format compatibility...")
    
    # Test alternative column names
    alt_data = pd.DataFrame({
        'model': ['alt_model1', 'alt_model2'],  # 'model' instead of 'model_name'
        'eager_time': [0.05, 0.08],            # 'eager_time' instead of 'baseline_time'
        'graph_time': [0.03, 0.06]             # 'graph_time' instead of 'compiled_time'
    })
    
    try:
        fig = create_inference_comparison_bar_chart(alt_data, "Alternative Column Names")
        fig.savefig('test_alt_columns.png', dpi=300, bbox_inches='tight')
        print("✓ Alternative column names test passed")
        fig.clf()
    except Exception as e:
        print(f"✗ Alternative column names test failed: {e}")

def display_performance_summary(data):
    """Display a summary of the performance analysis"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    model_avg = data.groupby('model_name').agg({
        'baseline_time': 'mean',
        'compiled_time': 'mean',
        'speedup': 'mean',
        'guard_check_time': 'mean'
    }).reset_index()
    
    # Models with best speedups
    best_speedups = model_avg.nlargest(3, 'speedup')
    print("\nTop 3 Models by Speedup:")
    for _, row in best_speedups.iterrows():
        print(f"  {row['model_name']}: {row['speedup']:.2f}x speedup")
    
    # Models with worst performance
    worst_speedups = model_avg.nsmallest(3, 'speedup')
    print("\nModels with Slowdowns:")
    for _, row in worst_speedups.iterrows():
        if row['speedup'] < 1.0:
            print(f"  {row['model_name']}: {row['speedup']:.2f}x (slower)")
    
    # Guard overhead analysis
    model_avg['guard_overhead_pct'] = (model_avg['guard_check_time'] / model_avg['baseline_time']) * 100
    high_overhead = model_avg[model_avg['guard_overhead_pct'] > 1.0]
    
    if not high_overhead.empty:
        print("\nModels with High Guard Overhead (>1%):")
        for _, row in high_overhead.iterrows():
            print(f"  {row['model_name']}: {row['guard_overhead_pct']:.2f}% overhead")
    
    print(f"\nOverall Statistics:")
    print(f"  Average speedup: {model_avg['speedup'].mean():.2f}x")
    print(f"  Models with speedup: {len(model_avg[model_avg['speedup'] > 1.0])}/{len(model_avg)}")
    print(f"  Average guard overhead: {model_avg['guard_overhead_pct'].mean():.3f}%")

def main():
    """Main test function"""
    print("Bar Chart Visualization Integration Test")
    print("="*50)
    
    # Test basic functionality
    data, model_avg = test_basic_functionality()
    
    # Test edge cases
    test_edge_cases()
    
    # Test data format compatibility
    test_data_format_compatibility()
    
    # Display performance summary
    display_performance_summary(data)
    
    print("\n" + "="*50)
    print("Integration test completed!")
    print("Check the following outputs:")
    print("  - test_inference_comparison.png")
    print("  - test_minimal.png") 
    print("  - test_alt_columns.png")
    print("  - ./test_charts/ directory")
    print("  - ./test_no_guard/ directory")
    print("="*50)

if __name__ == "__main__":
    main() 