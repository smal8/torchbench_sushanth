#!/usr/bin/env python3
"""
Test script to demonstrate the batch size comparison table format
"""

import pandas as pd

def create_sample_results():
    """Create sample results to demonstrate the format"""
    sample_results = [
        # GAT-Small results
        {'model_name': 'GAT-Small', 'batch_size': 1, 'eager_time_ms': 1.06, 'graph_time_ms': 0.39, 'speedup': 2.71},
        {'model_name': 'GAT-Small', 'batch_size': 2, 'eager_time_ms': 1.98, 'graph_time_ms': 0.60, 'speedup': 3.33},
        {'model_name': 'GAT-Small', 'batch_size': 4, 'eager_time_ms': 3.81, 'graph_time_ms': 1.00, 'speedup': 3.82},
        {'model_name': 'GAT-Small', 'batch_size': 8, 'eager_time_ms': 7.73, 'graph_time_ms': 1.76, 'speedup': 4.39},
        {'model_name': 'GAT-Small', 'batch_size': 16, 'eager_time_ms': 15.02, 'graph_time_ms': 3.23, 'speedup': 4.66},
        {'model_name': 'GAT-Small', 'batch_size': 32, 'eager_time_ms': 29.22, 'graph_time_ms': 5.76, 'speedup': 5.07},
        {'model_name': 'GAT-Small', 'batch_size': 64, 'eager_time_ms': 59.98, 'graph_time_ms': 11.79, 'speedup': 5.09},
        
        # GAT-Medium results (sample)
        {'model_name': 'GAT-Medium', 'batch_size': 1, 'eager_time_ms': 2.15, 'graph_time_ms': 0.58, 'speedup': 3.71},
        {'model_name': 'GAT-Medium', 'batch_size': 8, 'eager_time_ms': 18.45, 'graph_time_ms': 3.22, 'speedup': 5.73},
        {'model_name': 'GAT-Medium', 'batch_size': 32, 'eager_time_ms': 72.18, 'graph_time_ms': 10.45, 'speedup': 6.91},
        
        # GAT-Large results (sample)
        {'model_name': 'GAT-Large', 'batch_size': 1, 'eager_time_ms': 4.32, 'graph_time_ms': 0.89, 'speedup': 4.85},
        {'model_name': 'GAT-Large', 'batch_size': 8, 'eager_time_ms': 35.67, 'graph_time_ms': 5.12, 'speedup': 6.97},
        {'model_name': 'GAT-Large', 'batch_size': 32, 'eager_time_ms': 145.23, 'graph_time_ms': 18.76, 'speedup': 7.74},
    ]
    return sample_results

def create_batch_size_comparison_table(df):
    """Create a table with batch sizes as columns and models as rows"""
    # Get unique models and batch sizes
    models = sorted(df['model_name'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create column names for eager and graph modes
    columns = []
    for batch_size in batch_sizes:
        columns.append(f'Eager_BS{batch_size}')
        columns.append(f'Graph_BS{batch_size}')
        columns.append(f'Speedup_BS{batch_size}')
    
    # Initialize the comparison table
    comparison_table = pd.DataFrame(index=models, columns=columns)
    
    # Fill the table with data
    for _, row in df.iterrows():
        model = row['model_name']
        batch_size = row['batch_size']
        
        eager_col = f'Eager_BS{batch_size}'
        graph_col = f'Graph_BS{batch_size}'
        speedup_col = f'Speedup_BS{batch_size}'
        
        comparison_table.loc[model, eager_col] = row['eager_time_ms']
        comparison_table.loc[model, graph_col] = row['graph_time_ms']
        comparison_table.loc[model, speedup_col] = row['speedup']
    
    return comparison_table

def main():
    """Demonstrate the batch size comparison format"""
    print("🧪 Sample Batch Size Comparison Table Format")
    print("="*80)
    
    # Create sample data
    sample_results = create_sample_results()
    df = pd.DataFrame(sample_results)
    
    print(f"📊 Original data shape: {df.shape}")
    print("Original format (first few rows):")
    print(df.head())
    
    print(f"\n" + "="*80)
    print("📊 NEW FORMAT: GAT Inference Time by Batch Size (ms)")
    print("="*120)
    
    # Create the comparison table
    batch_table = create_batch_size_comparison_table(df)
    print(batch_table.round(2))
    
    print(f"\n📝 Table shape: {batch_table.shape}")
    print(f"📝 Models: {list(batch_table.index)}")
    print(f"📝 Batch sizes tested: {sorted(df['batch_size'].unique())}")
    
    # Save to CSV for inspection
    batch_table.to_csv('sample_batch_comparison.csv', index=True)
    print(f"\n💾 Sample saved to: sample_batch_comparison.csv")
    
    print(f"\n📈 Key Benefits:")
    print(f"  ✅ Easy to compare inference times across batch sizes")
    print(f"  ✅ Eager and Graph times side-by-side")  
    print(f"  ✅ Speedup calculations included")
    print(f"  ✅ Models as rows, batch sizes as columns")
    print(f"  ✅ Perfect for analysis and visualization")

if __name__ == "__main__":
    main() 