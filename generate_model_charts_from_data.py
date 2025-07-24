#!/usr/bin/env python3
"""
Generate focused bar charts from actual benchmark data.
This script can load data from CSV files, pickle files, or pandas DataFrames
and create individual model comparison charts.

Usage:
    python generate_model_charts_from_data.py --data_file benchmark_results.csv
    python generate_model_charts_from_data.py --data_file results.pkl
"""

import argparse
import pandas as pd
import pickle
from pathlib import Path
from focused_model_comparison import generate_all_model_charts

def load_data_from_file(file_path):
    """
    Load benchmark data from various file formats
    
    Args:
        file_path: Path to data file (CSV, pickle, or other pandas-readable format)
        
    Returns:
        pandas DataFrame with benchmark data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    
    # Determine file type and load accordingly
    if file_path.suffix.lower() == '.csv':
        data = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
    elif file_path.suffix.lower() in ['.json']:
        data = pd.read_json(file_path)
    else:
        # Try to read as CSV by default
        try:
            data = pd.read_csv(file_path)
        except:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    print(f"Columns: {list(data.columns)}")
    
    return data

def validate_and_prepare_data(data):
    """
    Validate that the data has required columns and prepare it for charting
    
    Args:
        data: pandas DataFrame with benchmark data
        
    Returns:
        validated and prepared DataFrame
    """
    required_columns = {
        'model': ['model_name', 'model'],
        'eager_time': ['baseline_time', 'eager_time'],
        'graph_time': ['compiled_time', 'graph_time']
    }
    
    # Check for required columns
    model_col = None
    eager_col = None
    graph_col = None
    
    for col_type, possible_names in required_columns.items():
        found_col = None
        for name in possible_names:
            if name in data.columns:
                found_col = name
                break
        
        if found_col is None:
            raise ValueError(f"Missing required column for {col_type}. Expected one of: {possible_names}")
        
        if col_type == 'model':
            model_col = found_col
        elif col_type == 'eager_time':
            eager_col = found_col
        elif col_type == 'graph_time':
            graph_col = found_col
    
    print(f"Using columns: model='{model_col}', eager_time='{eager_col}', graph_time='{graph_col}'")
    
    # Standardize column names
    if model_col != 'model_name':
        data = data.rename(columns={model_col: 'model_name'})
    if eager_col != 'baseline_time':
        data = data.rename(columns={eager_col: 'baseline_time'})
    if graph_col != 'compiled_time':
        data = data.rename(columns={graph_col: 'compiled_time'})
    
    # Add default guard time columns if they don't exist
    if 'guard_check_time' not in data.columns:
        data['guard_check_time'] = 0.001  # Default 1ms
        print("Added default guard_check_time (1ms)")
    
    if 'recompile_time' not in data.columns:
        data['recompile_time'] = 2.0  # Default 2s
        print("Added default recompile_time (2s)")
    
    if 'guard_failure_rate' not in data.columns:
        data['guard_failure_rate'] = 0.1  # Default 10%
        print("Added default guard_failure_rate (10%)")
    
    # Remove any rows with missing essential data
    initial_length = len(data)
    data = data.dropna(subset=['model_name', 'baseline_time', 'compiled_time'])
    final_length = len(data)
    
    if final_length < initial_length:
        print(f"Removed {initial_length - final_length} rows with missing data")
    
    return data

def print_data_summary(data):
    """Print a summary of the loaded data"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total models: {len(data)}")
    print(f"Model names: {list(data['model_name'].unique())}")
    
    if len(data) > 0:
        print(f"\nInference Times (seconds):")
        print(f"  Eager mode - avg: {data['baseline_time'].mean():.4f}s, range: {data['baseline_time'].min():.4f}s - {data['baseline_time'].max():.4f}s")
        print(f"  Graph mode - avg: {data['compiled_time'].mean():.4f}s, range: {data['compiled_time'].min():.4f}s - {data['compiled_time'].max():.4f}s")
        
        speedups = data['baseline_time'] / data['compiled_time']
        print(f"  Speedup - avg: {speedups.mean():.2f}x, range: {speedups.min():.2f}x - {speedups.max():.2f}x")
        
        if 'guard_check_time' in data.columns:
            print(f"\nGuard Times:")
            print(f"  Guard check - avg: {data['guard_check_time'].mean()*1000:.3f}ms")
            print(f"  Recompile - avg: {data['recompile_time'].mean():.2f}s")
            print(f"  Failure rate - avg: {data['guard_failure_rate'].mean()*100:.1f}%")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate focused model comparison charts from benchmark data')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to benchmark data file (CSV, pickle, JSON)')
    parser.add_argument('--output_dir', type=str, default='./model_charts_real',
                       help='Output directory for charts (default: ./model_charts_real)')
    parser.add_argument('--show_summary', action='store_true',
                       help='Show detailed data summary before generating charts')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data_from_file(args.data_file)
        
        # Validate and prepare data
        data = validate_and_prepare_data(data)
        
        # Show summary if requested
        if args.show_summary:
            print_data_summary(data)
        
        # Generate charts
        print(f"\nGenerating charts with real benchmark data...")
        inference_charts, guard_charts = generate_all_model_charts(data, args.output_dir)
        
        print(f"\n🎉 SUCCESS! Generated {len(inference_charts) + len(guard_charts)} charts")
        print(f"📁 Charts saved in: {args.output_dir}")
        print(f"   • {len(inference_charts)} inference comparison charts")  
        print(f"   • {len(guard_charts)} guard analysis charts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 