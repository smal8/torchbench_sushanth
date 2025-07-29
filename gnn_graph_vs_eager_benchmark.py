#!/usr/bin/env python3
"""
GCN Graph Mode vs Eager Mode Benchmark
=====================================
Benchmarking of real Graph Convolutional Network (GCN) architectures comparing:
- Eager Mode: Standard PyTorch execution
- Graph Mode: torch.compile() optimized execution

GNN Architecture Tested:
- Graph Convolutional Networks (GCN) - Real implementation with proper graph convolution

Metrics:
- Eager mode inference time
- Graph mode inference time
- Compilation time overhead
- Speedup ratio
- Memory usage comparison
- Guard time analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
import gc
import warnings
import os
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WARMUP_RUNS = 5
TIMING_RUNS = 20
GUARD_RUNS = 10
BATCH_SIZES = [32, 64, 128]
NODE_COUNTS = [100, 500, 1000]  # Number of nodes in graph
FEATURE_DIMS = [64, 128, 256]   # Node feature dimensions

print(f"Running GNN benchmarks on: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

def clear_cache():
    """Clear GPU cache and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def normalize_adj(A):
    """Normalize adjacency matrix with self-loops"""
    # Fill diagonal with one (i.e., add self-edge)
    A_mod = A + torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    
    # Create degree matrix for each graph
    diag = torch.sum(A_mod, dim=-1)
    D_inv_sqrt = torch.diag_embed(torch.pow(diag, -0.5))
    
    # Create the normalized adjacency matrix
    A_hat = torch.matmul(D_inv_sqrt, torch.matmul(A_mod, D_inv_sqrt))
    return A_hat

class GCNLayer(nn.Module):
    """Real GCN Layer implementing proper graph convolution"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, A_hat, x):
        # Aggregate: multiply by normalized adjacency matrix
        x = torch.matmul(A_hat, x)
        
        # Update: apply linear transformation
        x = self.linear(x)
        x = F.relu(x)
        return x

class GCNModel(nn.Module):
    """Real Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Final projection layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
    
    def forward(self, x, adj_matrix=None):
        batch_size, num_nodes, feature_dim = x.shape
        
        # Create adjacency matrix if not provided (random graph structure)
        if adj_matrix is None:
            # Create random adjacency matrices for each graph in batch
            adj_matrix = torch.rand(batch_size, num_nodes, num_nodes, device=x.device)
            adj_matrix = (adj_matrix > 0.7).float()  # Sparsify
            # Make symmetric
            adj_matrix = (adj_matrix + adj_matrix.transpose(-1, -2)) / 2
        
        # Normalize adjacency matrices
        A_hat = normalize_adj(adj_matrix)
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(A_hat, x)
            
            if i < len(self.batch_norms):
                # Reshape for batch norm: (batch_size * num_nodes, hidden_dim)
                x_reshaped = x.view(-1, x.shape[-1])
                x_reshaped = self.batch_norms[i](x_reshaped)
                x = x_reshaped.view(batch_size, num_nodes, -1)
                
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global average pooling across nodes
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        # Final projection
        x = self.final_layer(x)
        return x











def create_graph_data(batch_size, num_nodes, feature_dim):
    """Create synthetic graph data with proper adjacency matrices"""
    # Node features
    x = torch.randn(batch_size, num_nodes, feature_dim, device=DEVICE)
    
    # Create random adjacency matrices for each graph in batch
    adj_matrices = []
    for _ in range(batch_size):
        # Create random sparse adjacency matrix
        adj = torch.rand(num_nodes, num_nodes, device=DEVICE)
        adj = (adj > 0.8).float()  # Make sparse (20% connectivity)
        
        # Make symmetric (undirected graph)
        adj = (adj + adj.t()) / 2
        
        # Ensure at least some connectivity
        if adj.sum() == 0:
            # Add a few random edges if completely disconnected
            indices = torch.randint(0, num_nodes, (2, min(5, num_nodes)), device=DEVICE)
            adj[indices[0], indices[1]] = 1
            adj[indices[1], indices[0]] = 1
        
        adj_matrices.append(adj)
    
    # Stack into batch tensor
    adj_matrix = torch.stack(adj_matrices, dim=0)
    
    return x, adj_matrix

def measure_eager_inference(model, x, adj_matrix, model_name):
    """Measure eager mode inference time"""
    print(f"    Measuring eager mode for {model_name}...")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x, adj_matrix)
    
    clear_cache()
    
    # Timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = model(x, adj_matrix)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            end_memory = get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'memory_usage': np.mean(memory_usage)
    }

def measure_compilation_time(model, x, adj_matrix, model_name):
    """Measure compilation time and return compiled model"""
    print(f"    Measuring compilation time for {model_name}...")
    
    clear_cache()
    
    start_time = time.perf_counter()
    try:
        compiled_model = torch.compile(model, backend='inductor', mode='default')
        
        # First forward pass triggers compilation
        with torch.no_grad():
            _ = compiled_model(x, adj_matrix)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        compilation_time = end_time - start_time
        
        return compiled_model, compilation_time
        
    except Exception as e:
        print(f"    Compilation failed for {model_name}: {e}")
        return None, 0.0

def measure_graph_mode_inference(compiled_model, x, adj_matrix, model_name):
    """Measure graph mode inference time"""
    print(f"    Measuring graph mode for {model_name}...")
    
    if compiled_model is None:
        return None
    
    compiled_model.eval()
    
    # Warmup (compilation already done)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = compiled_model(x, adj_matrix)
    
    clear_cache()
    
    # Timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = compiled_model(x, adj_matrix)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            end_memory = get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'memory_usage': np.mean(memory_usage)
    }

def run_gnn_benchmark():
    """Run comprehensive GNN benchmarking"""
    
    # Define GCN models to test (real implementation based on user's code)
    gnn_models = {
        'GCN-Small': (GCNModel, {'input_dim': 64, 'hidden_dim': 128, 'output_dim': 32, 'num_layers': 2}),
        'GCN-Medium': (GCNModel, {'input_dim': 128, 'hidden_dim': 256, 'output_dim': 64, 'num_layers': 3}),
        'GCN-Large': (GCNModel, {'input_dim': 256, 'hidden_dim': 512, 'output_dim': 128, 'num_layers': 4}),
    }
    
    # Graph configurations to test
    graph_configs = [
        {'batch_size': 32, 'num_nodes': 100, 'feature_dim': 64},
        {'batch_size': 64, 'num_nodes': 500, 'feature_dim': 128},
        {'batch_size': 128, 'num_nodes': 1000, 'feature_dim': 256},
    ]
    
    results = []
    
    print("="*80)
    print("GCN GRAPH MODE vs EAGER MODE BENCHMARK")
    print("="*80)
    
    for config_idx, graph_config in enumerate(graph_configs):
        print(f"\n📊 Testing Configuration {config_idx + 1}/3:")
        print(f"   Batch Size: {graph_config['batch_size']}")
        print(f"   Nodes: {graph_config['num_nodes']}")
        print(f"   Feature Dim: {graph_config['feature_dim']}")
        print("-" * 60)
        
        # Create graph data for this configuration
        x, adj_matrix = create_graph_data(
            graph_config['batch_size'], 
            graph_config['num_nodes'], 
            graph_config['feature_dim']
        )
        
        for model_name, (model_class, model_kwargs) in gnn_models.items():
            # Skip models that don't match the feature dimension
            if model_kwargs['input_dim'] != graph_config['feature_dim']:
                continue
                
            print(f"\n🧠 Testing {model_name}...")
            
            try:
                # Create model
                model = model_class(**model_kwargs).to(DEVICE)
                
                # Measure eager mode
                eager_results = measure_eager_inference(model, x, adj_matrix, model_name)
                
                # Measure compilation time and get compiled model
                compiled_model, compilation_time = measure_compilation_time(model, x, adj_matrix, model_name)
                
                # Measure graph mode
                graph_results = measure_graph_mode_inference(compiled_model, x, adj_matrix, model_name)
                
                if graph_results is not None:
                    speedup = eager_results['mean_time'] / graph_results['mean_time']
                    
                    result = {
                        'model_name': model_name,
                        'batch_size': graph_config['batch_size'],
                        'num_nodes': graph_config['num_nodes'],
                        'feature_dim': graph_config['feature_dim'],
                        'eager_time_ms': eager_results['mean_time'] * 1000,
                        'eager_std_ms': eager_results['std_time'] * 1000,
                        'graph_time_ms': graph_results['mean_time'] * 1000,
                        'graph_std_ms': graph_results['std_time'] * 1000,
                        'speedup': speedup,
                        'compilation_time_s': compilation_time,
                        'eager_memory_mb': eager_results['memory_usage'],
                        'graph_memory_mb': graph_results['memory_usage'],
                        'memory_overhead_mb': graph_results['memory_usage'] - eager_results['memory_usage']
                    }
                    
                    results.append(result)
                    
                    print(f"  ✅ Eager: {eager_results['mean_time']*1000:.2f}±{eager_results['std_time']*1000:.2f}ms")
                    print(f"  🚀 Graph: {graph_results['mean_time']*1000:.2f}±{graph_results['std_time']*1000:.2f}ms")
                    print(f"  ⚡ Speedup: {speedup:.2f}x")
                    print(f"  🔨 Compilation: {compilation_time:.2f}s")
                    
                else:
                    print(f"  ❌ Graph mode failed for {model_name}")
                    
            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {e}")
                
            # Clean up
            del model
            if 'compiled_model' in locals():
                del compiled_model
            clear_cache()
    
    return results

def save_results(results):
    """Save results to CSV and generate summary"""
    if not results:
        print("No results to save!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = 'gnn_graph_vs_eager_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n📝 Results saved to {csv_filename}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal models tested: {len(df)}")
    print(f"Average speedup: {df['speedup'].mean():.2f}x")
    print(f"Median speedup: {df['speedup'].median():.2f}x")
    print(f"Best speedup: {df['speedup'].max():.2f}x ({df.loc[df['speedup'].idxmax(), 'model_name']})")
    print(f"Worst speedup: {df['speedup'].min():.2f}x ({df.loc[df['speedup'].idxmin(), 'model_name']})")
    
    # Model family analysis
    print(f"\n📊 Speedup by Model Family:")
    for family in ['GCN']:
        family_data = df[df['model_name'].str.contains(family)]
        if not family_data.empty:
            print(f"  {family}: {family_data['speedup'].mean():.2f}x (avg)")
    
    # Memory analysis
    print(f"\n💾 Memory Analysis:")
    print(f"Average memory overhead: {df['memory_overhead_mb'].mean():.1f} MB")
    print(f"Max memory overhead: {df['memory_overhead_mb'].max():.1f} MB")
    
    # Compilation time analysis
    print(f"\n🔨 Compilation Time Analysis:")
    print(f"Average compilation time: {df['compilation_time_s'].mean():.2f}s")
    print(f"Max compilation time: {df['compilation_time_s'].max():.2f}s")
    
    # Best performing models
    print(f"\n🏆 Top 5 Models by Speedup:")
    top_models = df.nlargest(5, 'speedup')[['model_name', 'speedup', 'eager_time_ms', 'graph_time_ms']]
    for _, row in top_models.iterrows():
        print(f"  {row['model_name']}: {row['speedup']:.2f}x ({row['eager_time_ms']:.1f}ms → {row['graph_time_ms']:.1f}ms)")
    
    print("\n" + "="*80)

def create_visualization(results):
    """Create visualization of results"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Speedup comparison by model
    model_speedups = df.groupby('model_name')['speedup'].mean().sort_values(ascending=True)
    ax1.barh(range(len(model_speedups)), model_speedups.values)
    ax1.set_yticks(range(len(model_speedups)))
    ax1.set_yticklabels(model_speedups.index, fontsize=8)
    ax1.set_xlabel('Speedup (x)')
    ax1.set_title('Graph Mode Speedup by Model')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax1.legend()
    
    # 2. Eager vs Graph mode times
    model_times = df.groupby('model_name')[['eager_time_ms', 'graph_time_ms']].mean()
    x = np.arange(len(model_times))
    width = 0.35
    ax2.bar(x - width/2, model_times['eager_time_ms'], width, label='Eager Mode', alpha=0.8)
    ax2.bar(x + width/2, model_times['graph_time_ms'], width, label='Graph Mode', alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Execution Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_times.index, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    
    # 3. Compilation time vs speedup
    ax3.scatter(df['compilation_time_s'], df['speedup'], alpha=0.6)
    ax3.set_xlabel('Compilation Time (s)')
    ax3.set_ylabel('Speedup (x)')
    ax3.set_title('Compilation Time vs Speedup')
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 4. Memory overhead
    memory_overhead = df.groupby('model_name')['memory_overhead_mb'].mean().sort_values()
    ax4.bar(range(len(memory_overhead)), memory_overhead.values)
    ax4.set_xticks(range(len(memory_overhead)))
    ax4.set_xticklabels(memory_overhead.index, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Memory Overhead (MB)')
    ax4.set_title('Memory Overhead by Model')
    
    plt.tight_layout()
    plt.savefig('gnn_graph_vs_eager_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"📈 Visualization saved to gnn_graph_vs_eager_benchmark.png")

def main():
    """Main benchmarking function"""
    print("Starting GCN Graph Mode vs Eager Mode Benchmark...")
    print(f"Device: {DEVICE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Timing runs: {TIMING_RUNS}")
    
    # Run benchmark
    results = run_gnn_benchmark()
    
    # Save and analyze results
    save_results(results)
    create_visualization(results)
    
    print("\n🎉 Benchmark completed successfully!")

if __name__ == "__main__":
    main() 