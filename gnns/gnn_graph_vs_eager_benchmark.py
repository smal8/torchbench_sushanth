import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphConv, ChebConv, APPNP, GatedGraphConv, TransformerConv, TAGConv, SGConv, PNAConv, BatchNorm
from torch_geometric.utils import degree              
from torch_geometric.data import Data, Batch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
    
class GraphConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    

class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K, alpha)

    def forward(self, x, edge_index):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x


class GGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.gated = GatedGraphConv(out_channels, num_layers)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = self.gated(x, edge_index)
        return x

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x



class TAGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class SGCNet(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=False)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)




class PNANet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, deg):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.conv1 = PNAConv(in_channels, hidden_channels, aggregators, scalers, deg)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = PNAConv(hidden_channels, out_channels, aggregators, scalers, deg)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.conv2(x, edge_index)
        return x


def generate_sample_data(batch_size: int, num_nodes: int = 1000, num_features: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate sample graph data for benchmarking."""
    # Create node features
    x = torch.randn(batch_size * num_nodes, num_features, device=device)
    
    # Create random edge indices for each graph in the batch
    edge_indices = []
    for i in range(batch_size):
        # Create edges within each graph
        num_edges = num_nodes * 2  # Average degree ~4
        source_nodes = torch.randint(0, num_nodes, (num_edges,))
        target_nodes = torch.randint(0, num_nodes, (num_edges,))
        
        # Add batch offset
        source_nodes += i * num_nodes
        target_nodes += i * num_nodes
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        edge_indices.append(edge_index)
    
    # Concatenate all edge indices
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    
    # Calculate degree for PNA model
    deg = degree(edge_index[0], batch_size * num_nodes, dtype=torch.long)
    
    return x, edge_index, deg


def measure_eager_time(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, 
                      num_iterations: int = 100) -> float:
    """Measure eager mode inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, edge_index)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x, edge_index)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # Return in milliseconds


def measure_graph_mode_time(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, 
                           warmup_iterations: int = 100, measure_iterations: int = 100) -> float:
    """Measure graph mode (torch.compile) inference time."""
    model.eval()
    
    # Compile the model
    compiled_model = torch.compile(model)
    
    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = compiled_model(x, edge_index)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(measure_iterations):
            _ = compiled_model(x, edge_index)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / measure_iterations
    return avg_time * 1000  # Return in milliseconds


def benchmark_models() -> Dict[str, Dict[str, List[float]]]:
    """Benchmark all models across different batch sizes."""
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    # First, generate sample data to get degree information for PNA
    sample_x, sample_edge_index, sample_deg = generate_sample_data(1)
    
    # Model configurations
    models = {
        'GCN': GCN(16, 64, 7),
        'GraphSAGE': GraphSAGE(16, 64, 7),
        'GAT': GAT(16, 64, 7, heads=8),
        'GIN': GIN(16, 64, 7),
        'GraphConvNet': GraphConvNet(16, 64, 7),
        'ChebNet': ChebNet(16, 64, 7, K=3),
        'APPNPNet': APPNPNet(16, 64, 7, K=10, alpha=0.1),
        'GGNN': GGNN(16, 7, num_layers=3),
        'GraphTransformer': GraphTransformer(16, 64, 7, heads=4),
        'TAGCN': TAGCN(16, 64, 7),
        'SGCNet': SGCNet(16, 7, K=2),
        'PNANet': PNANet(16, 64, 7, sample_deg)
    }
    
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}...")
        model = model.to(device)
        
        eager_times = []
        graph_times = []
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Generate data for this batch size
            x, edge_index, deg = generate_sample_data(batch_size)
            
            # Measure eager mode time
            eager_time = measure_eager_time(model, x, edge_index)
            eager_times.append(eager_time)
            print(f"    Eager mode: {eager_time:.2f} ms")
            
            # Measure graph mode time
            try:
                graph_time = measure_graph_mode_time(model, x, edge_index)
                graph_times.append(graph_time)
                print(f"    Graph mode: {graph_time:.2f} ms")
                print(f"    Speedup: {eager_time/graph_time:.2f}x")
            except Exception as e:
                print(f"    Graph mode failed: {e}")
                graph_times.append(float('nan'))
        
        results[model_name] = {
            'eager': eager_times,
            'graph': graph_times
        }
    
    return results, batch_sizes


def plot_results(results: Dict[str, Dict[str, List[float]]], batch_sizes: List[int]):
    """Create separate plots for each model."""
    
    model_names = list(results.keys())
    num_models = len(model_names)
    
    # Calculate grid size for subplots
    cols = 4
    rows = (num_models + cols - 1) // cols  # Ceiling division
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle('GNN Models: Eager vs Graph Mode Inference Time', fontsize=16)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        eager_times = results[model_name]['eager']
        graph_times = results[model_name]['graph']
        
        # Plot lines
        ax.plot(batch_sizes, eager_times, 'o-', label='Eager Mode', linewidth=2, markersize=6)
        ax.plot(batch_sizes, graph_times, 's-', label='Graph Mode', linewidth=2, markersize=6)
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Inference Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        
        # Set x-axis ticks
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
    
    # Hide empty subplots
    for idx in range(num_models, rows * cols):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gnn_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for each model
    for model_name in model_names:
        plt.figure(figsize=(8, 6))
        
        eager_times = results[model_name]['eager']
        graph_times = results[model_name]['graph']
        
        plt.plot(batch_sizes, eager_times, 'o-', label='Eager Mode', linewidth=2, markersize=8)
        plt.plot(batch_sizes, graph_times, 's-', label='Graph Mode', linewidth=2, markersize=8)
        
        plt.title(f'{model_name}: Eager vs Graph Mode Inference Time', fontsize=14, fontweight='bold')
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        # Set x-axis ticks
        plt.xticks(batch_sizes, batch_sizes)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()


def print_summary_table(results: Dict[str, Dict[str, List[float]]], batch_sizes: List[int]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    for model_name in results.keys():
        print(f"\n{model_name}:")
        print("-" * 40)
        print(f"{'Batch Size':<12} {'Eager (ms)':<12} {'Graph (ms)':<12} {'Speedup':<10}")
        print("-" * 40)
        
        eager_times = results[model_name]['eager']
        graph_times = results[model_name]['graph']
        
        for i, batch_size in enumerate(batch_sizes):
            eager_time = eager_times[i]
            graph_time = graph_times[i]
            speedup = eager_time / graph_time if not np.isnan(graph_time) else float('nan')
            
            print(f"{batch_size:<12} {eager_time:<12.2f} {graph_time:<12.2f} {speedup:<10.2f}")


if __name__ == "__main__":
    print("Starting GNN Benchmark: Eager vs Graph Mode")
    print("=" * 50)
    
    # Run benchmark
    results, batch_sizes = benchmark_models()
    
    # Print summary
    print_summary_table(results, batch_sizes)
    
    # Create plots
    plot_results(results, batch_sizes)
    
    print("\nBenchmark completed! Check the generated plots and summary above.")

