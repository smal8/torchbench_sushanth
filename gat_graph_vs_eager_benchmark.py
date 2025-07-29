#!/usr/bin/env python3
"""
GAT Graph Mode vs Eager Mode Benchmark
=====================================
Benchmarking of real Graph Attention Network (GAT) architectures comparing:
- Eager Mode: Standard PyTorch execution
- Graph Mode: torch.compile() optimized execution

GNN Architecture Tested:
- Graph Attention Networks (GAT) - Real implementation with proper attention mechanisms

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
import pandas as pd
import gc
import warnings
import os
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.constants import LayerType

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WARMUP_RUNS = 100
TIMING_RUNS = 100
GUARD_RUNS = 10
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
NODE_COUNTS = [100, 500, 1000]  # Number of nodes in graph
FEATURE_DIMS = [64, 128, 256]   # Node feature dimensions

print(f"Running GAT benchmarks on: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"Warmup iterations: {WARMUP_RUNS}")
print(f"Timing iterations: {TIMING_RUNS}")
print(f"Testing batch sizes: {BATCH_SIZES}")
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


class GAT(torch.nn.Module):
    """
    Real GAT implementation with 3 different implementations for efficiency comparison.
    Using the most efficient IMP3 implementation by default.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            # You can treat this one matrix as num_of_heads independent W matrices
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """Xavier uniform initialization as used in the original GAT implementation."""
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric but much more readable!
    It's suitable for both transductive and inductive settings.
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index
    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # Step 1: Linear Projection + regularization
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # Step 2: Edge attention calculation
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Step 3: Neighborhood aggregation
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        # Step 4: Residual/skip connections, concat and bias
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """Neighborhood-aware softmax for attention calculation"""
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """Lifts vectors depending on the edge index."""
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'
    
    if layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported in this benchmark.')


# Wrapper class to adapt GAT for benchmarking
class GATModel(nn.Module):
    """Wrapper around GAT for easier benchmarking integration"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Configure GAT architecture - handle multiple layers properly
        if num_layers == 2:
            num_features_per_layer = [input_dim, hidden_dim, output_dim]
        elif num_layers == 3:
            num_features_per_layer = [input_dim, hidden_dim, hidden_dim, output_dim]
        elif num_layers == 4:
            num_features_per_layer = [input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim]
        else:
            # General case: use hidden_dim for all intermediate layers
            num_features_per_layer = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        num_heads_per_layer = [num_heads] * num_layers
        
        self.gat = GAT(
            num_of_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            num_features_per_layer=num_features_per_layer,
            add_skip_connection=True,
            bias=True,
            dropout=dropout,
            layer_type=LayerType.IMP3  # Use most efficient implementation
        )
        
    def forward(self, x, edge_index):
        batch_size, num_nodes, feature_dim = x.shape
        
        # Process each graph in the batch
        outputs = []
        for i in range(batch_size):
            # Extract single graph
            graph_features = x[i]  # (num_nodes, feature_dim)
            graph_edge_index = edge_index[i]  # (2, num_edges)
            
            # GAT expects (features, edge_index) tuple
            data = (graph_features, graph_edge_index)
            graph_output, _ = self.gat(data)  # (num_nodes, output_dim)
            
            # Global average pooling
            pooled = torch.mean(graph_output, dim=0, keepdim=True)  # (1, output_dim)
            outputs.append(pooled)
        
        # Stack batch results
        return torch.stack(outputs, dim=0).squeeze(1)  # (batch_size, output_dim)


def create_graph_data(batch_size, num_nodes, feature_dim):
    """Create synthetic graph data with proper edge indices for GAT"""
    # Node features
    x = torch.randn(batch_size, num_nodes, feature_dim, device=DEVICE)
    
    # Create edge indices for each graph in batch
    edge_indices = []
    for _ in range(batch_size):
        # Create random edges (ensuring connectivity)
        num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1) // 2)  # At most complete graph
        
        # Generate random edges
        sources = torch.randint(0, num_nodes, (num_edges,), device=DEVICE)
        targets = torch.randint(0, num_nodes, (num_edges,), device=DEVICE)
        
        # Add self-loops to ensure each node is connected
        self_loops_src = torch.arange(num_nodes, device=DEVICE)
        self_loops_tgt = torch.arange(num_nodes, device=DEVICE)
        
        # Combine edges and self-loops
        all_sources = torch.cat([sources, self_loops_src])
        all_targets = torch.cat([targets, self_loops_tgt])
        
        # Create edge index (2, E)
        edge_index = torch.stack([all_sources, all_targets], dim=0)
        edge_indices.append(edge_index)
    
    # Stack into batch tensor
    edge_index_batch = torch.stack(edge_indices, dim=0)
    
    return x, edge_index_batch


def measure_eager_inference(model, x, edge_index, model_name):
    """Measure eager mode inference time"""
    print(f"    Measuring eager mode for {model_name}...")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x, edge_index)
    
    clear_cache()
    
    # Timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = model(x, edge_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            end_memory = get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
    
    return {
        'mean_time': sum(times) / len(times),
        'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min_time': min(times),
        'max_time': max(times),
        'memory_usage': sum(memory_usage) / len(memory_usage)
    }


def measure_compilation_time(model, x, edge_index, model_name):
    """Measure compilation time and return compiled model"""
    print(f"    Measuring compilation time for {model_name}...")
    
    clear_cache()
    
    start_time = time.perf_counter()
    try:
        compiled_model = torch.compile(model, backend='inductor', mode='default')
        
        # First forward pass triggers compilation
        with torch.no_grad():
            _ = compiled_model(x, edge_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        compilation_time = end_time - start_time
        
        return compiled_model, compilation_time
        
    except Exception as e:
        print(f"    Compilation failed for {model_name}: {e}")
        return None, 0.0


def measure_graph_mode_inference(compiled_model, x, edge_index, model_name):
    """Measure graph mode inference time"""
    print(f"    Measuring graph mode for {model_name}...")
    
    if compiled_model is None:
        return None
    
    compiled_model.eval()
    
    # Warmup (compilation already done)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = compiled_model(x, edge_index)
    
    clear_cache()
    
    # Timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = compiled_model(x, edge_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            end_memory = get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
    
    return {
        'mean_time': sum(times) / len(times),
        'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min_time': min(times),
        'max_time': max(times),
        'memory_usage': sum(memory_usage) / len(memory_usage)
    }


def run_gat_benchmark():
    """Run comprehensive GAT benchmarking"""
    
    # Define GAT models to test (real implementation based on user's code)
    gat_models = {
        'GAT-Small': (GATModel, {'input_dim': 64, 'hidden_dim': 128, 'output_dim': 32, 'num_layers': 2, 'num_heads': 4}),
        'GAT-Medium': (GATModel, {'input_dim': 128, 'hidden_dim': 256, 'output_dim': 64, 'num_layers': 3, 'num_heads': 8}),
        'GAT-Large': (GATModel, {'input_dim': 256, 'hidden_dim': 512, 'output_dim': 128, 'num_layers': 4, 'num_heads': 8}),
    }
    
    # Graph configurations to test - different batch sizes with fixed graph properties
    graph_configs = []
    for batch_size in BATCH_SIZES:
        # Test with medium-sized graphs to focus on batch size impact
        graph_configs.append({
            'batch_size': batch_size, 
            'num_nodes': 100, 
            'feature_dim': 64
        })
        
    # Also test a few larger configurations with select batch sizes
    for batch_size in [1, 8, 32]:
        graph_configs.append({
            'batch_size': batch_size, 
            'num_nodes': 500, 
            'feature_dim': 128
        })
    
    results = []
    
    print("="*80)
    print("GAT GRAPH MODE vs EAGER MODE BENCHMARK")
    print("="*80)
    print(f"Total configurations to test: {len(graph_configs)}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup iterations: {WARMUP_RUNS}")
    print(f"Timing iterations: {TIMING_RUNS}")
    print("="*80)
    
    for config_idx, graph_config in enumerate(graph_configs):
        print(f"\n📊 Testing Configuration {config_idx + 1}/{len(graph_configs)}:")
        print(f"   Batch Size: {graph_config['batch_size']}")
        print(f"   Nodes: {graph_config['num_nodes']}")
        print(f"   Feature Dim: {graph_config['feature_dim']}")
        print("-" * 60)
        
        # Create graph data for this configuration
        x, edge_index = create_graph_data(
            graph_config['batch_size'], 
            graph_config['num_nodes'], 
            graph_config['feature_dim']
        )
        
        for model_name, (model_class, model_kwargs) in gat_models.items():
            # Skip models that don't match the feature dimension
            if model_kwargs['input_dim'] != graph_config['feature_dim']:
                continue
                
            print(f"\n🧠 Testing {model_name}...")
            
            model = None
            compiled_model = None
            try:
                # Create model
                model = model_class(**model_kwargs).to(DEVICE)
                
                # Measure eager mode
                eager_results = measure_eager_inference(model, x, edge_index, model_name)
                
                # Measure compilation time and get compiled model
                compiled_model, compilation_time = measure_compilation_time(model, x, edge_index, model_name)
                
                # Measure graph mode
                graph_results = measure_graph_mode_inference(compiled_model, x, edge_index, model_name)
                
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
            if model is not None:
                del model
            if compiled_model is not None:
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
    
    # Save detailed results
    csv_filename = 'gat_graph_vs_eager_results_detailed.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n📝 Detailed results saved to {csv_filename}")
    
    # Create batch size comparison table
    batch_size_table = create_batch_size_comparison_table(df)
    comparison_filename = 'gat_batch_size_comparison.csv'
    batch_size_table.to_csv(comparison_filename, index=True)
    print(f"📝 Batch size comparison saved to {comparison_filename}")
    
    # Display the comparison table
    print(f"\n📊 GAT Inference Time by Batch Size (ms):")
    print("="*120)
    print(batch_size_table.round(2))
    
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
    for family in ['GAT']:
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


def create_visualization(results):
    """Create comprehensive visualizations of results"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Create batch size comparison visualizations
    create_batch_size_plots(df)
    
    # Create the main eager vs graph comparison chart
    create_main_comparison_chart(df)
    
    # Create traditional summary plots
    create_summary_plots(df)


def create_batch_size_plots(df):
    """Create visualizations focused on batch size comparisons"""
    # Get unique models and batch sizes
    models = sorted(df['model_name'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create figure with subplots for batch size analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. Combined Inference Time vs Batch Size - Both Eager and Graph Mode
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, model in enumerate(models):
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        color = colors[i % len(colors)]
        
        # Plot eager mode with solid line and circles
        ax1.plot(model_data['batch_size'], model_data['eager_time_ms'], 
                marker='o', linewidth=3, markersize=8, label=f'{model} (Eager)', 
                color=color, linestyle='-', alpha=0.8)
        
        # Plot graph mode with dashed line and squares
        ax1.plot(model_data['batch_size'], model_data['graph_time_ms'],
                marker='s', linewidth=3, markersize=8, label=f'{model} (Graph)', 
                color=color, linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('Batch Size', fontsize=14)
    ax1.set_ylabel('Inference Time (ms)', fontsize=14)
    ax1.set_title('Eager vs Graph Mode: Inference Time vs Batch Size', fontsize=16, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)
    
    # Add a text box explaining the line styles
    ax1.text(0.02, 0.98, 'Line styles:\nSolid = Eager Mode\nDashed = Graph Mode', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgray', alpha=0.8))
    
    # 2. Speedup vs Batch Size
    for i, model in enumerate(models):
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        ax2.plot(model_data['batch_size'], model_data['speedup'],
                marker='D', linewidth=3, markersize=8, label=model, 
                color=colors[i % len(colors)])
    
    ax2.set_xlabel('Batch Size', fontsize=14)
    ax2.set_ylabel('Speedup (x)', fontsize=14)
    ax2.set_title('Speedup vs Batch Size', fontsize=16, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No speedup')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    
    # 3. Improved Side-by-Side Comparison: Eager vs Graph Mode
    # Create a clean comparison chart with models on x-axis and paired bars
    model_data_avg = df.groupby('model_name').agg({
        'eager_time_ms': 'mean',
        'graph_time_ms': 'mean',
        'speedup': 'mean'
    }).reset_index()
    
    x_pos = range(len(model_data_avg))
    width = 0.35
    
    # Create side-by-side bars for easier comparison
    eager_bars = ax3.bar([x - width/2 for x in x_pos], model_data_avg['eager_time_ms'], 
                        width, label='Eager Mode', alpha=0.8, color='lightcoral', 
                        edgecolor='black', linewidth=0.5)
    graph_bars = ax3.bar([x + width/2 for x in x_pos], model_data_avg['graph_time_ms'], 
                        width, label='Graph Mode', alpha=0.8, color='lightblue', 
                        edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for i, (eager_time, graph_time, speedup) in enumerate(zip(
        model_data_avg['eager_time_ms'], 
        model_data_avg['graph_time_ms'], 
        model_data_avg['speedup']
    )):
        # Eager time label
        ax3.text(i - width/2, eager_time + max(model_data_avg['eager_time_ms']) * 0.02, 
                f'{eager_time:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Graph time label
        ax3.text(i + width/2, graph_time + max(model_data_avg['graph_time_ms']) * 0.02, 
                f'{graph_time:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Speedup label between bars
        max_height = max(eager_time, graph_time)
        ax3.text(i, max_height + max(model_data_avg['eager_time_ms']) * 0.08, 
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='darkgreen' if speedup > 1 else 'darkred')
    
    ax3.set_xlabel('Models', fontsize=14)
    ax3.set_ylabel('Average Inference Time (ms)', fontsize=14)
    ax3.set_title('Average Performance Comparison', fontsize=16, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_data_avg['model_name'], rotation=45, ha='right', fontsize=11)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add a note about the speedup
    ax3.text(0.02, 0.98, 'Green numbers: speedup > 1x (graph faster)\nRed numbers: speedup < 1x (eager faster)', 
             transform=ax3.transAxes, fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('gat_batch_size_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Batch size analysis saved to gat_batch_size_analysis.png")


def create_main_comparison_chart(df):
    """Create a standalone chart focused on direct eager vs graph mode comparison"""
    # Calculate average performance across all batch sizes for each model
    model_data_avg = df.groupby('model_name').agg({
        'eager_time_ms': 'mean',
        'graph_time_ms': 'mean',
        'speedup': 'mean'
    }).reset_index().sort_values('speedup', ascending=False)
    
    # Create figure for the main comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = range(len(model_data_avg))
    width = 0.35
    
    # Create side-by-side bars for direct comparison
    eager_bars = ax.bar([x - width/2 for x in x_pos], model_data_avg['eager_time_ms'], 
                       width, label='Eager Mode', alpha=0.8, color='lightcoral', 
                       edgecolor='black', linewidth=0.7)
    graph_bars = ax.bar([x + width/2 for x in x_pos], model_data_avg['graph_time_ms'], 
                       width, label='Graph Mode', alpha=0.8, color='lightblue', 
                       edgecolor='black', linewidth=0.7)
    
    # Add value labels on top of bars
    for i, (eager_time, graph_time, speedup) in enumerate(zip(
        model_data_avg['eager_time_ms'], 
        model_data_avg['graph_time_ms'], 
        model_data_avg['speedup']
    )):
        # Eager time label
        ax.text(i - width/2, eager_time + max(model_data_avg['eager_time_ms']) * 0.01, 
               f'{eager_time:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Graph time label
        ax.text(i + width/2, graph_time + max(model_data_avg['graph_time_ms']) * 0.01, 
               f'{graph_time:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Speedup label above both bars
        max_height = max(eager_time, graph_time)
        ax.text(i, max_height + max(model_data_avg['eager_time_ms']) * 0.05, 
               f'{speedup:.2f}x', ha='center', va='bottom', fontsize=12, 
               fontweight='bold', color='darkgreen' if speedup > 1 else 'darkred')
    
    # Styling
    ax.set_xlabel('GAT Model Configurations', fontsize=14)
    ax.set_ylabel('Average Inference Time (ms)', fontsize=14)
    ax.set_title('GAT Models: Eager vs Graph Mode Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_data_avg['model_name'], rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup explanation
    ax.text(0.02, 0.98, 'Speedup values above bars:\nGreen = Graph mode faster\nRed = Eager mode faster', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    # Add performance summary
    avg_speedup = model_data_avg['speedup'].mean()
    best_model = model_data_avg.loc[model_data_avg['speedup'].idxmax(), 'model_name']
    worst_model = model_data_avg.loc[model_data_avg['speedup'].idxmin(), 'model_name']
    
    summary_text = f'Average Speedup: {avg_speedup:.2f}x\nBest: {best_model}\nWorst: {worst_model}'
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightsteelblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('gat_graph_vs_eager_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"📈 Main comparison chart saved to gat_graph_vs_eager_benchmark.png")
    
    return fig


def create_summary_plots(df):
    """Create traditional summary visualizations"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average Speedup comparison by model
    model_speedups = df.groupby('model_name')['speedup'].mean().sort_values(ascending=True)
    bars = ax1.barh(range(len(model_speedups)), model_speedups.values, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(model_speedups)))
    ax1.set_yticklabels(model_speedups.index, fontsize=10)
    ax1.set_xlabel('Average Speedup (x)', fontsize=12)
    ax1.set_title('Average Graph Mode Speedup by Model', fontsize=14, fontweight='bold')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}x', ha='left', va='center', fontsize=9)
    
    # 2. Compilation Time vs Average Speedup
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
    
    # Add model labels
    for i, row in model_stats.iterrows():
        ax2.annotate(row['model_name'], (row['compilation_time_s'], row['speedup']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. Throughput Analysis (Batch Size / Time)
    df['eager_throughput'] = df['batch_size'] / (df['eager_time_ms'] / 1000)  # samples/sec
    df['graph_throughput'] = df['batch_size'] / (df['graph_time_ms'] / 1000)  # samples/sec
    
    throughput_data = df.groupby('model_name')[['eager_throughput', 'graph_throughput']].mean()
    x = range(len(throughput_data))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], throughput_data['eager_throughput'], width, 
           label='Eager Mode', alpha=0.8, color='lightcoral')
    ax3.bar([i + width/2 for i in x], throughput_data['graph_throughput'], width,
           label='Graph Mode', alpha=0.8, color='lightblue')
    ax3.set_xlabel('Models', fontsize=12)
    ax3.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax3.set_title('Average Throughput Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(throughput_data.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Memory Efficiency
    memory_data = df.groupby('model_name')[['eager_memory_mb', 'graph_memory_mb', 'memory_overhead_mb']].mean()
    
    x = range(len(memory_data))
    ax4.bar(x, memory_data['eager_memory_mb'], label='Eager Mode', alpha=0.8, color='lightgreen')
    ax4.bar(x, memory_data['memory_overhead_mb'], bottom=memory_data['eager_memory_mb'],
           label='Graph Mode Overhead', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Models', fontsize=12)
    ax4.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax4.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(memory_data.index, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gat_summary_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Summary analysis saved to gat_summary_analysis.png")


def main():
    """Main benchmarking function"""
    print("Starting GAT Graph Mode vs Eager Mode Benchmark...")
    print(f"Device: {DEVICE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Timing runs: {TIMING_RUNS}")
    
    # Run benchmark
    results = run_gat_benchmark()
    
    # Save and analyze results
    save_results(results)
    create_visualization(results)
    
    print("\n🎉 Benchmark completed successfully!")


if __name__ == "__main__":
    main() 