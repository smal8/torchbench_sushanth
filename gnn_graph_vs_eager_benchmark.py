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
WARMUP_RUNS = 100
TIMING_RUNS = 100
GUARD_RUNS = 10
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
NODE_COUNTS = [100, 500, 1000]  # Number of nodes in graph
FEATURE_DIMS = [64, 128, 256]   # Node feature dimensions

print(f"Running GCN benchmarks on: {DEVICE}")
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

from utils.constants import LayerType


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

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
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

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

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

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
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
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
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
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')


# Wrapper class to adapt GAT for benchmarking
class GATModel(nn.Module):
    """Wrapper around GAT for easier benchmarking integration"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Configure GAT architecture
        num_features_per_layer = [input_dim, hidden_dim, output_dim]
        num_heads_per_layer = [num_heads] * (num_layers - 1)
        
        self.gat = GAT(
            num_of_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            num_features_per_layer=num_features_per_layer,
            add_skip_connection=True,
            bias=True,
            dropout=dropout,
            layer_type=LayerType.IMP3  # Use most efficient implementation
        )
        
        # Global pooling layer
        final_dim = output_dim  # GAT's final layer doesn't concat heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
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
    print("GCN GRAPH MODE vs EAGER MODE BENCHMARK")
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
    
    # Save detailed results
    csv_filename = 'gcn_graph_vs_eager_results_detailed.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n📝 Detailed results saved to {csv_filename}")
    
    # Create batch size comparison table
    batch_size_table = create_batch_size_comparison_table(df)
    comparison_filename = 'gcn_batch_size_comparison.csv'
    batch_size_table.to_csv(comparison_filename, index=True)
    print(f"📝 Batch size comparison saved to {comparison_filename}")
    
    # Display the comparison table
    print(f"\n📊 GCN Inference Time by Batch Size (ms):")
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
    
    # Create traditional summary plots
    create_summary_plots(df)


def create_batch_size_plots(df):
    """Create visualizations focused on batch size comparisons"""
    # Get unique models and batch sizes
    models = sorted(df['model_name'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    # Create figure with subplots for batch size analysis
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
    
    # 4. Eager vs Graph Comparison (Side-by-side for each model)
    x_pos = range(len(batch_sizes))
    width = 0.15
    
    for i, model in enumerate(models):
        model_data = df[df['model_name'] == model].sort_values('batch_size')
        if len(model_data) > 0:
            eager_times = []
            graph_times = []
            for bs in batch_sizes:
                eager_time = model_data[model_data['batch_size'] == bs]['eager_time_ms']
                graph_time = model_data[model_data['batch_size'] == bs]['graph_time_ms']
                eager_times.append(eager_time.iloc[0] if len(eager_time) > 0 else 0)
                graph_times.append(graph_time.iloc[0] if len(graph_time) > 0 else 0)
            
            x_offset = [x + width * (i - len(models)/2) for x in x_pos]
            ax4.bar([x - width/2 for x in x_offset], eager_times, width, 
                   alpha=0.8, label=f'{model} (Eager)', color=colors[i % len(colors)])
            ax4.bar([x + width/2 for x in x_offset], graph_times, width,
                   alpha=0.6, label=f'{model} (Graph)', color=colors[i % len(colors)], hatch='//')
    
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('Inference Time (ms)', fontsize=12)
    ax4.set_title('Eager vs Graph Mode Comparison', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(batch_sizes)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gcn_batch_size_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Batch size analysis saved to gcn_batch_size_analysis.png")


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
    plt.savefig('gcn_summary_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Summary analysis saved to gcn_summary_analysis.png")

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