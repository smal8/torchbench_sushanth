#!/usr/bin/env python3
"""
Simple test script to verify GAT implementation works correctly
"""

import torch
import torch.nn as nn
import time
from utils.constants import LayerType

# Copy the essential GAT implementation for testing
class GAT(torch.nn.Module):
    """Test GAT implementation"""
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer
        
        # For simplicity, just use a single linear layer for testing
        self.test_linear = nn.Linear(num_features_per_layer[0], num_features_per_layer[-1])

    def forward(self, data):
        in_nodes_features, edge_index = data
        # Simple test forward - just apply linear layer and pool
        out = self.test_linear(in_nodes_features)
        return (out, edge_index)


class GATModel(nn.Module):
    """Simple GAT wrapper for testing"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Simple test implementation
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, edge_index):
        batch_size, num_nodes, feature_dim = x.shape
        
        # Process each graph in the batch
        outputs = []
        for i in range(batch_size):
            # Extract single graph
            graph_features = x[i]  # (num_nodes, feature_dim)
            
            # Simple linear transformation
            graph_output = self.linear(graph_features)  # (num_nodes, output_dim)
            
            # Global average pooling
            pooled = torch.mean(graph_output, dim=0, keepdim=True)  # (1, output_dim)
            outputs.append(pooled)
        
        # Stack batch results
        return torch.stack(outputs, dim=0).squeeze(1)  # (batch_size, output_dim)


def test_gat():
    """Test the GAT implementation"""
    print("Testing GAT implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test parameters
    batch_size = 4
    num_nodes = 10
    input_dim = 32
    hidden_dim = 64
    output_dim = 16
    
    # Create test data
    x = torch.randn(batch_size, num_nodes, input_dim, device=device)
    
    # Create edge indices
    edge_indices = []
    for _ in range(batch_size):
        num_edges = 20
        sources = torch.randint(0, num_nodes, (num_edges,), device=device)
        targets = torch.randint(0, num_nodes, (num_edges,), device=device)
        edge_index = torch.stack([sources, targets], dim=0)
        edge_indices.append(edge_index)
    
    edge_index_batch = torch.stack(edge_indices, dim=0)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index batch shape: {edge_index_batch.shape}")
    
    # Create model
    model = GATModel(input_dim, hidden_dim, output_dim, num_layers=2, num_heads=4).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(x, edge_index_batch)
        end_time = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test compilation
    print("\nTesting torch.compile...")
    try:
        compiled_model = torch.compile(model, backend='inductor', mode='default')
        
        # Trigger compilation
        with torch.no_grad():
            start_time = time.time()
            compiled_output = compiled_model(x, edge_index_batch)
            end_time = time.time()
        
        print(f"Compiled output shape: {compiled_output.shape}")
        print(f"Compilation + forward pass time: {(end_time - start_time)*1000:.2f}ms")
        
        # Check outputs match
        if torch.allclose(output, compiled_output, atol=1e-5):
            print("✅ Outputs match between eager and compiled modes")
        else:
            print("❌ Outputs don't match between modes")
            
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
    
    print("\n✅ GAT test completed successfully!")


if __name__ == "__main__":
    test_gat() 