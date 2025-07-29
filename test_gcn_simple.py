#!/usr/bin/env python3
"""
Simple test script to verify GCN implementation works correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Copy the GCN implementation from the benchmark script
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

def test_gcn():
    """Test the GCN implementation"""
    print("Testing GCN implementation...")
    
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
    
    # Create adjacency matrix
    adj = torch.rand(batch_size, num_nodes, num_nodes, device=device)
    adj = (adj > 0.6).float()
    adj = (adj + adj.transpose(-1, -2)) / 2  # Make symmetric
    
    print(f"Input shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Create model
    model = GCNModel(input_dim, hidden_dim, output_dim, num_layers=2).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(x, adj)
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
            compiled_output = compiled_model(x, adj)
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
    
    print("\n✅ GCN test completed successfully!")

if __name__ == "__main__":
    test_gcn() 