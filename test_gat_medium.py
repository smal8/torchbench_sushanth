#!/usr/bin/env python3
"""
Test GAT-Medium architecture parameters
"""

def test_gat_medium_params():
    """Test GAT-Medium parameter combinations"""
    
    print("Testing GAT-Medium architecture parameters...")
    
    # GAT-Medium configuration
    input_dim, hidden_dim, output_dim, num_layers, num_heads = 128, 256, 64, 3, 8
    
    # Configure architecture - same logic as in GATModel
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
    
    print(f"GAT-Medium config:")
    print(f"  Layers: {num_layers}")
    print(f"  Features: {num_features_per_layer}")
    print(f"  Heads: {num_heads_per_layer}")
    print(f"  Input->Hidden->Output: {input_dim}->{hidden_dim}->{output_dim}")
    
    # Test the assertion
    try:
        print(f"\nAssertion check: {num_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        print(f"Values: {num_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        
        assert num_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        print("✅ GAT-Medium parameters are valid!")
        
    except AssertionError as e:
        print(f"❌ GAT-Medium parameters failed: {e}")
    
    # Test GAT-Large too
    print(f"\n" + "="*50)
    print("Testing GAT-Large architecture parameters...")
    
    # GAT-Large configuration  
    input_dim, hidden_dim, output_dim, num_layers, num_heads = 256, 512, 128, 4, 8
    
    if num_layers == 2:
        num_features_per_layer = [input_dim, hidden_dim, output_dim]
    elif num_layers == 3:
        num_features_per_layer = [input_dim, hidden_dim, hidden_dim, output_dim]
    elif num_layers == 4:
        num_features_per_layer = [input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim]
    else:
        num_features_per_layer = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
    
    num_heads_per_layer = [num_heads] * num_layers
    
    print(f"GAT-Large config:")
    print(f"  Layers: {num_layers}")
    print(f"  Features: {num_features_per_layer}")
    print(f"  Heads: {num_heads_per_layer}")
    print(f"  Input->Hidden->Output: {input_dim}->{hidden_dim}->{output_dim}")
    
    try:
        print(f"\nAssertion check: {num_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        print(f"Values: {num_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        
        assert num_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        print("✅ GAT-Large parameters are valid!")
        
    except AssertionError as e:
        print(f"❌ GAT-Large parameters failed: {e}")

if __name__ == "__main__":
    test_gat_medium_params() 