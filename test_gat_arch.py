#!/usr/bin/env python3
"""
Test GAT architecture parameters
"""

import torch
import torch.nn as nn
from utils.constants import LayerType

def test_gat_params():
    """Test different GAT parameter combinations"""
    
    print("Testing GAT architecture parameters...")
    
    # Test case 1: 2 layers
    try:
        num_of_layers = 2
        num_features_per_layer = [64, 128, 32]  # input -> hidden -> output
        num_heads_per_layer = [4, 4]  # heads for each layer
        
        print(f"Test 1 - Layers: {num_of_layers}, Features: {num_features_per_layer}, Heads: {num_heads_per_layer}")
        print(f"Assertion check: {num_of_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        print(f"Values: {num_of_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        print("✅ Test 1 passed")
        
    except AssertionError as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test case 2: 3 layers  
    try:
        num_of_layers = 3
        num_features_per_layer = [64, 128, 256, 32]  # input -> hidden1 -> hidden2 -> output
        num_heads_per_layer = [4, 8, 4]  # heads for each layer
        
        print(f"\nTest 2 - Layers: {num_of_layers}, Features: {num_features_per_layer}, Heads: {num_heads_per_layer}")
        print(f"Assertion check: {num_of_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        print(f"Values: {num_of_layers} == {len(num_heads_per_layer)} == {len(num_features_per_layer) - 1}")
        
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1
        print("✅ Test 2 passed")
        
    except AssertionError as e:
        print(f"❌ Test 2 failed: {e}")
    
    print("\nArchitecture test completed!")

if __name__ == "__main__":
    test_gat_params() 