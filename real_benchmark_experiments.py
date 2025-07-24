#!/usr/bin/env python3
"""
Real PyTorch Model Benchmarking Experiments - TorchBench Style
==============================================================
This script benchmarks a diverse set of models similar to TorchBench:
- Vision models (CNNs, ViTs, etc.)
- Sequence models 
- Different architectures and sizes
- Real experimental measurements of inference and guard times

No hardcoded numbers - all data comes from real experiments!
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import numpy as np
import pandas as pd
import gc
import psutil
import os
from pathlib import Path

# Experimental configuration
WARMUP_RUNS = 8
TIMING_RUNS = 25  # Balance between accuracy and speed
GUARD_RUNS = 12
BATCH_SIZES = [16, 32, 64]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Running TorchBench-style experiments on: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def clear_cache():
    """Clear GPU cache and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def create_simple_lstm(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=2, seq_len=128):
    """Create a simple LSTM model for sequence modeling"""
    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.classifier(lstm_out[:, -1, :])  # Use last output
    
    return SimpleLSTM(vocab_size, embed_dim, hidden_dim, num_layers)

def create_simple_transformer(vocab_size=10000, d_model=256, nhead=8, num_layers=6):
    """Create a simple transformer model"""
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, nhead, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.classifier = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            seq_len = x.size(1)
            embedded = self.embedding(x) + self.pos_encoding[:seq_len]
            transformer_out = self.transformer(embedded)
            return self.classifier(transformer_out.mean(dim=1))  # Global average pooling
    
    return SimpleTransformer(vocab_size, d_model, nhead, num_layers)

def create_bidirectional_lstm(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=2):
    """Create a bidirectional LSTM model"""
    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
            
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.classifier(lstm_out[:, -1, :])  # Use last output
    
    return BiLSTM(vocab_size, embed_dim, hidden_dim, num_layers)

def create_simple_gru(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=2):
    """Create a simple GRU model"""
    class SimpleGRU(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x):
            embedded = self.embedding(x)
            gru_out, _ = self.gru(embedded)
            return self.classifier(gru_out[:, -1, :])  # Use last output
    
    return SimpleGRU(vocab_size, embed_dim, hidden_dim, num_layers)

def create_gcn_model(input_dim=128, hidden_dim=256, output_dim=64, num_layers=3):
    """Create a Graph Convolutional Network model"""
    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.activation = nn.ReLU()
            
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = self.activation(x)
            return x
    
    return GCN(input_dim, hidden_dim, output_dim, num_layers)

def create_graphsage_model(input_dim=128, hidden_dim=256, output_dim=64, num_layers=3):
    """Create a GraphSAGE model"""
    class GraphSAGE(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim * 2, hidden_dim))  # Concatenate node and neighbor features
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim * 2, output_dim))
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # Simple aggregation for demonstration
            neighbor_features = torch.roll(x, 1, dim=1)  # Simple neighbor simulation
            x = torch.cat([x, neighbor_features], dim=-1)
            
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = self.activation(x)
                    # Simple aggregation step
                    neighbor_features = torch.roll(x, 1, dim=1)
                    x = torch.cat([x, neighbor_features], dim=-1)
            return x
    
    return GraphSAGE(input_dim, hidden_dim, output_dim, num_layers)

def create_gat_model(input_dim=128, hidden_dim=256, output_dim=64, num_heads=4, num_layers=3):
    """Create a Graph Attention Network model"""
    class GAT(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
            super().__init__()
            self.num_heads = num_heads
            self.layers = nn.ModuleList()
            
            # First layer
            self.layers.append(nn.Linear(input_dim, hidden_dim * num_heads))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads))
            
            # Output layer
            self.layers.append(nn.Linear(hidden_dim * num_heads, output_dim))
            
            self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    # Apply attention mechanism
                    x_att, _ = self.attention(x, x, x)
                    x = self.activation(x + x_att)  # Residual connection
            return x
    
    return GAT(input_dim, hidden_dim, output_dim, num_heads, num_layers)

def create_wavenet_model(input_dim=80, hidden_dim=256, num_layers=8):
    """Create a WaveNet-style model"""
    class WaveNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            
            self.dilated_convs = nn.ModuleList()
            dilation = 1
            for _ in range(num_layers):
                self.dilated_convs.append(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation, padding=dilation)
                )
                dilation *= 2
            
            self.output_conv = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch, features, time)
            x = x.transpose(1, 2)  # Convert to (batch, features, time)
            x = self.activation(self.input_conv(x))
            
            for conv in self.dilated_convs:
                residual = x
                x = self.activation(conv(x))
                x = x + residual  # Residual connection
            
            x = self.output_conv(x)
            return x.transpose(1, 2)  # Convert back to (batch, time, features)
    
    return WaveNet(input_dim, hidden_dim, num_layers)

def create_deepspeech_model(input_dim=80, hidden_dim=512, num_layers=5):
    """Create a DeepSpeech-style model"""
    class DeepSpeech(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, hidden_dim//2, kernel_size=11, stride=2, padding=5)
            self.conv2 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=11, stride=1, padding=5)
            
            self.rnn_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.rnn_layers.append(nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True))
            
            self.output_layer = nn.Linear(hidden_dim * 2, 29)  # 29 characters for English
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch, time, features)
            x = x.transpose(1, 2)  # Convert to (batch, features, time)
            
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            
            x = x.transpose(1, 2)  # Convert back to (batch, time, features)
            
            for rnn in self.rnn_layers:
                x, _ = rnn(x)
            
            return self.output_layer(x)
    
    return DeepSpeech(input_dim, hidden_dim, num_layers)

def create_tacotron2_model(vocab_size=100, embedding_dim=256, hidden_dim=512):
    """Create a Tacotron2-style model"""
    class Tacotron2(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(hidden_dim * 2, 8, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
            self.mel_linear = nn.Linear(hidden_dim, 80)  # 80 mel-scale features
            
        def forward(self, x):
            # Text encoding
            embedded = self.embedding(x)
            encoded, _ = self.encoder(embedded)
            
            # Attention and decoding (simplified)
            attended, _ = self.attention(encoded, encoded, encoded)
            decoded, _ = self.decoder(attended)
            
            # Mel spectrogram prediction
            mel_output = self.mel_linear(decoded)
            return mel_output
    
    return Tacotron2(vocab_size, embedding_dim, hidden_dim)

def create_clip_model(text_dim=256, vision_dim=256, embed_dim=256):
    """Create a CLIP-style multimodal model"""
    class CLIP(nn.Module):
        def __init__(self, text_dim, vision_dim, embed_dim):
            super().__init__()
            self.text_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(text_dim, 8, batch_first=True), 6
            )
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, vision_dim//4, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(vision_dim//4 * 49, vision_dim)
            )
            
            self.text_projection = nn.Linear(text_dim, embed_dim)
            self.vision_projection = nn.Linear(vision_dim, embed_dim)
            
        def forward(self, text_input):
            # Simplified forward pass for benchmarking
            # In real CLIP, this would handle both text and image inputs
            text_features = self.text_encoder(text_input.float())
            text_features = text_features.mean(dim=1)  # Global average pooling
            text_embedding = self.text_projection(text_features)
            return text_embedding
    
    return CLIP(text_dim, vision_dim, embed_dim)

def create_visual_bert_model(vocab_size=5000, text_dim=256, vision_dim=256):
    """Create a VisualBERT-style model"""
    class VisualBERT(nn.Module):
        def __init__(self, vocab_size, text_dim, vision_dim):
            super().__init__()
            self.text_embedding = nn.Embedding(vocab_size, text_dim)
            self.vision_projection = nn.Linear(vision_dim, text_dim)
            
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(text_dim, 8, batch_first=True), 6
            )
            
            self.classifier = nn.Linear(text_dim, vocab_size)
            
        def forward(self, text_input):
            # Simplified forward pass for benchmarking
            text_embedded = self.text_embedding(text_input)
            
            # In real VisualBERT, vision features would be concatenated here
            multimodal_features = self.transformer(text_embedded)
            
            return self.classifier(multimodal_features.mean(dim=1))
    
    return VisualBERT(vocab_size, text_dim, vision_dim)

def create_vae_model(input_dim=784, hidden_dim=256, latent_dim=64):
    """Create a Variational Autoencoder model"""
    class VAE(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.mu_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            # Encode
            h = self.encoder(x)
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Decode
            reconstructed = self.decoder(z)
            return reconstructed
    
    return VAE(input_dim, hidden_dim, latent_dim)

def create_gan_generator(noise_dim=100, hidden_dim=256, output_dim=784):
    """Create a GAN Generator model"""
    class Generator(nn.Module):
        def __init__(self, noise_dim, hidden_dim, output_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(noise_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.Tanh()
            )
            
        def forward(self, x):
            return self.network(x)
    
    return Generator(noise_dim, hidden_dim, output_dim)

def create_gan_discriminator(input_dim=784, hidden_dim=256):
    """Create a GAN Discriminator model"""
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.network(x)
    
    return Discriminator(input_dim, hidden_dim)

def create_unet_model(in_channels=3, out_channels=3, hidden_dim=64):
    """Create a U-Net model"""
    class UNet(nn.Module):
        def __init__(self, in_channels, out_channels, hidden_dim):
            super().__init__()
            # Encoder
            self.enc1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
            self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
            self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
            
            # Decoder
            self.dec3 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1)
            self.dec2 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
            self.dec1 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
            
            self.pool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # Encoder
            e1 = self.activation(self.enc1(x))
            e2 = self.activation(self.enc2(self.pool(e1)))
            e3 = self.activation(self.enc3(self.pool(e2)))
            
            # Decoder
            d3 = self.activation(self.dec3(self.upsample(e3)))
            d2 = self.activation(self.dec2(self.upsample(d3)))
            d1 = self.dec1(self.upsample(d2))
            
            return d1
    
    return UNet(in_channels, out_channels, hidden_dim)

def create_dqn_model(state_dim=128, hidden_dim=256, action_dim=4):
    """Create a Deep Q-Network model"""
    class DQN(nn.Module):
        def __init__(self, state_dim, hidden_dim, action_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            
        def forward(self, x):
            return self.network(x)
    
    return DQN(state_dim, hidden_dim, action_dim)

def create_ppo_actor_model(state_dim=128, hidden_dim=256, action_dim=4):
    """Create a PPO Actor model"""
    class PPOActor(nn.Module):
        def __init__(self, state_dim, hidden_dim, action_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
            
        def forward(self, x):
            return self.network(x)
    
    return PPOActor(state_dim, hidden_dim, action_dim)

def create_ppo_critic_model(state_dim=128, hidden_dim=256):
    """Create a PPO Critic model"""
    class PPOCritic(nn.Module):
        def __init__(self, state_dim, hidden_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, x):
            return self.network(x)
    
    return PPOCritic(state_dim, hidden_dim)

def create_sac_actor_model(state_dim=128, hidden_dim=256, action_dim=4):
    """Create a SAC Actor model"""
    class SACActor(nn.Module):
        def __init__(self, state_dim, hidden_dim, action_dim):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)
            
        def forward(self, x):
            shared = self.shared(x)
            mean = self.mean(shared)
            log_std = self.log_std(shared)
            return mean, log_std
    
    return SACActor(state_dim, hidden_dim, action_dim)

def create_physics_net_model(input_dim=64, hidden_dim=256, output_dim=32):
    """Create a physics simulation network"""
    class PhysicsNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.network(x)
    
    return PhysicsNet(input_dim, hidden_dim, output_dim)

def create_chem_net_model(atom_features=16, bond_features=8, hidden_dim=256):
    """Create a chemistry network for molecular property prediction"""
    class ChemNet(nn.Module):
        def __init__(self, atom_features, bond_features, hidden_dim):
            super().__init__()
            self.atom_embedding = nn.Linear(atom_features, hidden_dim)
            self.bond_embedding = nn.Linear(bond_features, hidden_dim)
            
            self.message_passing = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, x):
            # Simplified molecular representation
            atom_emb = self.atom_embedding(x)
            
            # Simple message passing simulation
            messages = torch.cat([atom_emb, torch.roll(atom_emb, 1, dim=1)], dim=-1)
            updated = self.message_passing(messages)
            
            # Global pooling and readout
            global_features = updated.mean(dim=1)
            return self.readout(global_features)
    
    return ChemNet(atom_features, bond_features, hidden_dim)

def create_pde_solver_model(spatial_dim=64, temporal_dim=32, hidden_dim=256):
    """Create a PDE solver network"""
    class PDESolver(nn.Module):
        def __init__(self, spatial_dim, temporal_dim, hidden_dim):
            super().__init__()
            self.spatial_conv = nn.Conv1d(spatial_dim, hidden_dim, kernel_size=3, padding=1)
            self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.output_conv = nn.Conv1d(hidden_dim, spatial_dim, kernel_size=3, padding=1)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch, spatial, temporal)
            batch_size, spatial_dim, temporal_dim = x.shape
            
            # Process spatially
            spatial_features = self.activation(self.spatial_conv(x))
            
            # Process temporally
            spatial_features = spatial_features.transpose(1, 2)  # (batch, temporal, hidden)
            temporal_features, _ = self.temporal_lstm(spatial_features)
            
            # Back to spatial processing
            temporal_features = temporal_features.transpose(1, 2)  # (batch, hidden, temporal)
            output = self.output_conv(temporal_features)
            
            return output
    
    return PDESolver(spatial_dim, temporal_dim, hidden_dim)

def create_md_model(particle_dim=32, hidden_dim=256, num_layers=4):
    """Create a molecular dynamics simulation network"""
    class MDNet(nn.Module):
        def __init__(self, particle_dim, hidden_dim, num_layers):
            super().__init__()
            self.particle_embedding = nn.Linear(3, hidden_dim)  # 3D coordinates
            
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.force_prediction = nn.Linear(hidden_dim, 3)  # 3D force vectors
            self.activation = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch, particles, 3) - coordinates
            batch_size, num_particles, _ = x.shape
            
            # Embed particle positions
            embedded = self.particle_embedding(x)
            
            # Process through layers
            features = embedded
            for layer in self.layers:
                features = self.activation(layer(features))
            
            # Predict forces
            forces = self.force_prediction(features)
            
            return forces
    
    return MDNet(particle_dim, hidden_dim, num_layers)

def create_models():
    """Create comprehensive TorchBench-style model suite covering all major domains"""
    models_dict = {}
    
    print("Loading COMPREHENSIVE TorchBench model suite...")
    
    # === COMPUTER VISION MODELS ===
    print("  Loading Vision Models...")
    
    # CNN Architectures
    vision_models = [
        # ResNet Family
        ('ResNet18', lambda: models.resnet18(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ResNet34', lambda: models.resnet34(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ResNet50', lambda: models.resnet50(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ResNet101', lambda: models.resnet101(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ResNet152', lambda: models.resnet152(weights='DEFAULT'), (32, 3, 224, 224)),
        ('Wide-ResNet50-2', lambda: models.wide_resnet50_2(weights='DEFAULT'), (32, 3, 224, 224)),
        ('Wide-ResNet101-2', lambda: models.wide_resnet101_2(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # Vision Transformers
        ('ViT-Base-16', lambda: models.vit_b_16(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ViT-Base-32', lambda: models.vit_b_32(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ViT-Large-16', lambda: models.vit_l_16(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ViT-Large-32', lambda: models.vit_l_32(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ViT-Huge-14', lambda: models.vit_h_14(weights='DEFAULT'), (32, 3, 518, 518)),
        
        # DenseNet Family
        ('DenseNet121', lambda: models.densenet121(weights='DEFAULT'), (32, 3, 224, 224)),
        ('DenseNet161', lambda: models.densenet161(weights='DEFAULT'), (32, 3, 224, 224)),
        ('DenseNet169', lambda: models.densenet169(weights='DEFAULT'), (32, 3, 224, 224)),
        ('DenseNet201', lambda: models.densenet201(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # EfficientNet Family
        ('EfficientNet-B0', lambda: models.efficientnet_b0(weights='DEFAULT'), (32, 3, 224, 224)),
        ('EfficientNet-B1', lambda: models.efficientnet_b1(weights='DEFAULT'), (32, 3, 240, 240)),
        ('EfficientNet-B2', lambda: models.efficientnet_b2(weights='DEFAULT'), (32, 3, 260, 260)),
        ('EfficientNet-B3', lambda: models.efficientnet_b3(weights='DEFAULT'), (32, 3, 300, 300)),
        ('EfficientNet-B4', lambda: models.efficientnet_b4(weights='DEFAULT'), (32, 3, 380, 380)),
        ('EfficientNet-B5', lambda: models.efficientnet_b5(weights='DEFAULT'), (32, 3, 456, 456)),
        ('EfficientNet-B6', lambda: models.efficientnet_b6(weights='DEFAULT'), (32, 3, 528, 528)),
        ('EfficientNet-B7', lambda: models.efficientnet_b7(weights='DEFAULT'), (32, 3, 600, 600)),
        ('EfficientNetV2-S', lambda: models.efficientnet_v2_s(weights='DEFAULT'), (32, 3, 384, 384)),
        ('EfficientNetV2-M', lambda: models.efficientnet_v2_m(weights='DEFAULT'), (32, 3, 480, 480)),
        ('EfficientNetV2-L', lambda: models.efficientnet_v2_l(weights='DEFAULT'), (32, 3, 480, 480)),
        
        # MobileNet Family
        ('MobileNet-V2', lambda: models.mobilenet_v2(weights='DEFAULT'), (32, 3, 224, 224)),
        ('MobileNet-V3-Large', lambda: models.mobilenet_v3_large(weights='DEFAULT'), (32, 3, 224, 224)),
        ('MobileNet-V3-Small', lambda: models.mobilenet_v3_small(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # RegNet Family
        ('RegNet-X-400MF', lambda: models.regnet_x_400mf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-800MF', lambda: models.regnet_x_800mf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-1-6GF', lambda: models.regnet_x_1_6gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-3-2GF', lambda: models.regnet_x_3_2gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-8GF', lambda: models.regnet_x_8gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-16GF', lambda: models.regnet_x_16gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-X-32GF', lambda: models.regnet_x_32gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-400MF', lambda: models.regnet_y_400mf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-800MF', lambda: models.regnet_y_800mf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-1-6GF', lambda: models.regnet_y_1_6gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-3-2GF', lambda: models.regnet_y_3_2gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-8GF', lambda: models.regnet_y_8gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-16GF', lambda: models.regnet_y_16gf(weights='DEFAULT'), (32, 3, 224, 224)),
        ('RegNet-Y-32GF', lambda: models.regnet_y_32gf(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # ConvNeXt Family
        ('ConvNeXt-Tiny', lambda: models.convnext_tiny(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ConvNeXt-Small', lambda: models.convnext_small(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ConvNeXt-Base', lambda: models.convnext_base(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ConvNeXt-Large', lambda: models.convnext_large(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # Swin Transformer Family
        ('Swin-Tiny', lambda: models.swin_t(weights='DEFAULT'), (32, 3, 224, 224)),
        ('Swin-Small', lambda: models.swin_s(weights='DEFAULT'), (32, 3, 224, 224)),
        ('Swin-Base', lambda: models.swin_b(weights='DEFAULT'), (32, 3, 224, 224)),
        ('Swin-Large', lambda: models.swin_l(weights='DEFAULT'), (32, 3, 224, 224)),
        ('SwinV2-Tiny', lambda: models.swin_v2_t(weights='DEFAULT'), (32, 3, 256, 256)),
        ('SwinV2-Small', lambda: models.swin_v2_s(weights='DEFAULT'), (32, 3, 256, 256)),
        ('SwinV2-Base', lambda: models.swin_v2_b(weights='DEFAULT'), (32, 3, 256, 256)),
        
        # MaxViT Family
        ('MaxViT-Tiny', lambda: models.maxvit_t(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # Inception Family
        ('Inception-V3', lambda: models.inception_v3(weights='DEFAULT'), (32, 3, 299, 299)),
        
        # SqueezeNet Family
        ('SqueezeNet-1.0', lambda: models.squeezenet1_0(weights='DEFAULT'), (32, 3, 224, 224)),
        ('SqueezeNet-1.1', lambda: models.squeezenet1_1(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # AlexNet and VGG
        ('AlexNet', lambda: models.alexnet(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG11', lambda: models.vgg11(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG11-BN', lambda: models.vgg11_bn(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG13', lambda: models.vgg13(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG13-BN', lambda: models.vgg13_bn(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG16', lambda: models.vgg16(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG16-BN', lambda: models.vgg16_bn(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG19', lambda: models.vgg19(weights='DEFAULT'), (32, 3, 224, 224)),
        ('VGG19-BN', lambda: models.vgg19_bn(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # ShuffleNet Family
        ('ShuffleNet-V2-x0.5', lambda: models.shufflenet_v2_x0_5(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ShuffleNet-V2-x1.0', lambda: models.shufflenet_v2_x1_0(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ShuffleNet-V2-x1.5', lambda: models.shufflenet_v2_x1_5(weights='DEFAULT'), (32, 3, 224, 224)),
        ('ShuffleNet-V2-x2.0', lambda: models.shufflenet_v2_x2_0(weights='DEFAULT'), (32, 3, 224, 224)),
        
        # MNASNet Family
        ('MNASNet-0.5', lambda: models.mnasnet0_5(weights='DEFAULT'), (32, 3, 224, 224)),
        ('MNASNet-0.75', lambda: models.mnasnet0_75(weights='DEFAULT'), (32, 3, 224, 224)),
        ('MNASNet-1.0', lambda: models.mnasnet1_0(weights='DEFAULT'), (32, 3, 224, 224)),
        ('MNASNet-1.3', lambda: models.mnasnet1_3(weights='DEFAULT'), (32, 3, 224, 224)),
    ]
    
    for name, model_fn, input_shape in vision_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'vision',
                'input_type': 'tensor'
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === NATURAL LANGUAGE PROCESSING MODELS ===
    print("  Loading NLP Models...")
    
    # Various scale LSTM models
    lstm_models = [
        ('LSTM-Micro', lambda: create_simple_lstm(vocab_size=1000, embed_dim=64, hidden_dim=128, num_layers=1), (16, 32), 'ids'),
        ('LSTM-Tiny', lambda: create_simple_lstm(vocab_size=2000, embed_dim=96, hidden_dim=192, num_layers=2), (16, 48), 'ids'),
        ('LSTM-Small', lambda: create_simple_lstm(vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=2), (32, 64), 'ids'),
        ('LSTM-Medium', lambda: create_simple_lstm(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=3), (32, 128), 'ids'),
        ('LSTM-Large', lambda: create_simple_lstm(vocab_size=15000, embed_dim=512, hidden_dim=1024, num_layers=4), (32, 256), 'ids'),
        ('LSTM-XLarge', lambda: create_simple_lstm(vocab_size=20000, embed_dim=768, hidden_dim=1536, num_layers=6), (16, 512), 'ids'),
        
        # Bidirectional variants
        ('BiLSTM-Small', lambda: create_bidirectional_lstm(vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=2), (32, 64), 'ids'),
        ('BiLSTM-Medium', lambda: create_bidirectional_lstm(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=3), (32, 128), 'ids'),
        ('BiLSTM-Large', lambda: create_bidirectional_lstm(vocab_size=15000, embed_dim=512, hidden_dim=1024, num_layers=4), (32, 256), 'ids'),
    ]
    
    # Various scale Transformer models
    transformer_models = [
        ('Transformer-Nano', lambda: create_simple_transformer(vocab_size=1000, d_model=64, nhead=2, num_layers=2), (16, 32), 'ids'),
        ('Transformer-Micro', lambda: create_simple_transformer(vocab_size=2000, d_model=128, nhead=4, num_layers=3), (16, 48), 'ids'),
        ('Transformer-Tiny', lambda: create_simple_transformer(vocab_size=3000, d_model=192, nhead=6, num_layers=4), (16, 64), 'ids'),
        ('Transformer-Small', lambda: create_simple_transformer(vocab_size=5000, d_model=256, nhead=8, num_layers=6), (32, 64), 'ids'),
        ('Transformer-Medium', lambda: create_simple_transformer(vocab_size=10000, d_model=512, nhead=8, num_layers=8), (32, 128), 'ids'),
        ('Transformer-Large', lambda: create_simple_transformer(vocab_size=15000, d_model=768, nhead=12, num_layers=12), (16, 256), 'ids'),
        ('Transformer-XLarge', lambda: create_simple_transformer(vocab_size=20000, d_model=1024, nhead=16, num_layers=16), (16, 512), 'ids'),
        ('Transformer-XXLarge', lambda: create_simple_transformer(vocab_size=25000, d_model=1536, nhead=24, num_layers=24), (8, 1024), 'ids'),
    ]
    
    # GRU Models
    gru_models = [
        ('GRU-Small', lambda: create_simple_gru(vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=2), (32, 64), 'ids'),
        ('GRU-Medium', lambda: create_simple_gru(vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=3), (32, 128), 'ids'),
        ('GRU-Large', lambda: create_simple_gru(vocab_size=15000, embed_dim=512, hidden_dim=1024, num_layers=4), (32, 256), 'ids'),
    ]
    
    all_nlp_models = lstm_models + transformer_models + gru_models
    
    for name, model_fn, input_shape, input_type in all_nlp_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'nlp',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === GRAPH NEURAL NETWORKS ===
    print("  Loading Graph Neural Network Models...")
    
    graph_models = [
        ('GCN-Small', lambda: create_gcn_model(input_dim=128, hidden_dim=256, output_dim=64, num_layers=3), (32, 128), 'graph'),
        ('GCN-Medium', lambda: create_gcn_model(input_dim=256, hidden_dim=512, output_dim=128, num_layers=5), (64, 256), 'graph'),
        ('GCN-Large', lambda: create_gcn_model(input_dim=512, hidden_dim=1024, output_dim=256, num_layers=8), (128, 512), 'graph'),
        ('GraphSAGE-Small', lambda: create_graphsage_model(input_dim=128, hidden_dim=256, output_dim=64, num_layers=3), (32, 128), 'graph'),
        ('GraphSAGE-Medium', lambda: create_graphsage_model(input_dim=256, hidden_dim=512, output_dim=128, num_layers=5), (64, 256), 'graph'),
        ('GraphSAGE-Large', lambda: create_graphsage_model(input_dim=512, hidden_dim=1024, output_dim=256, num_layers=8), (128, 512), 'graph'),
        ('GAT-Small', lambda: create_gat_model(input_dim=128, hidden_dim=256, output_dim=64, num_heads=4, num_layers=3), (32, 128), 'graph'),
        ('GAT-Medium', lambda: create_gat_model(input_dim=256, hidden_dim=512, output_dim=128, num_heads=8, num_layers=5), (64, 256), 'graph'),
        ('GAT-Large', lambda: create_gat_model(input_dim=512, hidden_dim=1024, output_dim=256, num_heads=12, num_layers=8), (128, 512), 'graph'),
    ]
    
    for name, model_fn, input_shape, input_type in graph_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'graph',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === AUDIO/SPEECH MODELS ===
    print("  Loading Audio/Speech Models...")
    
    audio_models = [
        ('WaveNet-Small', lambda: create_wavenet_model(input_dim=80, hidden_dim=256, num_layers=8), (32, 80, 1000), 'audio'),
        ('WaveNet-Medium', lambda: create_wavenet_model(input_dim=80, hidden_dim=512, num_layers=16), (32, 80, 2000), 'audio'),
        ('WaveNet-Large', lambda: create_wavenet_model(input_dim=80, hidden_dim=1024, num_layers=24), (16, 80, 4000), 'audio'),
        ('DeepSpeech-Small', lambda: create_deepspeech_model(input_dim=80, hidden_dim=512, num_layers=5), (32, 80, 1500), 'audio'),
        ('DeepSpeech-Medium', lambda: create_deepspeech_model(input_dim=80, hidden_dim=1024, num_layers=7), (32, 80, 3000), 'audio'),
        ('DeepSpeech-Large', lambda: create_deepspeech_model(input_dim=80, hidden_dim=2048, num_layers=10), (16, 80, 6000), 'audio'),
        ('Tacotron2-Small', lambda: create_tacotron2_model(vocab_size=100, embedding_dim=256, hidden_dim=512), (16, 100), 'audio'),
        ('Tacotron2-Medium', lambda: create_tacotron2_model(vocab_size=200, embedding_dim=512, hidden_dim=1024), (16, 200), 'audio'),
        ('Tacotron2-Large', lambda: create_tacotron2_model(vocab_size=300, embedding_dim=768, hidden_dim=1536), (8, 300), 'audio'),
    ]
    
    for name, model_fn, input_shape, input_type in audio_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'audio',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === MULTIMODAL MODELS ===
    print("  Loading Multimodal Models...")
    
    multimodal_models = [
        ('CLIP-Small', lambda: create_clip_model(text_dim=256, vision_dim=256, embed_dim=256), (16, 77), 'multimodal'),
        ('CLIP-Medium', lambda: create_clip_model(text_dim=512, vision_dim=512, embed_dim=512), (16, 77), 'multimodal'),
        ('CLIP-Large', lambda: create_clip_model(text_dim=768, vision_dim=768, embed_dim=768), (16, 77), 'multimodal'),
        ('VisualBERT-Small', lambda: create_visual_bert_model(vocab_size=5000, text_dim=256, vision_dim=256), (16, 64), 'multimodal'),
        ('VisualBERT-Medium', lambda: create_visual_bert_model(vocab_size=10000, text_dim=512, vision_dim=512), (16, 128), 'multimodal'),
        ('VisualBERT-Large', lambda: create_visual_bert_model(vocab_size=15000, text_dim=768, vision_dim=768), (16, 256), 'multimodal'),
    ]
    
    for name, model_fn, input_shape, input_type in multimodal_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'multimodal',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === GENERATIVE MODELS ===
    print("  Loading Generative Models...")
    
    generative_models = [
        ('VAE-Small', lambda: create_vae_model(input_dim=784, hidden_dim=256, latent_dim=64), (32, 784), 'tensor'),
        ('VAE-Medium', lambda: create_vae_model(input_dim=3072, hidden_dim=512, latent_dim=128), (32, 3072), 'tensor'),
        ('VAE-Large', lambda: create_vae_model(input_dim=12288, hidden_dim=1024, latent_dim=256), (16, 12288), 'tensor'),
        ('GAN-Generator-Small', lambda: create_gan_generator(noise_dim=100, hidden_dim=256, output_dim=784), (32, 100), 'tensor'),
        ('GAN-Generator-Medium', lambda: create_gan_generator(noise_dim=128, hidden_dim=512, output_dim=3072), (32, 128), 'tensor'),
        ('GAN-Generator-Large', lambda: create_gan_generator(noise_dim=256, hidden_dim=1024, output_dim=12288), (16, 256), 'tensor'),
        ('GAN-Discriminator-Small', lambda: create_gan_discriminator(input_dim=784, hidden_dim=256), (32, 784), 'tensor'),
        ('GAN-Discriminator-Medium', lambda: create_gan_discriminator(input_dim=3072, hidden_dim=512), (32, 3072), 'tensor'),
        ('GAN-Discriminator-Large', lambda: create_gan_discriminator(input_dim=12288, hidden_dim=1024), (16, 12288), 'tensor'),
        ('UNet-Small', lambda: create_unet_model(in_channels=3, out_channels=3, hidden_dim=64), (16, 3, 64, 64), 'tensor'),
        ('UNet-Medium', lambda: create_unet_model(in_channels=3, out_channels=3, hidden_dim=128), (16, 3, 128, 128), 'tensor'),
        ('UNet-Large', lambda: create_unet_model(in_channels=3, out_channels=3, hidden_dim=256), (8, 3, 256, 256), 'tensor'),
    ]
    
    for name, model_fn, input_shape, input_type in generative_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'generative',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === REINFORCEMENT LEARNING MODELS ===
    print("  Loading Reinforcement Learning Models...")
    
    rl_models = [
        ('DQN-Small', lambda: create_dqn_model(state_dim=128, hidden_dim=256, action_dim=4), (32, 128), 'tensor'),
        ('DQN-Medium', lambda: create_dqn_model(state_dim=256, hidden_dim=512, action_dim=8), (32, 256), 'tensor'),
        ('DQN-Large', lambda: create_dqn_model(state_dim=512, hidden_dim=1024, action_dim=16), (32, 512), 'tensor'),
        ('PPO-Actor-Small', lambda: create_ppo_actor_model(state_dim=128, hidden_dim=256, action_dim=4), (32, 128), 'tensor'),
        ('PPO-Actor-Medium', lambda: create_ppo_actor_model(state_dim=256, hidden_dim=512, action_dim=8), (32, 256), 'tensor'),
        ('PPO-Actor-Large', lambda: create_ppo_actor_model(state_dim=512, hidden_dim=1024, action_dim=16), (32, 512), 'tensor'),
        ('PPO-Critic-Small', lambda: create_ppo_critic_model(state_dim=128, hidden_dim=256), (32, 128), 'tensor'),
        ('PPO-Critic-Medium', lambda: create_ppo_critic_model(state_dim=256, hidden_dim=512), (32, 256), 'tensor'),
        ('PPO-Critic-Large', lambda: create_ppo_critic_model(state_dim=512, hidden_dim=1024), (32, 512), 'tensor'),
        ('SAC-Actor-Small', lambda: create_sac_actor_model(state_dim=128, hidden_dim=256, action_dim=4), (32, 128), 'tensor'),
        ('SAC-Actor-Medium', lambda: create_sac_actor_model(state_dim=256, hidden_dim=512, action_dim=8), (32, 256), 'tensor'),
        ('SAC-Actor-Large', lambda: create_sac_actor_model(state_dim=512, hidden_dim=1024, action_dim=16), (32, 512), 'tensor'),
    ]
    
    for name, model_fn, input_shape, input_type in rl_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'reinforcement_learning',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    # === SCIENTIFIC COMPUTING MODELS ===
    print("  Loading Scientific Computing Models...")
    
    scientific_models = [
        ('PhysicsNet-Small', lambda: create_physics_net_model(input_dim=64, hidden_dim=256, output_dim=32), (32, 64), 'tensor'),
        ('PhysicsNet-Medium', lambda: create_physics_net_model(input_dim=128, hidden_dim=512, output_dim=64), (32, 128), 'tensor'),
        ('PhysicsNet-Large', lambda: create_physics_net_model(input_dim=256, hidden_dim=1024, output_dim=128), (32, 256), 'tensor'),
        ('ChemNet-Small', lambda: create_chem_net_model(atom_features=16, bond_features=8, hidden_dim=256), (32, 16), 'tensor'),
        ('ChemNet-Medium', lambda: create_chem_net_model(atom_features=32, bond_features=16, hidden_dim=512), (32, 32), 'tensor'),
        ('ChemNet-Large', lambda: create_chem_net_model(atom_features=64, bond_features=32, hidden_dim=1024), (32, 64), 'tensor'),
        ('PDE-Solver-Small', lambda: create_pde_solver_model(spatial_dim=64, temporal_dim=32, hidden_dim=256), (16, 64, 32), 'tensor'),
        ('PDE-Solver-Medium', lambda: create_pde_solver_model(spatial_dim=128, temporal_dim=64, hidden_dim=512), (16, 128, 64), 'tensor'),
        ('PDE-Solver-Large', lambda: create_pde_solver_model(spatial_dim=256, temporal_dim=128, hidden_dim=1024), (8, 256, 128), 'tensor'),
        ('MolecularDynamics-Small', lambda: create_md_model(particle_dim=32, hidden_dim=256, num_layers=4), (32, 32, 3), 'tensor'),
        ('MolecularDynamics-Medium', lambda: create_md_model(particle_dim=64, hidden_dim=512, num_layers=6), (32, 64, 3), 'tensor'),
        ('MolecularDynamics-Large', lambda: create_md_model(particle_dim=128, hidden_dim=1024, num_layers=8), (16, 128, 3), 'tensor'),
    ]
    
    for name, model_fn, input_shape, input_type in scientific_models:
        try:
            models_dict[name] = {
                'model': model_fn().to(DEVICE).eval(),
                'input_shape': input_shape,
                'category': 'scientific',
                'input_type': input_type
            }
        except Exception as e:
            print(f"    Failed to load {name}: {e}")
    
    print(f"\nSuccessfully loaded {len(models_dict)} models across categories:")
    categories = {}
    for model_info in models_dict.values():
        cat = model_info['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  - {cat}: {count} models")
    
    return models_dict

def create_input_tensor(input_shape, input_type='tensor'):
    """Create input tensor based on shape and type"""
    if input_type == 'ids':
        # For sequence models - create token IDs
        vocab_size = 10000  # Conservative vocab size
        return torch.randint(0, vocab_size, input_shape, device=DEVICE)
    else:
        # For vision models - create image tensors
        return torch.randn(input_shape, device=DEVICE)

def measure_eager_inference(model, input_tensor, model_name):
    """Measure eager mode inference time"""
    print(f"    Measuring eager mode for {model_name}...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(input_tensor)
    
    clear_cache()
    
    # Actual timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
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

def measure_compilation_time(model, input_tensor, model_name):
    """Measure actual compilation time"""
    print(f"    Measuring compilation time for {model_name}...")
    
    clear_cache()
    
    # Measure compilation time
    start_time = time.perf_counter()
    try:
        compiled_model = torch.compile(model, backend='inductor', mode='default')
        
        # First forward pass triggers compilation
        with torch.no_grad():
            _ = compiled_model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        end_time = time.perf_counter()
        compilation_time = end_time - start_time
        
        return compiled_model, compilation_time
        
    except Exception as e:
        print(f"    Compilation failed for {model_name}: {e}")
        return None, 0.0

def measure_compiled_inference(compiled_model, input_tensor, model_name):
    """Measure compiled mode inference time"""
    print(f"    Measuring compiled mode for {model_name}...")
    
    if compiled_model is None:
        return None
    
    # Warmup (compilation already done)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = compiled_model(input_tensor)
    
    clear_cache()
    
    # Actual timing
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(TIMING_RUNS):
            start_memory = get_memory_usage()
            
            start_time = time.perf_counter()
            _ = compiled_model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
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

def measure_guard_times(compiled_model, base_input_shape, model_name, input_type='tensor'):
    """Measure guard checking and recompilation times with shape changes"""
    print(f"    Measuring guard times for {model_name}...")
    
    if compiled_model is None:
        return {
            'guard_check_time': 0.0,
            'recompile_time': 0.0,
            'guard_failure_rate': 0.0,
            'recompile_count': 0
        }
    
    guard_times = []
    recompile_times = []
    recompile_count = 0
    
    # Test with consistent shape (should hit cache)
    consistent_input = create_input_tensor(base_input_shape, input_type)
    
    for _ in range(GUARD_RUNS):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(consistent_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        guard_times.append(end_time - start_time)
    
    # Test with different shapes to trigger recompilation
    test_shapes = []
    if input_type == 'tensor':
        # For vision models - change batch size
        for batch_size in [16, 48, 80]:
            new_shape = list(base_input_shape)
            new_shape[0] = batch_size
            test_shapes.append(tuple(new_shape))
    else:
        # For sequence models - change sequence length
        for seq_len in [32, 96, 200]:
            new_shape = list(base_input_shape)
            new_shape[1] = seq_len  # Change sequence length
            test_shapes.append(tuple(new_shape))
    
    for new_shape in test_shapes:
        try:
            new_input = create_input_tensor(new_shape, input_type)
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = compiled_model(new_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            recompile_time = end_time - start_time
            recompile_times.append(recompile_time)
            recompile_count += 1
            
        except Exception as e:
            print(f"      Recompilation failed for shape {new_shape}: {e}")
    
    return {
        'guard_check_time': np.mean(guard_times),
        'guard_check_std': np.std(guard_times),
        'recompile_time': np.mean(recompile_times) if recompile_times else 0.0,
        'guard_failure_rate': recompile_count / (GUARD_RUNS + len(test_shapes)),
        'recompile_count': recompile_count
    }

def run_full_benchmark():
    """Run complete TorchBench-style benchmarking experiment"""
    print("=" * 80)
    print("TORCHBENCH-STYLE PYTORCH MODEL BENCHMARKING EXPERIMENT")
    print("=" * 80)
    
    # Load models
    models_dict = create_models()
    
    if not models_dict:
        print("No models loaded successfully!")
        return None
    
    # Results storage
    results = []
    
    for model_name, model_info in models_dict.items():
        print(f"\nBenchmarking {model_name}...")
        print("-" * 50)
        
        model = model_info['model']
        input_shape = model_info['input_shape']
        input_type = model_info.get('input_type', 'tensor')
        category = model_info['category']
        
        # Create input tensor
        input_tensor = create_input_tensor(input_shape, input_type)
        print(f"  Input shape: {input_shape}")
        print(f"  Category: {category}")
        print(f"  Input type: {input_type}")
        
        try:
            # 1. Measure eager mode
            eager_results = measure_eager_inference(model, input_tensor, model_name)
            
            # 2. Measure compilation time and get compiled model
            compiled_model, compilation_time = measure_compilation_time(model, input_tensor, model_name)
            
            # 3. Measure compiled mode
            compiled_results = measure_compiled_inference(compiled_model, input_tensor, model_name)
            
            # 4. Measure guard times
            guard_results = measure_guard_times(compiled_model, input_shape, model_name, input_type)
            
            # Compile results
            if compiled_results is not None:
                speedup = eager_results['mean_time'] / compiled_results['mean_time']
                
                result = {
                    'model_name': model_name,
                    'category': category,
                    'input_shape': str(input_shape),
                    'input_type': input_type,
                    'baseline_time': eager_results['mean_time'],
                    'baseline_std': eager_results['std_time'],
                    'compiled_time': compiled_results['mean_time'],
                    'compiled_std': compiled_results['std_time'],
                    'speedup': speedup,
                    'compile_time': compilation_time,
                    'guard_check_time': guard_results['guard_check_time'],
                    'guard_check_std': guard_results.get('guard_check_std', 0.0),
                    'recompile_time': guard_results['recompile_time'],
                    'guard_failure_rate': guard_results['guard_failure_rate'],
                    'recompile_count': guard_results['recompile_count'],
                    'eager_memory': eager_results['memory_usage'],
                    'compiled_memory': compiled_results['memory_usage']
                }
                
                results.append(result)
                
                print(f"  ✓ Eager time: {eager_results['mean_time']*1000:.2f}±{eager_results['std_time']*1000:.2f}ms")
                print(f"  ✓ Compiled time: {compiled_results['mean_time']*1000:.2f}±{compiled_results['std_time']*1000:.2f}ms")
                print(f"  ✓ Speedup: {speedup:.2f}x")
                print(f"  ✓ Compilation time: {compilation_time:.2f}s")
                print(f"  ✓ Guard time: {guard_results['guard_check_time']*1000:.3f}ms")
                
            else:
                print(f"  ✗ Compilation failed for {model_name}")
                
        except Exception as e:
            print(f"  ✗ Error benchmarking {model_name}: {e}")
        
        clear_cache()
    
    if results:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_file = 'torchbench_style_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
        
        # Print summary by category
        print(f"\n" + "=" * 80)
        print("TORCHBENCH-STYLE BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total models tested: {len(results)}")
        print(f"Successful compilations: {len([r for r in results if r['speedup'] > 0])}")
        
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            print(f"\n{category.upper()} MODELS ({len(cat_df)} models):")
            print(f"  Average speedup: {cat_df['speedup'].mean():.2f}x")
            print(f"  Best speedup: {cat_df['speedup'].max():.2f}x ({cat_df.loc[cat_df['speedup'].idxmax(), 'model_name']})")
            print(f"  Average compilation time: {cat_df['compile_time'].mean():.2f}s")
            print(f"  Average guard time: {cat_df['guard_check_time'].mean()*1000:.3f}ms")
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Average speedup: {df['speedup'].mean():.2f}x")
        print(f"  Best speedup: {df['speedup'].max():.2f}x ({df.loc[df['speedup'].idxmax(), 'model_name']})")
        print(f"  Average compilation time: {df['compile_time'].mean():.2f}s")
        print(f"  Average guard time: {df['guard_check_time'].mean()*1000:.3f}ms")
        
        return df
    
    else:
        print("No successful benchmarks!")
        return None

if __name__ == "__main__":
    results_df = run_full_benchmark()
    
    if results_df is not None:
        print(f"\nTorchBench-style experimental data collected!")
        print(f"Use this data with: python generate_model_charts_from_data.py --data_file torchbench_style_results.csv") 