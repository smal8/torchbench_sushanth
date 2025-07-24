"""
TorchBench Attention Models Benchmark: Tests attention-based models from TorchBench
covering Vision, NLP, Speech, and Multimodal models.

Vision Models: DeiT, Swin Transformer, ConvNeXt, BEiT, RegNetY
NLP Models: BERT, GPT-2, XLNet, ALBERT, T5, BART, Electra
Speech Models: Wav2Vec2, HuBERT  
Multimodal: CLIP

Metrics:
- Compile time: How long PyTorch 2's torch.compile() takes
- Baseline inference time: Standard PyTorch model performance
- Compiled inference time: Performance after compilation
- Speedup: How much faster the compiled model is
- Throughput: Samples processed per second
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import gc
import warnings
import os
import sys
from typing import Dict, Any, Tuple, Optional

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
BATCH_SIZES = [1, 4, 8, 16, 32]
NUM_WARMUP_RUNS = 50  # Reduced for diverse models
NUM_TIMING_RUNS = 50   # Reduced for diverse models
NUM_GUARD_RUNS = 20     # For guard time measurement

# Model configurations for different types
MODEL_CONFIGS = {
    # Vision Models with Attention
    'vision': {
        'input_shape': (3, 224, 224),
        'models': {
            'deit_base': {
                'create_fn': lambda: create_deit_model(),
                'description': 'DeiT Base - Vision Transformer with distillation'
            },
            'swin_base': {
                'create_fn': lambda: create_swin_model(),
                'description': 'Swin Transformer Base - Hierarchical vision transformer'
            },
            'convnext_base': {
                'create_fn': lambda: create_convnext_model(),
                'description': 'ConvNeXt Base - CNN with attention-like mechanisms'
            },
            'beit_base': {
                'create_fn': lambda: create_beit_model(),
                'description': 'BEiT Base - BERT pre-training of image transformers'
            },
            'regnet_y': {
                'create_fn': lambda: create_regnet_model(),
                'description': 'RegNetY - CNN with SE (Squeeze-and-Excitation) blocks'
            }
        }
    },
    
    # NLP Models with Attention
    'nlp': {
        'input_shape': (512,),  # sequence length
        'models': {
            'bert_base': {
                'create_fn': lambda: create_bert_model(),
                'description': 'BERT Base - Bidirectional transformer encoder'
            },
            'gpt2': {
                'create_fn': lambda: create_gpt2_model(),
                'description': 'GPT-2 - Autoregressive transformer decoder'
            },
            'distilbert': {
                'create_fn': lambda: create_distilbert_model(),
                'description': 'DistilBERT - Distilled version of BERT'
            },
            'albert': {
                'create_fn': lambda: create_albert_model(),
                'description': 'ALBERT - A Lite BERT with parameter sharing'
            },
            't5_small': {
                'create_fn': lambda: create_t5_model(),
                'description': 'T5 Small - Text-to-Text Transfer Transformer'
            }
        }
    },
    
    # Speech Models with Attention
    'speech': {
        'input_shape': (16000,),  # 1 second of 16kHz audio
        'models': {
            'wav2vec2': {
                'create_fn': lambda: create_wav2vec2_model(),
                'description': 'Wav2Vec2 - Transformer-based speech representation'
            },
            'hubert': {
                'create_fn': lambda: create_hubert_model(),
                'description': 'HuBERT - Hidden-Unit BERT for speech'
            }
        }
    },
    
    # Multimodal Models
    'multimodal': {
        'input_shape': {'vision': (3, 224, 224), 'text': (77,)},
        'models': {
            'clip': {
                'create_fn': lambda: create_clip_model(),
                'description': 'CLIP - Contrastive Language-Image Pre-training'
            }
        }
    }
}

def try_import_torchbench():
    """Try to import torchbenchmark models with fallback to mock implementations"""
    try:
        import torchbenchmark
        return True, torchbenchmark
    except ImportError:
        print("TorchBench not available, using mock implementations for demonstration")
        return False, None

# Check TorchBench availability
TORCHBENCH_AVAILABLE, torchbench = try_import_torchbench()

def create_mock_transformer_model(input_dim: int, hidden_dim: int = 768, num_layers: int = 12, num_heads: int = 12):
    """Create a mock transformer model for testing when TorchBench is not available"""
    import torch.nn as nn
    
    class MockTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(50000, hidden_dim) if input_dim == 1 else nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output = nn.Linear(hidden_dim, 1000)  # Classification head
            
        def forward(self, x):
            if len(x.shape) == 2 and x.dtype == torch.long:  # Token indices
                x = self.embedding(x)
            elif len(x.shape) == 2:  # Already embedded or flattened
                x = x.unsqueeze(1)
                x = self.embedding(x)
            elif len(x.shape) == 4:  # Image data
                b, c, h, w = x.shape
                x = x.reshape(b, c * h * w)
                x = self.embedding(x).unsqueeze(1)
            
            x = self.transformer(x)
            return self.output(x.mean(dim=1))
    
    return MockTransformerModel()

# Model creation functions (with fallbacks to mock implementations)
def create_deit_model():
    if TORCHBENCH_AVAILABLE:
        try:
            from torchbenchmark.models.deit import Model
            return Model(test="eval").model
        except:
            pass
    # Fallback to timm if available, otherwise mock
    try:
        import timm
        return timm.create_model('deit_base_patch16_224', pretrained=False)
    except:
        return create_mock_transformer_model(3 * 224 * 224)

def create_swin_model():
    if TORCHBENCH_AVAILABLE:
        try:
            from torchbenchmark.models.swin_transformer import Model
            return Model(test="eval").model
        except:
            pass
    try:
        import timm
        return timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    except:
        return create_mock_transformer_model(3 * 224 * 224)

def create_convnext_model():
    try:
        import timm
        return timm.create_model('convnext_base', pretrained=False)
    except:
        return create_mock_transformer_model(3 * 224 * 224)

def create_beit_model():
    try:
        import timm
        return timm.create_model('beit_base_patch16_224', pretrained=False)
    except:
        return create_mock_transformer_model(3 * 224 * 224)

def create_regnet_model():
    try:
        import timm
        return timm.create_model('regnetx_008', pretrained=False)
    except:
        return create_mock_transformer_model(3 * 224 * 224)

def create_bert_model():
    if TORCHBENCH_AVAILABLE:
        try:
            from torchbenchmark.models.BERT_pytorch import Model
            return Model(test="eval").model
        except:
            pass
    try:
        from transformers import BertModel, BertConfig
        config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
        return BertModel(config)
    except:
        return create_mock_transformer_model(1, 768, 12, 12)

def create_gpt2_model():
    try:
        from transformers import GPT2Model, GPT2Config
        config = GPT2Config(vocab_size=50257, n_embd=768, n_layer=12, n_head=12)
        return GPT2Model(config)
    except:
        return create_mock_transformer_model(1, 768, 12, 12)

def create_distilbert_model():
    try:
        from transformers import DistilBertModel, DistilBertConfig
        config = DistilBertConfig(vocab_size=30522, dim=768, n_layers=6, n_heads=12)
        return DistilBertModel(config)
    except:
        return create_mock_transformer_model(1, 768, 6, 12)

def create_albert_model():
    try:
        from transformers import AlbertModel, AlbertConfig
        config = AlbertConfig(vocab_size=30000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
        return AlbertModel(config)
    except:
        return create_mock_transformer_model(1, 768, 12, 12)

def create_t5_model():
    try:
        from transformers import T5EncoderModel, T5Config
        config = T5Config(vocab_size=32128, d_model=512, num_layers=6, num_heads=8)
        return T5EncoderModel(config)
    except:
        return create_mock_transformer_model(1, 512, 6, 8)

def create_wav2vec2_model():
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config()
        return Wav2Vec2Model(config)
    except:
        return create_mock_transformer_model(16000, 768, 12, 12)

def create_hubert_model():
    try:
        from transformers import HubertModel, HubertConfig
        config = HubertConfig()
        return HubertModel(config)
    except:
        return create_mock_transformer_model(16000, 768, 12, 12)

def create_clip_model():
    try:
        from transformers import CLIPModel, CLIPConfig
        config = CLIPConfig()
        return CLIPModel(config)
    except:
        # Return a mock that handles both vision and text
        class MockCLIP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = create_mock_transformer_model(3 * 224 * 224)
                self.text_model = create_mock_transformer_model(1)
                
            def forward(self, input_ids=None, pixel_values=None, **kwargs):
                if pixel_values is not None:
                    return self.vision_model(pixel_values)
                elif input_ids is not None:
                    return self.text_model(input_ids)
                else:
                    # Return dummy output
                    return torch.randn(1, 512)
        return MockCLIP()

def create_dummy_input(model_type: str, input_shape, batch_size: int):
    """Create appropriate dummy input for different model types"""
    if model_type == 'vision':
        return torch.randn(batch_size, *input_shape, device=device, dtype=torch.float32)
    
    elif model_type == 'nlp':
        seq_len = input_shape[0]
        # Create random token indices for NLP models
        return torch.randint(0, 30000, (batch_size, seq_len), device=device, dtype=torch.long)
    
    elif model_type == 'speech':
        audio_len = input_shape[0]
        return torch.randn(batch_size, audio_len, device=device, dtype=torch.float32)
    
    elif model_type == 'multimodal':
        # For CLIP, create both vision and text inputs
        vision_shape = input_shape['vision']
        text_shape = input_shape['text']
        return {
            'pixel_values': torch.randn(batch_size, *vision_shape, device=device, dtype=torch.float32),
            'input_ids': torch.randint(0, 30000, (batch_size, *text_shape), device=device, dtype=torch.long)
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def model_forward(model, dummy_input, model_type: str):
    """Perform forward pass based on model type"""
    if model_type == 'multimodal':
        # Handle multimodal inputs (like CLIP)
        return model(**dummy_input)
    else:
        return model(dummy_input)

def measure_compile_time(model, dummy_input, model_type: str, compile_mode='default'):
    """Measure compilation time for a model with fallback options"""
    
    compile_modes = [compile_mode, 'reduce-overhead', 'max-autotune', 'default']
    
    for mode in compile_modes:
        try:
            print(f"      Trying compile mode: {mode}")
            start_time = time.perf_counter()
            
            if mode == 'default':
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False,
                    disable=False
                )
            elif mode == 'reduce-overhead':
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False,
                    dynamic=False
                )
            else:  # max-autotune
                compiled_model = torch.compile(
                    model, 
                    mode=mode, 
                    fullgraph=False, 
                    dynamic=False
                )
            
            # First forward pass triggers compilation
            with torch.no_grad():
                _ = model_forward(compiled_model, dummy_input, model_type)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            compile_time = time.perf_counter() - start_time
            print(f"      Success with mode: {mode}")
            return compiled_model, compile_time, mode
            
        except Exception as e:
            error_msg = str(e)
            print(f"      Failed with mode {mode}: {error_msg[:100]}...")
            
            if 'compiled_model' in locals():
                del compiled_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
    
    # If all compilation modes fail, try TorchScript as fallback
    try:
        print(f"      Trying TorchScript fallback...")
        start_time = time.perf_counter()
        
        # TorchScript compilation
        model.eval()
        if model_type == 'multimodal':
            # For multimodal, trace with vision input only for simplicity
            sample_input = dummy_input['pixel_values']
            compiled_model = torch.jit.trace(model.vision_model, sample_input)
        else:
            compiled_model = torch.jit.trace(model, dummy_input)
        
        compiled_model = torch.jit.optimize_for_inference(compiled_model)
        
        with torch.no_grad():
            if model_type == 'multimodal':
                _ = compiled_model(dummy_input['pixel_values'])
            else:
                _ = compiled_model(dummy_input)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        compile_time = time.perf_counter() - start_time
        print(f"      Success with TorchScript fallback")
        return compiled_model, compile_time, "torchscript_fallback"
        
    except Exception as e:
        print(f"      TorchScript fallback failed: {str(e)[:50]}...")
        if 'compiled_model' in locals():
            del compiled_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # If all compilation modes and fallbacks fail, return None
    print(f"      All compilation modes and fallbacks failed")
    return None, None, None

def measure_inference_time(model, dummy_input, model_type: str, num_runs=NUM_TIMING_RUNS, is_eager=False):
    """Measure inference time for a model"""
    model.eval()
    
    # Ensure eager mode for baseline measurements
    if is_eager:
        torch._dynamo.reset()
        
    # Warmup runs
    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model_forward(model, dummy_input, model_type)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Timing runs using perf_counter for better precision
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model_forward(model, dummy_input, model_type)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start_time)
    
    return np.mean(times), np.std(times)

def measure_guard_time(compiled_model, dummy_input, model_type: str, input_shape, batch_size: int):
    """Measure guard checking overhead and recompilation time"""
    guard_metrics = {
        'guard_check_time': 0.0,
        'guard_check_std': 0.0,
        'recompile_time': 0.0,
        'recompile_count': 0,
        'guard_failure_rate': 0.0
    }
    
    try:
        compiled_model.eval()
        
        # 1. Measure guard checking time with same input shape
        print(f"      Measuring guard checking time...")
        guard_times = []
        
        with torch.no_grad():
            # First run to ensure compilation is complete
            _ = model_forward(compiled_model, dummy_input, model_type)
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Measure guard checking overhead
            for _ in range(NUM_GUARD_RUNS):
                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model_forward(compiled_model, dummy_input, model_type)
                if device == "cuda":
                    torch.cuda.synchronize()
                guard_times.append(time.perf_counter() - start_time)
        
        guard_metrics['guard_check_time'] = np.mean(guard_times)
        guard_metrics['guard_check_std'] = np.std(guard_times)
        
        # 2. Measure recompilation time by changing input shapes
        print(f"      Measuring recompilation time...")
        recompile_times = []
        recompile_count = 0
        
        # Test different shapes that should trigger recompilation
        test_shapes = []
        if model_type == 'vision':
            # Try different image sizes
            if batch_size > 1:
                test_shapes.append((batch_size // 2, *input_shape))
            if batch_size < 32:
                test_shapes.append((batch_size * 2, *input_shape))
        elif model_type == 'nlp':
            # Try different sequence lengths
            seq_len = input_shape[0]
            if seq_len > 128:
                test_shapes.append((batch_size, seq_len // 2))
            if seq_len < 1024:
                test_shapes.append((batch_size, seq_len * 2))
        elif model_type == 'speech':
            # Try different audio lengths
            audio_len = input_shape[0]
            if audio_len > 8000:
                test_shapes.append((batch_size, audio_len // 2))
            if audio_len < 32000:
                test_shapes.append((batch_size, audio_len * 2))
        
        for test_shape in test_shapes[:2]:  # Limit to 2 tests to avoid too much overhead
            try:
                # Create input with different shape
                test_input = create_dummy_input(model_type, test_shape[1:], test_shape[0])
                
                with torch.no_grad():
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    _ = model_forward(compiled_model, test_input, model_type)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    recompile_time = time.perf_counter() - start_time
                    
                    recompile_times.append(recompile_time)
                    recompile_count += 1
                    
                    # Clean up
                    del test_input
                    
            except Exception as e:
                print(f"        Recompilation test failed: {str(e)[:50]}...")
                continue
        
        if recompile_times:
            guard_metrics['recompile_time'] = np.mean(recompile_times)
            guard_metrics['recompile_count'] = recompile_count
            guard_metrics['guard_failure_rate'] = recompile_count / len(test_shapes) if test_shapes else 0
        
    except Exception as e:
        print(f"      Guard measurement failed: {str(e)[:50]}...")
    
    return guard_metrics

def test_model_performance():
    """Test all models with different batch sizes"""
    results = defaultdict(list)
    successful_tests = 0
    total_tests = 0
    
    # Count total tests
    for model_type, config in MODEL_CONFIGS.items():
        total_tests += len(config['models']) * len(BATCH_SIZES)
    
    print(f"Total planned tests: {total_tests}")
    
    test_num = 0
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} models...")
        print(f"{'='*60}")
        
        input_shape = config['input_shape']
        models = config['models']
        
        for model_name, model_info in models.items():
            print(f"\nTesting {model_name}...")
            print(f"Description: {model_info['description']}")
            
            for batch_size in BATCH_SIZES:
                test_num += 1
                print(f"  Test {test_num}/{total_tests} - Batch size: {batch_size}")
                
                baseline_model = None
                compiled_model = None
                dummy_input = None
                
                try:
                    # Create dummy input
                    dummy_input = create_dummy_input(model_type, input_shape, batch_size)
                    
                    # Load baseline model (eager mode)
                    print(f"    Loading baseline model...")
                    baseline_model = model_info['create_fn']()
                    baseline_model = baseline_model.to(device)
                    baseline_model.eval()
                    
                    # Ensure we're in eager mode for baseline
                    torch._dynamo.reset()
                    
                    # Measure baseline (eager mode) inference time
                    print(f"    Measuring baseline (eager) performance...")
                    baseline_time, baseline_std = measure_inference_time(
                        baseline_model, dummy_input, model_type, is_eager=True
                    )
                    
                    # Create a fresh model copy for compilation
                    print(f"    Loading model for compilation...")
                    compiled_model_base = model_info['create_fn']()
                    compiled_model_base = compiled_model_base.to(device)
                    compiled_model_base.eval()
                    
                    # Measure compile time and get compiled model
                    print(f"    Compiling model...")
                    compiled_model, compile_time, compile_mode = measure_compile_time(
                        compiled_model_base, dummy_input, model_type
                    )
                    
                    if compiled_model is None:
                        # If compilation failed, record results with failed compilation
                        print(f"    Compilation failed - using baseline times")
                        compiled_time = baseline_time
                        compiled_std = baseline_std
                        compile_time = 0.0
                        compile_mode = "failed"
                        speedup = 1.0
                        # Initialize guard metrics for failed compilation
                        guard_metrics = {
                            'guard_check_time': 0.0,
                            'guard_check_std': 0.0,
                            'recompile_time': 0.0,
                            'recompile_count': 0,
                            'guard_failure_rate': 0.0
                        }
                    else:
                        # Measure compiled inference time
                        print(f"    Measuring compiled performance...")
                        compiled_time, compiled_std = measure_inference_time(
                            compiled_model, dummy_input, model_type
                        )
                        speedup = baseline_time / compiled_time if compiled_time > 0 else 1.0
                        
                        # Measure guard overhead and recompilation time
                        print(f"    Measuring guard overhead...")
                        guard_metrics = measure_guard_time(
                            compiled_model, dummy_input, model_type, input_shape, batch_size
                        )
                    
                    # Calculate throughput (samples/second)
                    baseline_throughput = batch_size / baseline_time if baseline_time > 0 else 0
                    compiled_throughput = batch_size / compiled_time if compiled_time > 0 else 0
                    
                    # Store results
                    results['model_type'].append(model_type)
                    results['model_name'].append(model_name)
                    results['batch_size'].append(batch_size)
                    results['compile_time'].append(compile_time)
                    results['compile_mode'].append(compile_mode)
                    results['baseline_time'].append(baseline_time)
                    results['baseline_std'].append(baseline_std)
                    results['compiled_time'].append(compiled_time)
                    results['compiled_std'].append(compiled_std)
                    results['speedup'].append(speedup)
                    results['baseline_throughput'].append(baseline_throughput)
                    results['compiled_throughput'].append(compiled_throughput)
                    results['description'].append(model_info['description'])
                    
                    # Store guard metrics
                    results['guard_check_time'].append(guard_metrics['guard_check_time'])
                    results['guard_check_std'].append(guard_metrics['guard_check_std'])
                    results['recompile_time'].append(guard_metrics['recompile_time'])
                    results['recompile_count'].append(guard_metrics['recompile_count'])
                    results['guard_failure_rate'].append(guard_metrics['guard_failure_rate'])
                    
                    print(f"    ✓ Results:")
                    print(f"      Compile time: {compile_time:.3f}s (mode: {compile_mode})")
                    print(f"      Baseline (eager): {baseline_time:.3f}±{baseline_std:.3f}s")
                    print(f"      Compiled: {compiled_time:.3f}±{compiled_std:.3f}s")
                    print(f"      Speedup: {speedup:.2f}x")
                    print(f"      Throughput - Baseline: {baseline_throughput:.1f} samples/s, Compiled: {compiled_throughput:.1f} samples/s")
                    print(f"      Guard check time: {guard_metrics['guard_check_time']:.4f}±{guard_metrics['guard_check_std']:.4f}s")
                    print(f"      Recompile time: {guard_metrics['recompile_time']:.3f}s (failures: {guard_metrics['recompile_count']})")
                    
                    successful_tests += 1
                    
                except Exception as e:
                    print(f"    ✗ Error with {model_name} batch_size={batch_size}: {str(e)[:100]}...")
                    # Still record a failed result to maintain the full matrix
                    results['model_type'].append(model_type)
                    results['model_name'].append(model_name)
                    results['batch_size'].append(batch_size)
                    results['compile_time'].append(0.0)
                    results['compile_mode'].append("error")
                    results['baseline_time'].append(0.0)
                    results['baseline_std'].append(0.0)
                    results['compiled_time'].append(0.0)
                    results['compiled_std'].append(0.0)
                    results['speedup'].append(0.0)
                    results['baseline_throughput'].append(0.0)
                    results['compiled_throughput'].append(0.0)
                    results['description'].append(model_info.get('description', 'N/A'))
                    
                    # Store empty guard metrics for errors
                    results['guard_check_time'].append(0.0)
                    results['guard_check_std'].append(0.0)
                    results['recompile_time'].append(0.0)
                    results['recompile_count'].append(0)
                    results['guard_failure_rate'].append(0.0)
                    
                finally:
                    # Clean up memory thoroughly
                    if dummy_input is not None:
                        del dummy_input
                    if baseline_model is not None:
                        del baseline_model
                    if compiled_model is not None:
                        del compiled_model
                    if 'compiled_model_base' in locals():
                        del compiled_model_base
                        
                    # Reset dynamo state
                    torch._dynamo.reset()
                    
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
    
    print(f"\nCompleted: {successful_tests}/{total_tests} tests successful")
    
    # Create DataFrame and verify results
    df = pd.DataFrame(results)
    if not df.empty:
        print(f"Results matrix: {len(df)} total results")
        print(f"Model types tested: {sorted(df['model_type'].unique())}")
        print(f"Models tested: {sorted(df['model_name'].unique())}")
        print(f"Batch sizes tested: {sorted(df['batch_size'].unique())}")
        
        # Check for missing combinations
        if len(df) == total_tests:
            print(f"✓ Complete matrix: all {total_tests} combinations tested")
        else:
            print(f"⚠ Incomplete matrix: {len(df)}/{total_tests} combinations")
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations for all model types"""
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Filter out error cases for visualization
    df_viz = df[df['baseline_time'] > 0].copy()
    
    if df_viz.empty:
        print("No valid data to visualize")
        return
    
    # Create focused bar charts for timing metrics
    create_timing_bar_charts(df_viz)

def create_timing_bar_charts(df):
    """Create focused bar charts for timing metrics with red for underperformance"""
    
    # Get average metrics per model (across all batch sizes)
    model_avg = df.groupby(['model_type', 'model_name']).agg({
        'baseline_time': 'mean',
        'compiled_time': 'mean',
        'guard_check_time': 'mean',
        'compile_time': 'first',  # Compile time is same across batch sizes
        'speedup': 'mean'
    }).reset_index()
    
    # Create combined model labels
    model_avg['model_label'] = model_avg['model_type'] + '_' + model_avg['model_name']
    model_avg = model_avg.sort_values('speedup', ascending=False)  # Sort by speedup
    
    # Create separate charts
    create_speedup_chart(model_avg)
    create_inference_time_chart(model_avg)
    create_guard_time_chart(model_avg)
    create_compile_time_chart(model_avg)

def create_speedup_chart(model_avg):
    """Create speedup bar chart with red for underperformance"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color bars: red for speedup < 1.0, green for speedup >= 1.0
    colors = ['red' if speedup < 1.0 else 'green' for speedup in model_avg['speedup']]
    
    bars = ax.bar(range(len(model_avg)), model_avg['speedup'], 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Speedup: Compiled vs Eager Mode (Red = Underperformance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Speedup (x times faster)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(range(len(model_avg)))
    ax.set_xticklabels(model_avg['model_label'], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at y=1.0 for reference
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, model_avg['speedup']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{value:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("Speedup chart saved: speedup_comparison.png")
    plt.close()

def create_inference_time_chart(model_avg):
    """Create inference time comparison chart"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = np.arange(len(model_avg))
    width = 0.35
    
    # Eager mode bars
    bars1 = ax.bar(x_pos - width/2, model_avg['baseline_time'] * 1000, width, 
                   label='Eager Mode', color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Compiled mode bars - red if slower, green if faster
    compiled_colors = ['red' if speedup < 1.0 else 'lightgreen' for speedup in model_avg['speedup']]
    bars2 = ax.bar(x_pos + width/2, model_avg['compiled_time'] * 1000, width, 
                   label='Compiled Mode', color=compiled_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Inference Time: Eager vs Compiled Mode (Red = Slower)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Inference Time (ms)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_avg['model_label'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('inference_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Inference time chart saved: inference_time_comparison.png")
    plt.close()

def create_guard_time_chart(model_avg):
    """Create guard time chart"""
    guard_data = model_avg[model_avg['guard_check_time'] > 0]
    
    if guard_data.empty:
        print("No guard time data available for chart")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars = ax.bar(range(len(guard_data)), guard_data['guard_check_time'] * 1000, 
                  color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Guard Check Time per Inference', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Guard Check Time (ms)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(range(len(guard_data)))
    ax.set_xticklabels(guard_data['model_label'], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, guard_data['guard_check_time'] * 1000):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(guard_data['guard_check_time'] * 1000) * 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('guard_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Guard time chart saved: guard_time_comparison.png")
    plt.close()

def create_compile_time_chart(model_avg):
    """Create compilation time chart"""
    compile_data = model_avg[model_avg['compile_time'] > 0]
    
    if compile_data.empty:
        print("No compilation time data available for chart")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars = ax.bar(range(len(compile_data)), compile_data['compile_time'], 
                  color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Model Compilation Time', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Compile Time (seconds)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(range(len(compile_data)))
    ax.set_xticklabels(compile_data['model_label'], rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, compile_data['compile_time']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(compile_data['compile_time']) * 0.01, 
                f'{value:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('compile_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Compile time chart saved: compile_time_comparison.png")
    plt.close()

def print_summary_statistics(df):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("TORCHBENCH ATTENTION MODELS - SUMMARY STATISTICS")
    print("="*80)
    
    if df.empty:
        print("No successful tests to analyze")
        return
    
    # Filter out error cases for statistics
    df_valid = df[df['baseline_time'] > 0].copy()
    
    if df_valid.empty:
        print("No valid test results to analyze")
        return
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total tests planned: {len(df)}")
    print(f"Successful tests: {len(df_valid)}")
    print(f"Success rate: {len(df_valid)/len(df)*100:.1f}%")
    
    if 'compile_mode' in df_valid.columns:
        successful_compiles = df_valid[df_valid['compile_mode'] != 'failed']
        print(f"Successful compilations: {len(successful_compiles)}/{len(df_valid)} ({len(successful_compiles)/len(df_valid)*100:.1f}%)")
        if not successful_compiles.empty:
            print(f"Average compile time: {successful_compiles['compile_time'].mean():.2f}±{successful_compiles['compile_time'].std():.2f}s")
    
    print(f"Average speedup: {df_valid['speedup'].mean():.2f}x")
    print(f"Best speedup: {df_valid['speedup'].max():.2f}x")
    print(f"Tests with >1.5x speedup: {(df_valid['speedup'] > 1.5).sum()}/{len(df_valid)} cases ({(df_valid['speedup'] > 1.5).sum()/len(df_valid)*100:.1f}%)")
    
    # Guard time statistics
    guard_data = df_valid[df_valid['guard_check_time'] > 0]
    if not guard_data.empty:
        print(f"\nGuard Time Statistics:")
        print(f"Models with guard data: {len(guard_data)}/{len(df_valid)}")
        print(f"Average guard check time: {guard_data['guard_check_time'].mean()*1000:.3f}ms")
        print(f"Max guard check time: {guard_data['guard_check_time'].max()*1000:.3f}ms")
        print(f"Average recompile time: {guard_data['recompile_time'].mean():.3f}s")
        print(f"Average recompile count: {guard_data['recompile_count'].mean():.1f}")
    
    # By model type statistics
    print(f"\nBy Model Type:")
    type_stats = df_valid.groupby('model_type').agg({
        'speedup': ['count', 'mean', 'max'],
        'compile_time': 'mean',
        'baseline_throughput': 'mean',
        'compiled_throughput': 'mean',
        'guard_check_time': 'mean'
    }).round(4)
    print(type_stats)
    
    # By model statistics (top performers)
    print(f"\nTop 10 Models by Average Speedup:")
    model_stats = df_valid.groupby(['model_type', 'model_name']).agg({
        'speedup': ['mean', 'max'],
        'compile_time': 'first',
        'baseline_throughput': 'mean',
        'compiled_throughput': 'mean'
    }).round(2)
    
    top_models = model_stats.sort_values(('speedup', 'mean'), ascending=False).head(10)
    print(top_models)

if __name__ == "__main__":
    print("Starting TorchBench Attention Models Benchmark...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print(f"TorchBench available: {TORCHBENCH_AVAILABLE}")
    if not TORCHBENCH_AVAILABLE:
        print("⚠️  Using fallback implementations - install torchbenchmark for full functionality")
    
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup runs: {NUM_WARMUP_RUNS}, Timing runs: {NUM_TIMING_RUNS}")
    
    # Count and display models
    total_models = sum(len(config['models']) for config in MODEL_CONFIGS.values())
    total_tests = total_models * len(BATCH_SIZES)
    print(f"Model types: {list(MODEL_CONFIGS.keys())}")
    print(f"Total models: {total_models}")
    print(f"Expected total tests: {total_tests}")
    
    # Run the benchmark
    results_df = test_model_performance()
    
    if not results_df.empty:
        # Save results to CSV
        results_df.to_csv('torchbench_attention_results.csv', index=False)
        print(f"\nResults saved to: torchbench_attention_results.csv")
        
        # Print summary statistics
        print_summary_statistics(results_df)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_visualizations(results_df)
        print("\n📊 Charts Generated:")
        print("- Speedup comparison (red = underperformance): speedup_comparison.png")
        print("- Inference time comparison (red = slower): inference_time_comparison.png")
        print("- Guard check time: guard_time_comparison.png")
        print("- Compilation time: compile_time_comparison.png")
        
        print("\nBenchmark completed!")
    else:
        print("No results collected. All models failed to execute.") 