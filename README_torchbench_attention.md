# TorchBench Attention Models Benchmark

A comprehensive benchmarking tool for testing PyTorch 2's `torch.compile()` performance on attention-based models from TorchBench, covering Vision, NLP, Speech, and Multimodal domains.

## 🎯 Models Tested

### Vision Models with Attention
- **DeiT** - Vision Transformer with distillation
- **Swin Transformer** - Hierarchical vision transformer
- **ConvNeXt** - CNN with attention-like mechanisms
- **BEiT** - BERT pre-training of image transformers
- **RegNetY** - CNN with SE (Squeeze-and-Excitation) blocks

### NLP Models with Attention
- **BERT** - Bidirectional transformer encoder
- **GPT-2** - Autoregressive transformer decoder
- **DistilBERT** - Distilled version of BERT
- **ALBERT** - A Lite BERT with parameter sharing
- **T5** - Text-to-Text Transfer Transformer

### Speech Models with Attention
- **Wav2Vec2** - Transformer-based speech representation
- **HuBERT** - Hidden-Unit BERT for speech

### Multimodal Models
- **CLIP** - Contrastive Language-Image Pre-training

## 📦 Installation

### Required Dependencies
```bash
pip install torch torchvision torchaudio
pip install matplotlib pandas numpy
```

### Optional Dependencies (for full functionality)
```bash
# TorchBench (for official model implementations)
pip install torchbenchmark

# Transformers (for NLP models)
pip install transformers

# Timm (for vision models)
pip install timm
```

**Note**: The script includes smart fallbacks and will work even without optional dependencies by using mock transformer implementations.

## 🚀 Usage

### Basic Usage
```bash
python test_torchbench_attention_models.py
```

### What It Tests
- **Batch sizes**: 1, 4, 8, 16, 32
- **Compilation modes**: default, reduce-overhead, max-autotune (with TorchScript fallback)
- **Metrics**: Compile time, baseline vs compiled inference time, speedup, throughput

### Expected Runtime
- **With GPU**: ~30-60 minutes (depending on available models)
- **With CPU**: ~2-4 hours (significantly slower, especially for larger models)

## 📊 Output Files

### Data Export
- `torchbench_attention_results.csv` - Complete benchmark results in CSV format

### Visualizations
- `torchbench_attention_vision.png` - Vision models performance charts
- `torchbench_attention_nlp.png` - NLP models performance charts  
- `torchbench_attention_speech.png` - Speech models performance charts
- `torchbench_attention_multimodal.png` - Multimodal models performance charts
- `torchbench_attention_summary.png` - Overall performance summary

### Charts Include
1. **Speedup vs Batch Size** - How compilation speedup varies with batch size
2. **Inference Time Comparison** - Baseline vs compiled inference times
3. **Compilation Time** - Time taken for torch.compile() to complete
4. **Throughput Analysis** - Samples processed per second

## 📈 Interpreting Results

### Speedup Colors
- 🔴 **Red**: Slowdown (<1.0x) - Compilation made things worse
- 🟠 **Orange**: Minimal speedup (1.0-1.2x) - Small improvement
- 🟢 **Light Green**: Good speedup (1.2-1.5x) - Noticeable improvement  
- 🟢 **Dark Green**: Excellent speedup (>1.5x) - Significant improvement

### Key Metrics
- **Compile Time**: One-time cost of compilation (first run only)
- **Baseline Time**: Standard PyTorch eager mode performance
- **Compiled Time**: Performance after torch.compile() optimization
- **Speedup**: Ratio of baseline/compiled time (higher = better)
- **Throughput**: Samples processed per second (higher = better)

## 🔧 Customization

### Modify Batch Sizes
```python
BATCH_SIZES = [1, 2, 4, 8]  # Customize as needed
```

### Adjust Timing Precision
```python
NUM_WARMUP_RUNS = 20   # Reduce for faster testing
NUM_TIMING_RUNS = 20   # Reduce for faster testing
```

### Add Custom Models
Add to the `MODEL_CONFIGS` dictionary:
```python
'vision': {
    'models': {
        'my_model': {
            'create_fn': lambda: create_my_model(),
            'description': 'My custom vision transformer'
        }
    }
}
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch sizes: `BATCH_SIZES = [1, 2, 4]`
- Use smaller models or fewer models

**2. Import Errors**
- The script will use fallback implementations if optional packages aren't available
- Check which dependencies you need for specific models

**3. Compilation Failures**
- The script tries multiple compilation modes and fallbacks
- Check the console output to see which compilation modes succeeded

**4. Slow Performance**
- Reduce `NUM_WARMUP_RUNS` and `NUM_TIMING_RUNS` for faster testing
- Use GPU for significantly better performance

### Expected Behavior
- Some models may fail to compile - this is normal and expected
- The script will continue with other models and provide comprehensive statistics
- Fallback implementations ensure the benchmark can run even without all dependencies

## 📋 Sample Output

```
Starting TorchBench Attention Models Benchmark...
PyTorch version: 2.1.0
Device: cuda
TorchBench available: True
Total models: 13
Expected total tests: 65

Testing VISION models...
Testing deit_base...
  Test 1/65 - Batch size: 1
    ✓ Results:
      Compile time: 12.456s (mode: default)
      Baseline (eager): 0.023±0.001s
      Compiled: 0.015±0.001s
      Speedup: 1.53x
      Throughput - Baseline: 43.5 samples/s, Compiled: 66.7 samples/s
```

## 📚 References

- [PyTorch 2.0 torch.compile()](https://pytorch.org/docs/stable/torch.compile.html)
- [TorchBench](https://github.com/pytorch/benchmark)
- [Transformers Library](https://huggingface.co/transformers/)
- [Timm Library](https://github.com/rwightman/pytorch-image-models) 