# Focused Model Comparison Bar Charts

This directory contains scripts to generate **individual bar graphs for each model** comparing eager mode vs graph mode inference time and guard time analysis.

## Generated Charts

### 1. Inference Time Comparison
- Individual bar chart for each model
- Compares eager mode vs graph mode inference time
- Shows speedup/slowdown with color coding (green=faster, red=slower)
- Displays actual time values and speedup ratios

### 2. Guard Time Analysis  
- Individual analysis for each model
- Shows guard check time (in milliseconds)
- Shows recompilation time and failure rates
- Displays overhead as percentage of inference time

## Usage

### Option 1: Generate Sample Charts (Demo)
```bash
python focused_model_comparison.py
```
This creates demo charts in `./model_charts/` using simulated data.

### Option 2: Use Your Real Benchmark Data
```bash
# From CSV file
python generate_model_charts_from_data.py --data_file your_benchmark_results.csv

# From pickle file  
python generate_model_charts_from_data.py --data_file your_results.pkl

# With custom output directory and data summary
python generate_model_charts_from_data.py --data_file data.csv --output_dir ./real_charts --show_summary
```

## Required Data Format

Your data file must contain these columns (with any of these names):

**Required:**
- `model_name` or `model` - Model names
- `baseline_time` or `eager_time` - Eager mode inference time (seconds)  
- `compiled_time` or `graph_time` - Graph mode inference time (seconds)

**Optional (will use defaults if missing):**
- `guard_check_time` - Guard checking time (seconds)
- `recompile_time` - Recompilation time (seconds)
- `guard_failure_rate` - Guard failure rate (0.0-1.0)

### Example CSV Format:
```csv
model_name,baseline_time,compiled_time,guard_check_time,recompile_time,guard_failure_rate
ResNet50,0.045,0.032,0.0012,1.8,0.15
ViT-Base,0.089,0.056,0.0018,2.2,0.12
BERT-Base,0.067,0.041,0.0015,1.9,0.18
```

## Output

Each run generates:
- **Per-model inference comparison charts**: `{MODEL}_inference_comparison.png`
- **Per-model guard analysis charts**: `{MODEL}_guard_analysis.png`

## Chart Features

✅ **Individual charts per model** (not grouped)  
✅ **Bar graphs only** (no line plots or other visualizations)  
✅ **Eager vs Graph mode comparison**  
✅ **Guard time measurements**  
✅ **Color-coded performance indicators**  
✅ **High-resolution PNG outputs** (300 DPI)  
✅ **Professional styling with value labels**

## Files

- `focused_model_comparison.py` - Main chart generation with sample data
- `generate_model_charts_from_data.py` - Load real data and generate charts  
- `model_charts/` - Generated demo charts
- `model_charts_real/` - Generated charts from real data (default) 