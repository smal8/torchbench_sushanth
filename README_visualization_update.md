# Updated Visualization System: Bar Charts for Eager vs Graph Mode Analysis

## Overview

The visualization system has been completely updated to focus exclusively on **bar charts** for comparing **eager mode vs graph mode inference** performance, with extensive **guard time analysis** as requested.

## Key Changes

### ✅ What's New
- **Only bar charts** - Removed all scatter plots, line charts, and boxplots
- **Eager vs Graph Mode focus** - All charts directly compare these two inference modes
- **Comprehensive guard time analysis** - Multiple dedicated charts for guard time metrics
- **Color-coded performance indicators** - Visual cues for speedups, slowdowns, and overhead levels
- **Automated chart generation** - Single function to create all charts at once

### ❌ What's Removed
- Density scatter plots
- Input/output visualizations  
- Boxplot error analysis
- Line charts for trends
- Other non-bar chart visualizations

## New Chart Types

### 1. Inference Time Comparison
- Side-by-side bars comparing eager vs graph mode
- Times displayed in milliseconds with value labels
- Red bars for slower graph mode, green for faster
- Speedup values displayed above each pair

### 2. Guard Time Analysis (4 Charts)
- **Guard Check Time**: Time spent checking guards per inference
- **Recompilation Time**: Time taken when guards fail and recompilation occurs
- **Guard Failure Rate**: Percentage of inferences that trigger recompilation
- **Guard Overhead**: Guard checking time as percentage of inference time

### 3. Speedup Analysis
- Color-coded bars showing performance improvement
- Red (<1.0x), Orange (1.0-1.2x), Light Green (1.2-1.5x), Dark Green (>1.5x)
- Horizontal line at 1.0x to indicate break-even point

### 4. Compilation Time Analysis
- Bar chart showing initial compilation time for each model
- Useful for understanding one-time setup costs

### 5. Batch Size Analysis (Per Model)
- Inference time comparison across different batch sizes
- Speedup trends as batch size increases

## Usage Examples

### Quick Start
```python
from visualization import save_all_bar_charts
import pandas as pd

# Your performance data
data = pd.DataFrame({
    'model_name': ['ResNet50', 'ViT-Base', 'BERT-Base'],
    'baseline_time': [0.045, 0.032, 0.078],     # Eager mode times (seconds)
    'compiled_time': [0.023, 0.018, 0.041],     # Graph mode times (seconds)
    'compile_time': [12.3, 8.7, 15.2],          # Compilation times (seconds)
    'guard_check_time': [0.0008, 0.0005, 0.0012], # Guard checking (seconds)
    'recompile_time': [1.2, 0.8, 1.5],          # Recompilation times (seconds)
    'guard_failure_rate': [0.15, 0.08, 0.22]    # Failure rate (0.15 = 15%)
})

# Generate all charts at once
save_all_bar_charts(data, output_dir="./performance_charts", prefix="my_models")
```

### Individual Charts
```python
from visualization import (
    create_inference_comparison_bar_chart,
    create_guard_time_analysis_charts,
    create_speedup_analysis_bar_chart
)

# Create specific charts
fig1 = create_inference_comparison_bar_chart(data, "My Performance Analysis")
fig1.savefig('inference_comparison.png', dpi=300, bbox_inches='tight')

guard_figures = create_guard_time_analysis_charts(data)
for i, fig in enumerate(guard_figures):
    fig.savefig(f'guard_analysis_{i}.png', dpi=300, bbox_inches='tight')
```

### Run the Demo
```bash
cd /home/htc/smalipati/ai4forest_sushanth/
python example_bar_charts.py
```

## Data Format Requirements

### Minimum Required Columns
- `model_name` (or `model`): Model identifier
- `baseline_time` (or `eager_time`): Eager mode inference time in seconds
- `compiled_time` (or `graph_time`): Graph mode inference time in seconds

### Additional Columns for Guard Analysis
- `guard_check_time`: Guard checking time in seconds
- `recompile_time`: Recompilation time in seconds
- `guard_failure_rate`: Failure rate as decimal (0.1 = 10%)
- `recompile_count`: Number of recompilations (integer)

### For Batch Size Analysis
- `batch_size`: Batch size (integer)

## Chart Features

### Visual Indicators
- **Performance Colors**:
  - 🔴 Red: Slowdown or high overhead
  - 🟠 Orange: Minimal improvement or medium overhead  
  - 🟡 Yellow: Moderate improvement or overhead
  - 🟢 Green: Good improvement or low overhead

### Value Labels
- All bars include precise value labels
- Speedup ratios displayed prominently
- Percentage values for rates and overhead

### Professional Styling
- High-resolution PNG output (300 DPI)
- Clean grid lines and typography
- Consistent color schemes across charts
- Proper legends and axis labels

## Integration with Existing Code

The updated `visualization.py` maintains backward compatibility:
- Legacy function names redirect to new bar chart functions
- Existing code will work but show bar charts instead of old chart types
- Warning messages inform about the chart type changes

## File Outputs

When using `save_all_bar_charts()`, the following files are created:
- `{prefix}_inference_comparison.png` - Main eager vs graph comparison
- `{prefix}_speedup_analysis.png` - Speedup analysis with color coding
- `{prefix}_compilation_time.png` - Compilation time analysis
- `{prefix}_guard_check_time.png` - Guard checking overhead
- `{prefix}_recompile_time.png` - Recompilation time analysis
- `{prefix}_guard_failure_rate.png` - Guard failure rate analysis
- `{prefix}_guard_overhead.png` - Guard overhead percentage
- `{prefix}_{model}_batch_inference.png` - Per-model batch analysis
- `{prefix}_{model}_batch_speedup.png` - Per-model batch speedup

## Benefits

1. **Clear Performance Comparison**: Direct visual comparison of eager vs graph mode
2. **Comprehensive Guard Analysis**: Deep dive into guard-related overhead
3. **Easy Interpretation**: Color coding makes performance immediately obvious
4. **Professional Quality**: High-resolution charts suitable for presentations
5. **Automated Generation**: Create all charts with a single function call
6. **Flexible Data Input**: Works with various data formats and column names

## Example Integration

To integrate with your existing PyTorch benchmarking:

```python
# After collecting your performance data
results_df = pd.DataFrame(your_benchmark_results)

# Generate comprehensive bar chart analysis
save_all_bar_charts(
    results_df, 
    output_dir="./eager_vs_graph_analysis", 
    prefix="pytorch_models"
)

print("Bar chart analysis complete!")
print("Check ./eager_vs_graph_analysis/ for all charts")
```

This provides a complete, focused analysis of eager vs graph mode performance with the guard time insights you requested! 