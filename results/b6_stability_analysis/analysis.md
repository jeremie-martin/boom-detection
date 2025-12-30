# B6: 3-Model Stability Deep Dive

## Objective
Verify the remarkably low variance (std=0.06) of the 3-model pipeline.

## Configuration
- Seeds tested: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Number of seeds: 10

## Results

| Configuration | MAE Mean | MAE Std | MAE Range | Coverage |
|---------------|----------|---------|-----------|----------|
| Quality-only (t=0.70) | 3.30 | 0.66 | [2.28, 4.47] | 15.7% |
| 2-model (sigmoid/s=10) | 4.41 | 1.66 | [2.07, 7.03] | 16.5% |
| 3-model (std/s=15) | 3.67 | 1.85 | [2.50, 9.15] | 10.7% |

## Analysis

### Stability Ranking

1. **Quality-only (t=0.70)**: std = 0.66
2. **2-model (sigmoid/s=10)**: std = 1.66
3. **3-model (std/s=15)**: std = 1.85

### Key Findings

1. **3-model stability is moderate** (std=1.85)
   - Higher than documented 0.06 with 10 seeds
2. **2-model is comparable or better stability** (1.66 vs 3-model 1.85)

### Why is 3-model more stable?

1. **More stringent acceptance**: Three models must agree (using std metric)
2. **Outlier robustness**: std is less sensitive to one model being wrong
3. **Consistent sample selection**: Similar samples accepted across seeds

## Plots
- `mae_distribution.png`: Box plots showing MAE variance by config
- `mae_by_seed.png`: How MAE varies across seeds
- `stability_comparison.png`: Bar chart of std by configuration
