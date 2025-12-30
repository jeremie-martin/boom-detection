# B2: Oracle Quality Study

## Objective
Determine the ceiling of quality-based selection by using true_quality
instead of predicted_quality. This shows how much better we could do
with perfect quality predictions.

## Configuration
- Seeds: [42, 43, 44]
- Total samples: 525 (5-fold CV x 3 seeds)
- Models: CNN + HGB + LSTM (3-model pipeline)
- Combiner: std/s=15/t=0.70

## Key Results

### Current vs Oracle (at threshold=0.70)

| Quality Source | MAE | Coverage |
|----------------|-----|----------|
| Predicted | 2.98 | 10.9% |
| Oracle | 1.62 | 1.5% |
| **Improvement** | **+1.36** | |

### Best MAE at Target Coverage Levels

| Target | Predicted MAE | Pred Cov | Oracle MAE | Oracle Cov | Room to Improve |
|--------|---------------|----------|------------|------------|----------------|
| ~10% | 2.98 | 10.9% | 3.25 | 9.9% | -0.27 |
| ~15% | 4.22 | 15.2% | 3.54 | 15.6% | 0.69 |
| ~20% | 4.42 | 20.4% | 3.68 | 19.6% | 0.74 |
| ~25% | 4.90 | 24.8% | 3.80 | 26.1% | 1.10 |
| ~30% | 5.01 | 31.0% | 3.83 | 29.7% | 1.19 |

### Quality-Gated Combiner Comparison

| Threshold | Pred MAE | Pred Cov | Oracle MAE | Oracle Cov |
|-----------|----------|----------|------------|------------|
| 0.60 | 6.10 | 42.1% | 9.30 | 50.3% |
| 0.65 | 4.41 | 24.8% | 6.29 | 40.6% |
| 0.70 | 3.71 | 17.5% | 4.70 | 30.3% |
| 0.75 | 2.29 | 2.7% | 3.73 | 19.4% |
| 0.80 | 1.50 | 0.4% | 3.92 | 9.7% |

## Analysis

**Moderate room for improvement (0.7 frames on average).**

Quality prediction is decent but not optimal. Some gains possible
from better quality models.

**Recommendations:**
- Quality model improvements may help marginally
- Focus more on model agreement signals

## Plots
- `risk_coverage_curves.png`: MAE vs coverage for predicted and oracle quality
- `oracle_improvement.png`: Room for improvement by coverage level
- `quality_prediction.png`: Scatter of predicted vs true quality
