# B4: Temporal Error Bias Analysis

## Objective
Determine if predictions are systematically early or late.
Positive signed error = late prediction, negative = early.

## Configuration
- Seeds: [42, 43, 44]
- Total samples: 525 (5-fold CV x 3 seeds)

## Overall Bias

| Model | Mean Bias | Median Bias | t-statistic | p-value |
|-------|-----------|-------------|-------------|----------|
| CNN | -3.79 | -1.0 | -2.52 | 0.012** |
| HGB | -5.60 | -1.0 | -3.43 | 0.001** |
| LSTM | -1.39 | +0.0 | -0.94 | 0.349 |
| Final | -3.42 | +0.0 | -2.34 | 0.020** |

## Bias by Quality

| Quality | N | Final Mean | Final Median | CNN | HGB | LSTM |
|---------|---|------------|--------------|-----|-----|------|
| Low (0-0.5) | 204 | +0.79 | +0.0 | -1.58 | -1.62 | +3.62 |
| Med (0.5-0.7) | 162 | -11.92 | -3.0 | -10.04 | -14.86 | -9.67 |
| High (0.7+) | 159 | -0.18 | +1.0 | -0.26 | -1.28 | +0.60 |

## Accepted vs Rejected

- Accepted (n=57): mean=-0.11, median=+0.0
- Rejected (n=468): mean=-3.83, median=-1.0

## Analysis

**Significant temporal bias detected!**

Predictions are systematically early by 3.4 frames on average.

**Recommendation:** Consider implementing bias correction (D2) by:
1. Subtracting -3.4 frames from predictions
2. Or training models with shifted labels

### Quality-Dependent Bias

Bias is relatively consistent across quality levels.

## Plots
- `signed_error_histograms.png`: Distribution of signed errors per model
- `bias_by_quality.png`: Mean bias by quality stratum
- `signed_error_vs_quality.png`: Scatter of signed error vs quality
