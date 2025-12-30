# B1: Error vs Ground Truth Quality Correlation

## Objective
Understand whether true quality predicts prediction error and whether
model disagreement adds information beyond quality.

## Data
- Seeds: [42, 43, 44]
- Total samples: 525 (5-fold CV x 3 seeds)
- Accepted: 57 (10.9%)
- Rejected: 468 (89.1%)

## Correlation Analysis

| Variable | Spearman r | p-value | Interpretation |
|----------|------------|---------|----------------|
| True Quality vs Error | -0.571 | 1.08e-46 | Strong negative| Disagreement vs Error | 0.463 | 3.26e-29 | Strong positive| Pred Quality vs Error | -0.612 | 2.35e-55 | Strong negative## Rejection Analysis

- Mean error for accepted samples: 2.98 frames
- Mean error for rejected samples: 20.44 frames
- **Rejection avoids 17.46 frames higher error**

## Key Insights

1. **True quality is strongly predictive of error** - high quality simulations have lower errors
2. **Model disagreement strongly predicts error** - valuable selection signal
3. **Rejection is effective** - rejected samples have higher errors on average

## Plots
- `error_vs_true_quality.png`: Scatter plot of error vs true quality
- `error_vs_disagreement.png`: Scatter plot of error vs model disagreement
- `quality_distribution.png`: Distribution of true quality for accepted/rejected
