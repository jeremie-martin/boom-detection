# C1: Quality Model Alternatives

## Objective
Compare different quality prediction models to find the best one.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- Quality window: 50 frames
- 5-fold cross-validation

## Results

| Model | MAE | RMSE | Spearman r |
|-------|-----|------|------------|
| RF (n=100, d=None) | 0.1432 +/- 0.0015 | 0.1788 | 0.649 |
| RF (n=100, d=7) | 0.1433 +/- 0.0016 | 0.1787 | 0.651 |
| RF (n=50, d=5) | 0.1446 +/- 0.0006 | 0.1806 | 0.639 |
| HistGBM (default) | 0.1448 +/- 0.0026 | 0.1831 | 0.635 |
| Ridge (alpha=1.0) | 0.2047 +/- 0.0071 | 0.2561 | 0.472 |

## Best Model

**RF (n=100, d=None)**

## Analysis

All models perform similarly. The default HistGBM is a good choice.

**Recommendation:** Keep the current default model.
