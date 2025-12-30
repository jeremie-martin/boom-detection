# C2: Asymmetric Quality Window

## Objective
Test whether asymmetric windows around boom improve quality prediction.
Hypothesis: frames after the boom may be more informative.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- Model: HistGradientBoostingRegressor
- 5-fold cross-validation

## Results

| Config | Window (before/after) | MAE | RMSE | Spearman r |
|--------|----------------------|-----|------|------------|
| Narrow 10/10 | 10/10 | 0.1354 +/- 0.0004 | 0.1687 | 0.688 |
| Symmetric 50/50 | 50/50 | 0.1448 +/- 0.0026 | 0.1831 | 0.635 |
| Symmetric 25/25 | 25/25 | 0.1490 +/- 0.0017 | 0.1867 | 0.626 |
| Before-heavy 75/25 | 75/25 | 0.1506 +/- 0.0033 | 0.1921 | 0.591 |
| Before-heavy 85/15 | 85/15 | 0.1514 +/- 0.0019 | 0.1884 | 0.601 |
| After-heavy 15/85 | 15/85 | 0.1526 +/- 0.0022 | 0.1907 | 0.617 |
| After-heavy 25/75 | 25/75 | 0.1568 +/- 0.0021 | 0.1941 | 0.593 |
| Wide 100/100 | 100/100 | 0.1651 +/- 0.0015 | 0.2032 | 0.552 |

## Best Configuration

**Narrow 10/10**

## Analysis

**Narrow 10/10** improves quality prediction by 0.0093 MAE over the default symmetric 50/50 window.

**Recommendation:** Consider switching to this window configuration.

### Window Size Analysis

Narrower windows perform better than wider windows.
This suggests quality is determined close to the boom moment.

### Asymmetry Analysis

Before-heavy windows outperform after-heavy windows.
Frames **before** the boom are more informative for quality prediction.
