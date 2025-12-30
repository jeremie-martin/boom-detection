# C3: LSTM + CNN Combination (Drop HGB)

## Objective
Test whether dropping HGB and using only neural networks (LSTM + CNN) is competitive.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation

## Results

| Configuration | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| 3-model (std/s=15/t=0.70) | 2.97 +/- 0.06 | 4.25 | 10.9% |
| LSTM+CNN (std/s=15/t=0.75) | 5.20 +/- 1.69 | 8.49 | 9.7% |
| LSTM+CNN (std/s=20) | 6.49 +/- 1.58 | 11.57 | 23.8% |
| LSTM+CNN (std/s=15) | 6.51 +/- 1.71 | 12.01 | 19.0% |
| LSTM+CNN (std/s=10) | 6.85 +/- 1.91 | 12.90 | 14.7% |
| LSTM+CNN (range/s=20) | 6.85 +/- 1.91 | 12.90 | 14.7% |
| LSTM+CNN (std/s=15/t=0.65) | 7.00 +/- 1.34 | 12.98 | 33.7% |
| LSTM+CNN (range/s=15) | 7.26 +/- 2.17 | 13.83 | 12.2% |
| LSTM+CNN (range/s=10) | 7.72 +/- 2.16 | 14.56 | 10.9% |

## Analysis

**Best LSTM+CNN configuration:** LSTM+CNN (std/s=15/t=0.75)
- MAE: 5.20
- Coverage: 9.7%

**LSTM+CNN underperforms 3-model baseline** by 2.23 frames.

**Recommendation:** Keep the 3-model pipeline.

### Metric Comparison (std vs range)

- `std` metric is better (5.20 vs 6.85)
