# C4: Single Model Baselines with Quality Gating

## Objective
Establish individual model + quality-gating baselines.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation

## Results

| Configuration | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| LSTM-only (t=0.75) | 1.56 +/- 1.39 | 2.08 | 1.5% |
| HGB-only (t=0.75) | 1.62 +/- 0.88 | 2.19 | 1.7% |
| CNN-only (t=0.75) | 3.22 +/- 1.58 | 3.39 | 1.1% |
| CNN-only (t=0.7) | 4.09 +/- 0.38 | 5.72 | 14.9% |
| HGB-only (t=0.7) | 4.17 +/- 0.73 | 8.03 | 16.0% |
| HGB-only (t=0.65) | 4.19 +/- 0.56 | 7.54 | 22.9% |
| LSTM-only (t=0.7) | 4.74 +/- 1.04 | 6.23 | 14.5% |
| CNN-only (t=0.65) | 4.91 +/- 0.81 | 6.90 | 23.4% |
| LSTM-only (t=0.65) | 4.95 +/- 0.79 | 6.52 | 22.9% |
| LSTM-only (t=0.6) | 6.84 +/- 0.58 | 11.15 | 41.1% |
| HGB-only (t=0.6) | 7.53 +/- 0.39 | 17.19 | 41.7% |
| CNN-only (t=0.6) | 8.20 +/- 0.57 | 15.82 | 42.5% |

## Best per Model

- **CNN**: CNN-only (t=0.75) - MAE 3.22, Coverage 1.1%
- **HGB**: HGB-only (t=0.75) - MAE 1.62, Coverage 1.7%
- **LSTM**: LSTM-only (t=0.75) - MAE 1.56, Coverage 1.5%

## Analysis

**Best single model configuration:** LSTM-only (t=0.75)
- MAE: 1.56
- Coverage: 1.5%

**Comparison with 3-model baseline (MAE 2.97):**
- Single model is -1.41 frames worse

**Single model is competitive** - only -1.41 frames worse.
Consider single model for simplicity if MAE difference is acceptable.

### Model Ranking

1. CNN-only (t=0.7): MAE 4.09
2. HGB-only (t=0.7): MAE 4.17
3. LSTM-only (t=0.7): MAE 4.74
