# D2: Bias Correction Post-Processing

## Objective
Test if correcting for systematic early bias improves performance.
From B4, predictions are systematically early by ~3.4 frames.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Results

| Offset | MAE | Coverage |
|--------|-----|----------|
| 0 | 2.97 ± 0.05 | 10.9% |
| 1 | 3.19 ± 0.13 | 10.9% |
| 2 | 3.62 ± 0.29 | 10.9% |
| 3 | 4.17 ± 0.36 | 10.9% |
| 4 | 4.88 ± 0.46 | 10.9% |
| 5 | 5.71 ± 0.58 | 10.9% |

## Best Configuration

**Offset = 0**: MAE 2.97 ± 0.05

## Analysis

**Bias correction does not help.** Baseline (offset=0) is optimal.

Recommendation: Keep using predictions without bias correction.

### Why Bias Correction May/May Not Help

1. B4 showed bias on ALL predictions, but we filter with ThresholdCombiner
2. Accepted samples (high quality, low disagreement) may have different bias
3. The combiner already selects for predictions that are likely correct
