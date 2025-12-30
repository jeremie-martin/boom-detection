# C7: CNN Learning Rate Sensitivity

## Objective
Test different learning rates for the CNN model to find optimal setting.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Results

| Learning Rate | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| lr=2e-03 | 2.37 ± 0.56 | 3.05 | 9.3% |
| lr=1e-03 | 2.46 ± 0.15 | 3.30 | 9.3% |
| lr=5e-04 | 2.97 ± 0.06 | 4.25 | 10.9% |
| lr=2e-04 | 3.48 ± 1.37 | 6.55 | 11.4% |
| lr=1e-04 | 4.30 ± 2.03 | 7.13 | 10.5% |

## Best Configuration

**lr=2e-03**

## Analysis

**lr=2e-03** improves over default (5e-4) by 0.61 frames.
**Recommendation:** Switch to lr=2e-03.

### Sensitivity Analysis

CNN is **highly sensitive** to learning rate (MAE range: 1.93).

### Comparison with Baseline

Baseline (std/s=15/t=0.70): MAE 2.97
Best with LR sweep: MAE 2.37

LR sweep found 0.60 frame improvement.
