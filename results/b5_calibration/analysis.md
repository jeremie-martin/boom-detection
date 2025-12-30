# B5: Quality Calibration Verification

## Objective
Test whether isotonic regression calibration improves quality predictions.

## Configuration
- Seeds: [42, 43, 44]
- Models: CNN + HGB + LSTM (3-model pipeline)
- Combiner: std/s=15/t=0.70

## Results

| Configuration | ECE | Spearman r | Selective MAE | Coverage |
|---------------|-----|------------|---------------|----------|
| calibrated | 0.2192 | 0.557 | 2.98 | 10.9% |
| uncalibrated | 0.2206 | 0.564 | 3.53 | 10.1% |

## Comparison

- ECE reduction from calibration: 0.0014
- MAE change from calibration: +0.55 frames

## Analysis

**Calibration has minimal effect.** ECE difference is small.

**Recommendation:** Either setting works; calibration is optional.

## Expected Calibration Error (ECE)

ECE measures how well predicted probabilities match actual outcomes.
Lower is better. A perfectly calibrated model has ECE=0.

## Plots
- `reliability_diagrams.png`: Shows calibration quality
- `quality_scatter.png`: Predicted vs true quality
