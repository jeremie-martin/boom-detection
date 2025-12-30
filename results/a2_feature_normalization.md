# A2: Feature Normalization Experiment

## Objective
Test whether normalizing features to zero mean and unit variance improves
neural network (CNN, LSTM) training and prediction performance.

## Configuration
- Seeds: [42, 43, 44]
- Models: CNN + HGB + LSTM (3-model pipeline)
- Combiner: std/s=15/t=0.70
- Dataset: 175 simulations

## Implementation
- Normalization statistics computed on training data only
- Same statistics applied during inference
- Only applies to CNN and LSTM (not HGB, which is tree-based)

## Results

| Configuration | MAE | MAE std | Coverage |
|---------------|-----|---------|----------|
| Baseline (no normalization) | 2.97 | 0.06 | 10.9% |
| With normalization | 3.02 | 0.52 | 15.0% |

## Analysis

Feature normalization has **negligible effect** on MAE (+0.05 frames).
This suggests the features are already reasonably scaled, or the
AdamW optimizer with weight decay is robust to feature scaling.

**Interesting observation:** Coverage increased from 10.9% to 15.0% with normalization.
This suggests models are producing more consistent predictions (lower disagreement)
when features are normalized, causing more samples to pass the acceptance threshold.
However, the MAE std also increased from 0.06 to 0.52, indicating higher variance
across seeds.

**Recommendation:** Not recommended - normalization adds complexity and increases
variance without meaningful improvement to MAE. The coverage increase is offset
by the higher variance and nearly identical MAE.
