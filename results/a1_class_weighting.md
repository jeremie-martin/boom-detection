# A1: HGB Class Weighting Experiment

## Objective
Test whether adding class weights to HistGradientBoostingClassifier training
improves boom detection performance. Class imbalance exists because boom frame
divides each simulation into before/after sections of varying lengths.

## Configuration
- Seeds: [42, 43, 44]
- Models: CNN + HGB + LSTM (3-model pipeline)
- Combiner: std/s=15/t=0.70
- Dataset: 175 simulations

## Results

| Configuration | MAE | MAE std | Coverage |
|---------------|-----|---------|----------|
| Baseline (no weights) | 2.97 | 0.06 | 10.9% |
| With class weights | 3.79 | 1.45 | 10.5% |

## Analysis

Class weighting **hurts** performance by 0.82 frames.
The unweighted model actually performs better, possibly because
the majority class (before-boom frames) provides useful regularization.

**Recommendation:** Do not add class weighting.
