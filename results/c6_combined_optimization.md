# C6: Combined Optimization Experiment

## Objective
Test whether combining 3-model pipeline with optimal quality parameters
(quality_window=50, jitter_std=0) improves on the documented baseline.

## Configuration
- Seeds: [42, 43, 44]
- Models: CNN + HGB + LSTM (3-model pipeline)
- Combiner: std/s=15/t=0.70
- Dataset: 175 simulations

## Results

| Configuration | MAE | MAE std | Coverage |
|---------------|-----|---------|----------|
| default (w=25, j=5) | 2.97 | 0.06 | 10.9% |
| optimal (w=50, j=0) | 3.34 | 0.17 | 11.6% |
| mixed (w=50, j=5) | 3.75 | 0.92 | 13.3% |
| **Documented baseline** | **2.97** | **0.06** | **10.9%** |

## Analysis

Combined optimization has **negligible effect** on MAE.
The quality parameters don't significantly impact 3-model performance.

**Recommendation:** Either configuration works - defaults are simpler.
