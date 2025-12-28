# Understanding the Boom Phenomenon

## What is the Boom?

The "boom" is **NOT** simply when pendulums diverge or spread out. It's a specific visual event:

**The boom is when two (or more) groups of pendulums that had separated CONVERGE back together at a single point, creating a "collision" effect.**

## Visual Timeline

| Stage | Frame | Description | Visual Signature |
|-------|-------|-------------|------------------|
| 1. Start | ~0 | All pendulums identical, move as one | Single line |
| 2. Early divergence | ~100-200 | Slight spread, cone shape | Thin wedge |
| 3. Separation | ~200-400 | Two distinct groups forming | Bowtie/hourglass |
| 4. Pre-boom | ~400-600 | Groups spread widely, separate regions | Pac-man / two arcs |
| 5. **BOOM** | Variable | Groups CONVERGE at a point | Bright collision point |
| 6. Post-boom | +few frames | Caustic patterns emerge | Sharp bright curves |
| 7. Explosion | +1 second | Full caustics, rays | Complex light patterns |
| 8. Chaos | Late | Uniform randomness | Diffuse circle |

## Key Insight: It's About Convergence, Not Divergence

Current features measure the wrong thing:
- **Variance/spread** - measures divergence (increases monotonically)
- **Angular coverage** - measures how spread out (not boom-specific)
- **Caustic metrics** - measures concentration (appears AFTER boom)

The boom signature is:
1. **Before**: Two distinct clusters moving apart (bimodal distribution)
2. **At boom**: Clusters converge at a single point (bimodality → unimodality)
3. **After**: Clusters pass through, caustics form (new patterns)

## Proposed Detection Features

### 1. Bimodality Index
Fit a 2-component model to tip positions/angles. Measure:
- Separation between cluster centers
- Within-cluster vs between-cluster variance
- Dip statistic for bimodality

### 2. Convergence Rate
Track distance between the two main clusters over time:
- Boom = maximum negative rate (fastest approach)
- Boom = when distance reaches minimum

### 3. Angular Peak Distance
Look at angular histogram of tip positions:
- Find the two largest peaks
- Measure angular distance between them
- Boom = when peaks merge (distance → 0)

### 4. Point Concentration Spike
At boom, many pendulums pass through the same small region:
- Measure maximum local density
- Boom = sudden spike in concentration

### 5. Velocity Alignment
Before boom, clusters approach each other:
- Cluster velocities point toward each other
- Boom = anti-aligned velocities + small distance

## Boom Quality Annotation

High quality boom:
- Clear separation into two distinct groups before
- Single well-defined convergence point
- High-velocity crossing (dramatic collision)
- Clean caustic patterns after

Low quality boom:
- Multiple groups (3+) or gradual separation
- Fuzzy convergence point
- Slow crossing
- Weak or no caustics

## Experimental Findings

### Direct Detection Attempts (Failed)

| Method | MAE | Notes |
|--------|-----|-------|
| Max convergence rate | 69.1 | Simple median-split clustering doesn't work |
| Min cluster distance | 275.4 | Clusters don't reliably separate/merge |
| Max tip concentration | 96.0 | Concentration spike not reliable indicator |
| Min spatial entropy | 205.0 | Doesn't correlate with boom |

**Why direct convergence detection failed:**
- Simple clustering (median-split by angle) doesn't capture the complex group dynamics
- The "two groups" may not always cleanly separate in angle space
- Multiple convergence events may occur before/after the actual boom
- The boom is partially perceptual - involves velocity, not just position

### What Works

| Method | MAE | Notes |
|--------|-----|-------|
| Variance threshold | 31.2 | Simple baseline |
| HistGBM classifier | 18.9 | Frame-level classification |
| **CNN** | **14.3** | Best - sequence model |

The ML models work because they learn complex combinations of many features.

**Note**: Phase 4 (changepoint detection + ensemble) was not fully evaluated because the ensemble methods with nested CV take 20+ minutes to run. The changepoint detectors alone performed worse than baseline (CUSUM MAE ~35, BOCPD similar).

## Boom Quality Prediction (Phase 5)

### Quality Distribution
- Range: 0.10 - 0.92
- Mean: 0.54, Std: 0.24
- High quality (≥0.5): 30 simulations (61%)
- Low quality (<0.5): 19 simulations (39%)

### Quality-Error Correlation
**Key Finding**: Quality strongly predicts frame detection error.

| Quality Group | Frame MAE | Within 10 |
|--------------|-----------|-----------|
| High (≥0.5)  | 11.2      | 67%       |
| Low (<0.5)   | 31.1      | 26%       |
| All          | 18.9      | 51%       |

Spearman correlation: -0.454 (p=0.001)

### Quality Prediction Models

| Model | MAE | Correlation | Notes |
|-------|-----|-------------|-------|
| Mean baseline | 0.228 | -0.329 | Just predict mean |
| Median baseline | 0.233 | -0.280 | Just predict median |
| Ridge regression | 0.211 | 0.309 | On sim-level features |
| HistGBM regression | 0.228 | -0.015 | Overfits |
| **Ridge boom-aware (w=50)** | **0.182** | **0.491** | Best - uses boom frame window |

Binary Classification (high/low quality):
- HistGBM: 59% accuracy, F1=0.74
- Logistic: 55% accuracy, F1=0.67

### Multi-Stage Pipeline Results

Tested pipeline: Quality filter → Frame detector (trained on high-quality)

| Approach | Overall MAE | High-Q MAE | Low-Q MAE |
|----------|-------------|------------|-----------|
| Baseline (single model) | 18.9 | 11.2 | 31.1 |
| Train on high-Q only | 22.5 | 10.5 | 41.5 |
| Conditional (oracle quality) | 18.7 | 10.5 | 31.6 |
| Conditional (predicted quality) | 20.5 | 12.4 | 33.3 |

**Conclusion**: Multi-stage pipeline doesn't improve overall performance because:
1. Quality prediction isn't accurate enough (59% classification accuracy)
2. Training on high-Q only hurts low-Q predictions significantly
3. The benefit on high-Q cases is offset by the loss on low-Q cases

### Best Models Summary

| Task | Best Model | Performance |
|------|------------|-------------|
| Frame Detection | CNN | MAE 17.1, Within 10: 55% |
| Frame Detection | HistGBM | MAE 18.9, Within 10: 51% |
| Quality Prediction | Ridge boom-aware | MAE 0.182, Corr: 0.491 |

## Next Steps

1. **Better clustering**: Use actual k-means or GMM instead of median-split
2. **Velocity features**: Track cluster centroid velocities, detect approach
3. **Temporal patterns**: Look for specific sequence signatures (separate→approach→merge)
4. **Ensemble**: Combine CNN + HistGBM + convergence detector
5. **Quality as confidence**: Use predicted quality as confidence score, not filter
6. **Weighted loss**: Train with higher weight on high-quality samples
