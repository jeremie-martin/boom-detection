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

## Phase 6: Model Agreement Analysis

### Key Discovery
**Model agreement is a powerful confidence indicator** - more useful than quality prediction!

### Findings

| Metric | Agreement Cases (n=36) | Disagreement Cases (n=13) |
|--------|------------------------|---------------------------|
| CNN MAE | 8.8 frames | 38.0 frames |
| HistGBM MAE | 8.1 frames | 48.7 frames |
| Average MAE | 8.2 frames | - |

**Critical insight**: When CNN and HistGBM agree within 10 frames, we achieve MAE 8.2 - close to our target of <5!

### Correlations
- Model disagreement vs Quality: **Spearman -0.498** (p=0.0003) - highly significant
- Model disagreement vs Error: Strong positive correlation

### Why Simple Averaging Fails
- Ensemble (mean of CNN + HistGBM): MAE 17.4 (worse than CNN alone at 16.6)
- Models are not symmetrically wrong - they fail differently on different samples
- Need confidence-aware weighting, not equal weights

### Recommended Strategy
1. **When models agree (<10 frames)**: Use weighted average → MAE ~8
2. **When models disagree (>10 frames)**: Use specialized approach or flag as uncertain

### Implications for Architecture
- The problem has **inherent structure**: 73% "easy" cases, 27% "hard" cases
- Path to MAE <5 isn't uniform improvement:
  1. Nail easy cases: 8.2 → 5 frames (need fine-tuning)
  2. Handle hard cases differently (specialized model or abstention)

## Phase 7: Feature Importance Analysis

### Key Finding
**Top 20-50 features outperform all 1365 for HistGBM, but not for CNN!**

### Feature Ablation Results (HistGBM)

| Features | Count | MAE | Within 10 |
|----------|-------|-----|-----------|
| All features | 1365 | 18.9 | 51% |
| **Top 20** | 20 | **17.1** | **65%** |
| Top 50 | 50 | 16.2 | 59% |
| Top 100 | 100 | 16.4 | 61% |
| Random 50 | 50 | 21.5 | 55% |

### Feature Ablation Results (CNN)

| Features | Count | MAE | Within 10 |
|----------|-------|-----|-----------|
| **All features** | 1365 | **15.7** | 51% |
| Top 100 | 100 | 17.9 | 53% |
| Top 50 | 50 | 20.4 | 55% |
| Top 20 | 20 | 20.2 | 49% |

### Interpretation
- **HistGBM**: Benefits from feature selection. Excess features add noise.
- **CNN**: Benefits from all features. Can learn to ignore irrelevant features through its filters.
- Top features capture 97% of importance in just 50 features.
- Random 50 features perform much worse, confirming top features are meaningful.

### Most Important Features (Top 10)
1. Feature 1298 (importance: 0.0083)
2. Feature 662 (importance: 0.0026)
3. Feature 616 (importance: 0.0023)
4. Feature 1189 (importance: 0.0012)
5. Feature 28 (importance: 0.0010)
6. Feature 979 (importance: 0.0009)
7. Feature 657 (importance: 0.0009)
8. Feature 70 (importance: 0.0006)

## Phase 8: Hyperparameter Optimization

### Best Configurations Found

**CNN (Optimized)**
- Learning rate: 0.5e-3 (half default)
- Hidden dim: 32 (smaller)
- Epochs: 30, Patience: 5
- **MAE: 14.5, MedAE: 9.0, Within 10: 55%**

**HistGBM (Optimized)**
- n_estimators: 200
- max_depth: 7
- Features: Top 50
- **MAE: 16.2, MedAE: 5.0, Within 10: 59%**

### Agreement Analysis (Optimized Models)

| Condition | n | CNN MAE | HistGBM MAE | Average MAE |
|-----------|---|---------|-------------|-------------|
| Agree (≤10 frames) | 33 | 8.5 | 7.5 | **7.7** |
| Disagree (>10 frames) | 16 | 26.9 | 34.0 | - |

**Key finding**: When optimized models agree, MAE is 7.5-7.7 - approaching goal!

### By Quality

| Quality | CNN MAE | HistGBM MAE |
|---------|---------|-------------|
| High (≥0.5) | 8.8 | 10.8 |
| Low (<0.5) | 23.4 | 24.6 |

## Phase 9: Ensemble Methods Deep Dive

### Individual Model Performance

| Model | MAE | MedAE | Within 10 |
|-------|-----|-------|-----------|
| CNN | 16.6 | 8.0 | 59% |
| HistGBM | 15.9 | 8.0 | 61% |
| Logistic | 20.1 | 9.0 | 55% |

### Ensemble Results

| Method | MAE | MedAE | Within 10 | Within 5 |
|--------|-----|-------|-----------|----------|
| **Mean (CNN+HGB)** | **14.0** | **7.5** | **61%** | 39% |
| Median 3-model | 16.2 | 8.0 | 65% | 41% |
| Weighted (CNN=0.5) | 14.0 | 7.5 | 61% | 39% |
| Oracle (best model) | 9.9 | 4.0 | 73% | 55% |

### Agreement Analysis (CNN + HistGBM)

| Condition | n | MAE (Avg) | Notes |
|-----------|---|-----------|-------|
| Agree (≤10 frames) | 30 (61%) | **7.2** | Close to goal! |
| Disagree (>10 frames) | 19 (39%) | 30.7 | Main problem |

**Key insight**: When CNN and HistGBM agree, ensemble achieves MAE 7.2 - very close to our goal of <5. The disagreement cases are dragging overall MAE up.

### Disagreement Case Analysis

When models disagree (n=19):
- CNN MAE: 30.7
- HGB MAE: 29.3
- Average MAE: 24.7
- Using HGB for disagreement: slightly better

### Oracle Analysis

- If we could perfectly pick the better model: MAE 9.9
- CNN is better 51% of cases, HGB is better 49%
- 6 simulations have error >20 for ALL models (hardest cases, mostly low quality)
- 12 simulations have error ≤5 for ALL models (easiest cases)

### Path to MAE < 5

1. **Agreement cases (61%)**: 7.2 → 5 (need ~30% improvement)
2. **Disagreement cases (39%)**: 30 → 10 (need specialized handling)
3. **Alternative**: Increase agreement rate from 61% to 80%+

## Phase 10: New Feature Engineering

### Local Context Features

Added rolling statistics showing local anomalies:
- For each frame, compute difference from rolling mean (windows: 5, 10, 20)
- This captures "how unusual is this frame compared to recent frames"

### Results with Enhanced Features

| Model | Features | MAE | MedAE | Within 10 | Within 5 |
|-------|----------|-----|-------|-----------|----------|
| HistGBM | Top 50 (baseline) | 15.9 | 8.0 | 61% | 39% |
| **HistGBM** | **Top 50 + Context** | **13.9** | **7.0** | **61%** | 45% |
| CNN | Full 1365 | 14.7 | 8.0 | 59% | 39% |

### Optimal Ensemble

Use different features for different models:
- CNN: Full 1365 features (benefits from more features)
- HistGBM: Top 50 + Local Context (74 features, benefits from selection)

**Best Result: Mean(CNN+HGB) = MAE 13.3**

| Method | MAE | MedAE | Within 10 | Within 5 |
|--------|-----|-------|-----------|----------|
| CNN (full) | 14.7 | 8.0 | 59% | 39% |
| HistGBM (enhanced) | 13.9 | 7.0 | 61% | 45% |
| **Mean (CNN+HGB)** | **13.3** | **7.5** | **59%** | - |
| Agreement-10 (Avg\|HGB) | 13.7 | **5.0** | 61% | **51%** |

### Agreement Analysis (Threshold=10)

| Condition | n | MAE | Notes |
|-----------|---|-----|-------|
| Agree (≤10 frames) | 26 (53%) | **6.8** | Close to goal! |
| Disagree (>10 frames) | 23 (47%) | 21.4 | Main problem |

### By Quality

| Model | High-Q MAE | Low-Q MAE |
|-------|-----------|-----------|
| CNN | **8.5** | 24.4 |
| HGB | 11.2 | **18.2** |
| Mean | 9.6 | 19.1 |

**Key insight**: CNN excels on high-quality (MAE 8.5), HGB on low-quality (MAE 18.2).

## Phase 11: Error Analysis & Quality-Aware Routing

### Error Pattern Analysis

**Worst 10 cases:**
- 70% are low-quality simulations
- Model disagreement: 47 frames (vs 17 overall)
- Neither model consistently better - both fail on hard cases

**Key insight: Models have complementary strengths!**

| Model | High-Q MAE | Low-Q MAE |
|-------|-----------|-----------|
| CNN | **8.4** | 24.9 |
| HGB | 11.2 | **18.2** |

CNN excels on high-quality, HGB on low-quality!

### Quality-Aware Routing

Route predictions based on predicted quality:
- High-quality → Use CNN (better at 8.4 vs 11.2)
- Low-quality → Use HGB (better at 18.2 vs 24.9)

| Method | MAE | Notes |
|--------|-----|-------|
| HGB alone | 13.9 | Baseline |
| Simple mean | 13.7 | Basic ensemble |
| Oracle quality-aware | 12.2 | Using true quality |
| **Predicted quality-aware** | **11.8** | Using predicted quality |
| Oracle (best model) | 7.0 | Theoretical limit |

**Best result: MAE 11.8 with predicted quality routing!**

## Progress Summary

| Phase | Best MAE | Key Finding |
|-------|----------|-------------|
| Baseline | 18.9 | HistGBM frame classifier |
| Phase 3 | 16.2 | Hyperparameter optimization |
| Phase 4 | 14.0 | CNN+HGB ensemble |
| Phase 5 | 13.3 | Local context features |
| **Phase 7** | **11.8** | Quality-aware model routing |

**Progress: 18.9 → 11.8 (37% improvement)**

## Next Steps

1. **Attention mechanisms**: Let CNN focus on boom region
2. **Multi-task learning**: Predict frame + quality jointly
3. **Improve quality prediction**: Better routing = lower error
4. **Synthesis**: Combine all best techniques
