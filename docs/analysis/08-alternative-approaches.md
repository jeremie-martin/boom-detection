# Alternative Approaches Exploration

**Agent Task**: Explore alternative framings and approaches for the boom-detection problem.

**Date**: 2025-12-28

---

## Executive Summary

Eight promising alternative approaches identified. The most promising are changepoint detection (2-3 hours to test), temporal clustering (2-3 hours), and uncertainty quantification (4-6 hours). These align with the physical nature of the boom event and could reduce the 51% rejection rate due to model disagreement.

---

## Current State Summary

- **Best result**: MAE 6.4 ± 0.5 frames on ~35% accepted simulations
- **Current approach**: CNN + HistGBM ensemble with agreement + predicted quality filters
- **Data**: ~96 simulations, 300-600 frames each, 183 engineered features
- **Key limitation**: 51% of rejections are due to model disagreement (even on high-quality sims)

---

## Alternative Approaches

### 1. Changepoint Detection via Density/Curvature Shifts

**Status**: Briefly attempted, needs deeper exploration

**Rationale**:
- Boom is a discrete event (cluster convergence), not continuous regression
- Could use statistical changepoint methods: PELT, Dynamic Programming, or Bayesian approaches
- Capture the *rate of change* in key features rather than absolute values

**Approach**:
```python
import ruptures as rpt

# Extract key features that spike at boom
features = [position_variance, velocity_magnitude, acceleration]

# Apply PELT algorithm (fast O(n log n))
algo = rpt.Pelt().fit(features)
breakpoints = algo.predict(pen=10)
```

**Why it could work**:
- Treats boom as what it is: a sharp discontinuity
- Doesn't require learning from limited examples; uses statistical principles
- Could be combined with learned models as confidence check

**Feasibility**: High (existing libraries: `ruptures`, `changefinder`)
**Expected improvement**: Could reduce disagreement cases from 30 to 20+ frames

---

### 2. Multi-Task Learning: Boom Frame + Cluster Divergence

**Status**: Not attempted

**Rationale**:
- Current models only learn boom frame (scalar output)
- Could jointly learn:
  - Task A: Boom frame prediction
  - Task B: Cluster divergence score (pre-boom) / convergence score (at boom)
  - Task C: Energy/momentum flux (physical signal of collision)

**Approach**:
```python
class MultiTaskBoomNet(nn.Module):
    def forward(self, x):
        shared_features = self.cnn_backbone(x)

        boom_logits = self.boom_head(shared_features)
        convergence = self.convergence_head(shared_features)
        energy_flux = self.energy_head(shared_features)

        return boom_logits, convergence, energy_flux
```

**Why it works**:
- Auxiliary tasks act as regularization (inductive bias)
- Convergence/energy scores could directly identify boom
- Better use of physical domain knowledge

**Feasibility**: Medium-High (need physical labels for aux tasks)
**Expected impact**: Could improve CNN from MAE 16 → 12 frames overall

---

### 3. Masked Language Model Pretraining (Self-Supervised)

**Status**: Not attempted

**Rationale**:
- 96 annotated sims, but presumably many unlabeled simulations available
- Self-supervised learning: mask random frames, predict features from neighbors
- Pre-trained model learns general boom dynamics without labels

**Approach**:
1. Train on all unlabeled simulations (if available):
   - Mask 15% of frames randomly
   - Predict masked features from context
   - Learn general temporal patterns
2. Fine-tune CNN on labeled data (96 sims)
3. Transfer learning should improve with small annotated set

**Why it works**:
- Reduces overfitting on 96 sims by pre-training on more data
- Learns boom-independent temporal structures
- Similar to BERT success in NLP

**Feasibility**: High (if unlabeled data exists)
**Expected impact**: Could improve MAE by 10-20% via better regularization

---

### 4. Ranking-Based Prediction (Ordinal Regression)

**Status**: Not attempted

**Rationale**:
- Current: Predict exact boom frame (regression)
- Alternative: Predict relative ordering: "frame 100 < frame 200 in boom likelihood"
- Frame with highest "boom score" wins

**Approach**:
```python
class OrdinalBoomModel(nn.Module):
    def forward(self, x):
        # Predict relative scores instead of absolute positions
        scores = self.cnn(x)  # (batch, frames)
        return scores

    def loss(self, scores, boom_frame):
        # Ranking loss: boom_frame should have highest score
        return -log(softmax(scores)[boom_frame])
```

**Why it works**:
- Ordinal regression is less sensitive to outliers
- Naturally handles ties (nearby frames are similar)
- Focuses on *relative* ranking, not absolute calibration

**Feasibility**: Medium
**Expected impact**: Might reduce outliers (>20 frame errors)

---

### 5. Likelihood-Based Interval Prediction (Uncertainty Quantification)

**Status**: Not attempted

**Rationale**:
- Instead of point estimate, predict: "boom is between frames X and Y"
- Models disagreement cases as overlapping intervals
- Could output confidence interval: [boom_frame ± sigma]

**Approach**:
```python
class UncertainBoomModel(nn.Module):
    def forward(self, x):
        shared = self.cnn_backbone(x)
        mean = self.mean_head(shared)      # Predicted boom frame
        log_std = self.std_head(shared)    # Log-uncertainty

        return mean, log_std

    def loss(self, mean, log_std, true_boom):
        # Negative log-likelihood
        var = torch.exp(2 * log_std)
        nll = ((true_boom - mean)**2 / var + log_std).mean()
        return nll
```

**Why it works**:
- Naturally models uncertainty
- Hard cases get wider intervals (flagged as uncertain)
- Can use interval width for rejection threshold

**Feasibility**: High (standard UQ technique)
**Expected impact**: Better calibration for rejection; could improve quality prediction

---

### 6. Temporal Clustering + Local Peak Detection

**Status**: Novel, not attempted

**Rationale**:
- Don't predict frame directly
- Instead: cluster frames into "regions" (pre-boom, boom, post-boom)
- Boom is the *boundary* between clusters

**Approach**:
```python
from sklearn.cluster import KMeans

# Step 1: Cluster frames
features_normalized = normalize_features(all_frames)
clusters = KMeans(n_clusters=3).fit(features_normalized).labels_

# Step 2: Find transition points (cluster boundary)
transitions = np.where(np.diff(clusters) != 0)[0]

# Step 3: Peak detection on "boundary sharpness"
sharpness = compute_sharpness(features, transitions)
boom_frame = transitions[np.argmax(sharpness)]
```

**Why it works**:
- Exploits natural structure (3-phase process: clusters, converge, explode)
- Robust to small frame errors
- Combines unsupervised + supervised signals

**Feasibility**: High (simple clustering + heuristics)
**Expected impact**: Could work for 60%+ of cases with minimal training

---

### 7. Physics-Informed Neural Networks (PINNs)

**Status**: Not attempted, research-level

**Rationale**:
- Boom has physical laws: energy conservation, momentum transfer
- Could encode conservation laws as loss constraints
- Model learns from both data AND physics

**Approach**:
```python
class PhysicsInformedBoomNet(nn.Module):
    def forward(self, t, features):
        u = self.cnn(features)  # Predict boom likelihood

        # Compute physical quantities
        positions = features[:, [X1, Y1, X2, Y2]]
        velocities = features[:, [W1, W2]]

        # Physical constraint: Energy conservation
        kinetic_energy = 0.5 * m * (velocities ** 2).sum()
        potential_energy = m * g * positions[:, 1]
        total_energy = kinetic_energy + potential_energy

        # Loss = prediction loss + physics constraint
        phys_loss = (total_energy - constant_energy).mean()

        return u, phys_loss
```

**Why it works**:
- Boom is fundamentally a *physical* event
- Adds inductive bias that speeds learning
- Could work with fewer examples

**Feasibility**: Medium-High (requires domain knowledge)
**Expected impact**: Potential 15-30% improvement in hard cases

---

### 8. Ensemble of Diverse Framings

**Status**: Partially attempted, could be extended

**Rationale**:
- Different models capture different signals
- Ensemble vote on boom frame

**Approach**:
```python
def ensemble_diverse_models(features):
    predictions = {
        'cnn': cnn_model.predict(features),
        'changepoint': pelt_detector.predict(features),
        'clustering': cluster_boundary_detector.predict(features),
        'peak': find_peak_in_energy(features),
        'hgb': hgb_classifier.predict(features),
    }

    # Weighted vote or median
    boom_frame = np.median(list(predictions.values()))
    confidence = fraction_voting_for_winner

    return boom_frame, confidence
```

**Why it works**:
- Diversity reduces systematic errors
- Disagreement reveals uncertainty
- Simple ensemble often beats complex single model

**Feasibility**: High (extend current pipeline)
**Expected impact**: Could reduce MAE by 10-20%

---

## Comparison & Recommendations

| Approach | Feasibility | Expected Impact | Complexity | Time to Test |
|----------|-------------|-----------------|------------|--------------|
| 1. Changepoint Detection | High | 15-25% | Low | 2-3 hours |
| 2. Multi-Task Learning | Medium-High | 15-30% | Medium | 1 day |
| 3. Self-Supervised Pretraining | High | 10-20% | Low | 4-6 hours |
| 4. Ordinal Regression | Medium | 5-15% | Low | 3-4 hours |
| 5. Uncertainty Quantification | High | 10-20% | Low | 4-6 hours |
| 6. Temporal Clustering | High | 20-40% | Very Low | 2-3 hours |
| 7. Physics-Informed NNs | Medium | 15-30% | High | 2-3 days |
| 8. Diverse Ensemble | High | 10-20% | Low | 4-6 hours |

---

## Quick Wins to Try First (Recommended Order)

1. **Changepoint Detection (Approach #1)**: 2-3 hours
   - Test PELT on position variance + velocity features
   - Should excel at finding the discontinuity

2. **Temporal Clustering (Approach #6)**: 2-3 hours
   - Test if 3-cluster KMeans naturally splits pre/boom/post phases
   - Very interpretable

3. **Diverse Ensemble (Approach #8)**: 4-6 hours
   - Changepoint + CNN + HGB vote
   - Reduces outliers via median

4. **Uncertainty Quantification (Approach #5)**: 4-6 hours
   - Replace CNN's point prediction with N(mean, std)
   - Better quality prediction

---

## Physical Intuition for Better Approaches

The boom has three distinct phases:

1. **Before**: Pendulums separate into clusters, features relatively stable
2. **At boom**: Clusters rapidly converge, position variance drops sharply, velocity spikes
3. **After**: Explosion, caustic patterns, chaotic

**Best discriminators**:
- Position variance: drops 10-50x at boom (changepoint detector strength)
- Velocity magnitude: spikes at boom (peak detection)
- Cluster structure: changes from 2+ clusters to 1 (clustering approach strength)
- Energy transfer: kinetic↔potential switches (physics-informed strength)

This suggests **approaches #1, #6, and #7 are most aligned with physics**, not just random ML tricks.

---

## Key Insights from Codebase

1. **Agreement is powerful**: CNN+HGB agreement achieves MAE 7.2 on "easy" cases
   - Implication: Models are good at certain cases; problem is selective

2. **Different features matter for different tasks**:
   - Boom detection: variance, range, tip position
   - Quality prediction: derivatives (d1_std, d1_var)
   - Implication: Specialized feature sets per model could help

3. **51% fail on agreement, 15% on quality**:
   - Most rejections are "hard cases" where both models struggle
   - Implication: Better base models (changepoint, clustering) would help more than better ensemble

4. **Phase 4 (changepoint) was tried but not fully evaluated**:
   - Not tested as part of ensemble, not tuned deeply
   - **Opportunity**: Hybrid changepoint + learned confidence could work
