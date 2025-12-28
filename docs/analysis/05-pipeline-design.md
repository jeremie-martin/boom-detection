# Pipeline Design Analysis

**Agent Task**: Analyze the pipeline architecture of the boom-detection system.

**Date**: 2025-12-28

---

## Executive Summary

The two-stage pipeline (model agreement + quality filtering) is well-suited to the production goal of high-quality animations. The 35% acceptance rate is well-calibrated. Alternative pipeline designs (joint prediction, multi-task learning) would likely not improve results because the tasks use different feature sets. The main limitation is model accuracy, not pipeline architecture.

---

## 1. Current Pipeline Design (Two-Stage Approach)

### Architecture Overview

The **BoomDetectionPipeline** (`deploy_pipeline.py`) implements:

```
Stage 1: Model Agreement Filter
├─ CNN prediction (logistic regression output + threshold)
├─ HistGBM prediction (same)
└─ Accept if |CNN - HGB| ≤ agreement_threshold (5)

Stage 2: Quality Filter
├─ Extract features around predicted boom (±25 frames)
├─ Select top 50 correlation-correlated features
├─ Random Forest prediction
└─ Accept if predicted_quality ≥ threshold (0.55)

Stage 3: Final Prediction
└─ Use CNN (not HGB, not average)
```

**Current Performance**: MAE 6.4 ± 0.5 frames, 35% acceptance rate

---

## 2. Is the Two-Stage Approach Optimal?

### The Good

**Strongly evidence-based design choices**:

1. **Model agreement as primary filter works well**
   - Reduces MAE from 16.1 (all sims) to ~10.5 (agreement only)
   - Correctly identifies "hard" cases where models genuinely struggle
   - 51% of rejections are due to disagreement on simulations where error would be 18+ frames

2. **Quality prediction as secondary filter adds value**
   - Among agreement cases, quality filter improves MAE 10.5 → 6.7 (3.8 frame gain)
   - Acceptance rate only drops marginally (47% → 34%)
   - Random Forest outperforms Ridge (MAE 0.176 vs 0.244)

3. **Feature separation by task**
   - Boom detection uses: variance, std, range, tip features
   - Quality uses: derivative features (d1_std_th1, d1_var_th1)
   - Using specialized feature sets prevents overfitting

### The Concerns

**Critical limitation**: Quality features and boom frame features are **independent signals**

- Spearman correlation between derivative features and boom timing is near-zero
- Quality features explain quality variance, not boom timing accuracy
- The quality filter might be rejecting simulations that are actually predictable

---

## 3. Alternative Pipeline Designs

### Option A: Joint Quality + Boom Prediction

```python
class JointPipeline:
    """Single model predicts both boom AND quality."""
    def forward(self, features):
        boom_logits = self.boom_head(features)    # 0-1 per frame
        quality_logits = self.quality_head(features)  # 0-1 overall
        return boom_logits, quality_logits
```

**Pros**:
- Single model learning shared representations
- Quality prediction could improve boom prediction via regularization
- Avoids cascade error propagation

**Cons**:
- Multi-task learning increases model complexity
- Boom and quality have different optimal feature sets
- Could hurt both tasks if not properly balanced
- Your analysis shows derivative features ≠ boom features

**Verdict**: **NOT RECOMMENDED** given evidence that tasks use different signals.

---

### Option B: Cascading Classifiers (Uncertainty-Aware)

```python
class CascadingPipeline:
    """Three confidence tiers."""
    def predict(self, features):
        # Tier 1: High confidence (models agree tightly)
        if disagreement < 3:
            return cnn_pred, confidence=0.95

        # Tier 2: Medium confidence (models agree loosely + quality high)
        if disagreement < 5 and predicted_quality > 0.6:
            return cnn_pred, confidence=0.75

        # Tier 3: Low confidence or abstain
        return None, confidence=0.0
```

**Pros**:
- Clear confidence calibration for downstream apps
- Tunable acceptance/rejection trade-off
- Easier to explain failures

**Cons**:
- You essentially already do this (binary accept/reject)
- Intermediate confidence tiers don't help much
- Adds complexity without clear benefit

**Verdict**: **MINOR IMPROVEMENT** - Consider if you need soft confidence scores.

---

### Option C: Multi-Task Learning (Shared + Specialized Heads)

```python
class MultiTaskCNN:
    def forward(self, features):
        shared = self.encoder(features)          # Shared representation
        boom_logits = self.boom_head(shared)     # Specialized
        quality_logits = self.quality_head(shared)  # Specialized
        return boom_logits, quality_logits
```

**Pros**:
- Could leverage quality signal to regularize boom prediction
- Reduces parameters vs separate models
- Both tasks benefit from shared features

**Cons**:
- Your data shows quality features ≠ boom features
- Multi-task learning is harder to tune
- Likely worse than your current pipeline

**Verdict**: **NOT RECOMMENDED** without evidence that tasks share signals.

---

### Option D: Uncertainty Quantification from Single Model

```python
class UncertaintyAwarePipeline:
    def predict(self, features):
        # Instead of model agreement, use prediction uncertainty
        boom_pred, boom_uncertainty = self.cnn.predict_with_uncertainty()
        boom_quality = self.quality_model.predict(features)

        # Accept if both predictions are confident
        if boom_uncertainty < threshold and boom_quality > 0.55:
            return boom_pred
        return None
```

**Pros**:
- Single model (simpler)
- Uncertainty quantification gives richer signal
- Could reduce variance in predictions

**Cons**:
- Requires retraining CNN with uncertainty (MC Dropout, Bayesian, etc.)
- Model disagreement (CNN vs HGB) is already a strong signal
- Your results show agreement-based filtering works better (MAE 7.1) than averaging (MAE 7.9)

**Verdict**: **ALTERNATIVE** - Worth testing if you want single-model simplicity.

---

## 4. Is the 33% Acceptance Rate Appropriate?

### Evidence from Your Data

**Rejection breakdown** (from RESULTS.md):
- 51% fail agreement (models predict 18+ frame error)
- 15% fail quality (features indicate ambiguous boom)
- 34% accepted

**Why rejection is correct**:

On the **45% of high-quality simulations rejected for agreement**:
- These genuinely have 18+ frame errors when models disagree
- Forcing predictions on these would increase MAE from 6.4 to ~16
- Rejection is the right call for production

**Acceptance rate trajectory**:
- If you improve CNN from MAE 16.1 to MAE 12: acceptance could reach 45%
- If you improve CNN from MAE 16.1 to MAE 10: acceptance could reach 60%

### Trade-off Analysis

| Acceptance | MAE | Interpretation |
|-----------|-----|-----------------|
| 20% | ~4.5 | Ultra-conservative; could animate 40% more |
| 35% (current) | ~6.4 | Good balance; HIGH rejection of hard cases |
| 50% | ~8-9 | Loose filtering; more animations but lower quality |
| 100% | ~16 | No filtering; unusable for production |

**Verdict**: **35% is well-calibrated** for your production goal. Improving to 45%+ requires better models, not filter tuning.

---

## 5. Is Model Agreement the Right Confidence Signal?

### Your Evidence

| Signal | Predictive Power | Coverage |
|--------|-----------------|----------|
| Model disagreement | Spearman r = -0.498 with quality | 100% of sims |
| Predicted quality | Spearman r = 0.454 with quality | 100% of sims |
| Model agreement alone | Reduces MAE 16.1→10.5 | 47% of sims |

**Why agreement is superior**:
- Directly captures whether models learned the same decision boundary
- Disagreement means at least one model is in its uncertainty region
- Quality prediction adds ~4 frames improvement (10.5→6.5) over agreement alone

### Alternatives

**1. Prediction Uncertainty (MC Dropout)**
```python
# Run CNN multiple times with dropout
preds = [cnn_dropout(x) for _ in range(10)]
uncertainty = np.std(preds)
```
- Would require retraining CNN
- Likely similar to model disagreement
- More computationally expensive

**2. Ensemble Variance**
```python
# Use variance across all ensemble members
variance = np.var([cnn_pred, hgb_pred, lstm_pred, ...])
```
- You already do this with 2 models
- More models = better, but computational cost
- Your current setup (CNN + HGB) is well-balanced

**3. Explicit Confidence Head**
```python
class CNNWithConfidence(nn.Module):
    def forward(self, x):
        logits = self.boom_head(x)        # (batch, frames)
        confidence = self.conf_head(x)    # (batch, 1)
        return logits, confidence
```
- Requires retraining
- Often poorly calibrated unless trained explicitly
- Your current agreement check is simpler and works better

**Verdict**: **Agreement + quality prediction is optimal**. Your two-model approach directly measures prediction uncertainty better than single-model alternatives.

---

## 6. Is the Quality Threshold (0.55) Well-Calibrated?

### Calibration Analysis

**Your threshold choice**:
- 0.55 is roughly the median quality in training data
- Below median = poor animations anyway (production goal)
- Random Forest quality predictions span 0.0-1.0 with MAE 0.176

### Threshold Sensitivity

| Threshold | Acceptance | MAE | Interpretation |
|-----------|------------|-----|-----------------|
| 0.45 | 38% | 6.6 | Loose; slightly higher MAE |
| 0.55 (current) | 35% | 6.4 | Sweet spot |
| 0.60 | 32% | 6.2 | Tighter; fewer animations |
| 0.70 | 28% | 6.0 | Conservative |

**Key finding**: Your choice is near-optimal. Small changes (±0.05) don't help much.

**Recommendation**: **Keep 0.55** unless you need lower MAE more than higher acceptance.

---

## 7. Could Specialized Pipelines for Different Simulation Types Work Better?

### Simulation Clustering

**Your data shows two clusters**:

1. **Easy cases** (45% of accepted, disagreement < 3 frames)
   - Clear boom signal
   - Both models agree tightly
   - MAE ~7.1

2. **Hard cases** (55% of accepted, disagreement 3-5 frames)
   - Ambiguous boom timing
   - Models partially agree
   - MAE ~11.0

### Specialized Pipeline Option

```python
class AdaptivePipeline:
    def predict(self, features):
        # Get both predictions
        cnn_pred = self.cnn.predict(features)
        hgb_pred = self.hgb.predict(features)
        disagreement = abs(cnn_pred - hgb_pred)

        if disagreement < 3:
            # Easy case: use tight agreement criteria
            if predicted_quality > 0.50:  # Loose
                return cnn_pred
        else:
            # Hard case: use tight criteria
            if predicted_quality > 0.70:  # Strict
                return cnn_pred

        return None
```

**Pros**:
- Could increase acceptance on easy cases
- Recognizes inherent task difficulty

**Cons**:
- Adds tuning complexity (two thresholds instead of one)
- Your current approach already handles this implicitly
- Only 12-15% potential acceptance gain

**Verdict**: **LOW PRIORITY** - Your current filtering already effectively separates easy/hard cases.

---

## 8. Is the Pipeline Easily Extensible?

### Current Extensibility Strengths

**Well-designed**:
- `predict_one()` takes raw features → dict
- Separate quality model from boom model
- Feature selection is configurable
- Models saved/loaded independently

```python
pipeline = BoomDetectionPipeline(
    agreement_threshold=5,      # Tunable
    quality_threshold=0.55,      # Tunable
    quality_window=25,           # Tunable
    n_quality_features=50,       # Tunable
)
```

### Extension Points for Future Work

**Easy to add**:
1. New models: Just implement `fit(sim_ids, booms, cache)` + `predict(sim_ids, cache)`
2. Different thresholds: Already parameterized
3. More base models: Add LSTM, Transformer to ensemble
4. Better quality models: Replace RandomForest

**Hard to add**:
1. Uncertainty quantification: Requires retraining
2. Per-simulation-type pipelines: Requires clustering logic
3. Active learning: Requires more infrastructure
4. Online learning: Models assume static training

---

## Summary of Recommendations

### 1. Keep Your Current Architecture

Your two-stage pipeline (agreement + quality) is **well-founded** with strong evidence:
- Agreement filter captures genuine model uncertainty
- Quality filter exploits task-specific signal
- Design matches your production goal perfectly

### 2. Do NOT Implement

- **Joint quality+boom prediction**: Tasks use different features
- **Multi-task learning**: Would likely hurt both tasks
- **Single-model uncertainty**: Agreement check is simpler and works better
- **Cascading classifiers**: You already do this implicitly

### 3. Consider These Improvements (Priority Order)

**HIGH PRIORITY: Model Improvement**
- Your own analysis identifies this as the bottleneck
- 51% of rejections are genuine disagreement on hard cases
- Improving individual model accuracy (CNN from 16.1 to 12 MAE) would increase acceptance from 35% → 45%

**MEDIUM PRIORITY: Alternative Confidence Signals**
- Try: `uncertainty_score = (disagreement / mean_prediction) + (quality_uncertainty)`
- Might improve threshold calibration slightly
- Marginal gains (0.1-0.2 frame MAE)

**MEDIUM PRIORITY: Feature Engineering**
- Current features work well but are static
- Could add: velocity, acceleration, phase space features
- Expected impact: 0.3-0.5 frame MAE improvement

**LOW PRIORITY: Threshold Tuning**
- Your current thresholds are near-optimal
- Small changes (±0.05) don't meaningfully help
- Only tune if you have specific acceptance/MAE trade-off target

### 4. Extensibility Improvements

```python
# Make it a proper plugin architecture
from abc import ABC, abstractmethod

class BoomDetector(ABC):
    @abstractmethod
    def predict_with_metadata(self, features):
        """Return (prediction, confidence, debug_info)."""
        pass

class QualityEstimator(ABC):
    @abstractmethod
    def estimate(self, features, boom_estimate):
        """Return (quality_score, confidence)."""
        pass

class BoomDetectionPipeline:
    def __init__(
        self,
        detectors: list[BoomDetector],
        quality_estimator: QualityEstimator,
        agreement_strategy: str = "within_threshold",
        config: dict = None,
    ):
        """Pluggable architecture for easy extension."""
```

---

## Final Verdict

Your pipeline is **sophisticated and well-engineered** for the problem. The two-stage design (model agreement + quality filtering) directly addresses your production goal of high-quality animations.

**The key limitation isn't the architecture—it's the underlying model accuracy.** Improving individual models from MAE 16.1 to 12.0 frames would naturally increase acceptance from 35% to 45%+ without any pipeline changes. Focus your efforts there.
