"""
Change point detection methods for boom detection.

These methods look for sudden changes in feature distributions
that indicate the boom event.

Usage:
    from boom_detection.changepoint import CUSUMDetector, BOCPDDetector

    detector = CUSUMDetector(feature_idx='variance_x2')
    detector.fit(train_ids, train_boom_frames, cache)
    predictions = detector.predict(test_ids, cache)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .features import FeatureCache, FEATURE_GROUPS


def get_feature_index(feature_name: str, config_flags: dict | None = None) -> int:
    """Get the index of a feature by name."""
    # Build feature list based on default config
    all_features = []
    all_features.extend(FEATURE_GROUPS['statistical'])
    all_features.extend(FEATURE_GROUPS['diff'])

    if config_flags is None or config_flags.get('include_caustic', True):
        all_features.extend(FEATURE_GROUPS['caustic'])
    if config_flags is None or config_flags.get('include_rolling', True):
        all_features.extend(FEATURE_GROUPS['rolling'])

    try:
        return all_features.index(feature_name)
    except ValueError:
        raise ValueError(f"Feature '{feature_name}' not found. Available: {all_features[:10]}...")


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change point detector.

    Detects when a signal deviates significantly from its expected value.
    Works well for detecting mean shifts in time series.

    The algorithm accumulates deviations from a target value. When the
    cumulative sum exceeds a threshold, a change point is detected.
    """

    def __init__(
        self,
        feature_idx: int | str = 0,
        threshold_percentile: float = 95,
        drift: float = 0.5,
        use_derivative: bool = True,
    ):
        """
        Args:
            feature_idx: Index or name of feature to monitor
            threshold_percentile: Percentile of training CUSUM values for threshold
            drift: Allowable drift before accumulating (in std units)
            use_derivative: If True, detect changes in rate of change
        """
        self.feature_idx = feature_idx
        self.threshold_percentile = threshold_percentile
        self.drift = drift
        self.use_derivative = use_derivative
        self.threshold_ = None
        self.mean_ = None
        self.std_ = None

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Learn threshold from training data."""
        # Resolve feature name to index if needed
        if isinstance(self.feature_idx, str):
            self.feature_idx = get_feature_index(self.feature_idx)

        # Collect pre-boom statistics to establish baseline
        all_pre_boom = []
        all_cusum_at_boom = []

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]
            signal = features[:, self.feature_idx]

            if self.use_derivative:
                signal = np.diff(signal, prepend=signal[0])

            # Use pre-boom region to establish baseline
            pre_boom = signal[:int(boom)]
            if len(pre_boom) > 10:
                all_pre_boom.extend(pre_boom)

            # Compute CUSUM and record value at boom
            cusum = self._compute_cusum(signal, np.mean(pre_boom), np.std(pre_boom) + 1e-8)
            if int(boom) < len(cusum):
                all_cusum_at_boom.append(cusum[int(boom)])

        # Set threshold based on CUSUM values at known boom frames
        if all_cusum_at_boom:
            self.threshold_ = np.percentile(all_cusum_at_boom, 100 - self.threshold_percentile)
        else:
            self.threshold_ = 0

        self.mean_ = np.mean(all_pre_boom) if all_pre_boom else 0
        self.std_ = np.std(all_pre_boom) + 1e-8 if all_pre_boom else 1

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict boom frames using CUSUM."""
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            signal = features[:, self.feature_idx]

            if self.use_derivative:
                signal = np.diff(signal, prepend=signal[0])

            cusum = self._compute_cusum(signal, self.mean_, self.std_)

            # Find first threshold crossing
            crossings = np.where(cusum > self.threshold_)[0]
            if len(crossings) > 0:
                predictions.append(int(crossings[0]))
            else:
                # Fallback: find max CUSUM change
                predictions.append(int(np.argmax(np.diff(cusum, prepend=0))))

        return np.array(predictions)

    def _compute_cusum(self, signal: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Compute cumulative sum statistic."""
        # Normalize
        z = (signal - mean) / std

        # CUSUM with drift
        cusum_pos = np.zeros_like(signal)
        cusum_neg = np.zeros_like(signal)

        for i in range(1, len(signal)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - self.drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - self.drift)

        # Return max of positive and negative CUSUM
        return np.maximum(cusum_pos, cusum_neg)


class BOCPDDetector:
    """
    Bayesian Online Change Point Detection.

    Maintains a probability distribution over the time since the last
    change point. When this distribution shifts dramatically, a change
    point is detected.

    Reference: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
    """

    def __init__(
        self,
        feature_idx: int | str = 0,
        hazard_rate: float = 1/200,  # Expected run length ~200 frames
        threshold: float = 0.5,
    ):
        """
        Args:
            feature_idx: Index or name of feature to monitor
            hazard_rate: Prior probability of change point at each step
            threshold: Probability threshold for detecting change
        """
        self.feature_idx = feature_idx
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.prior_mean_ = None
        self.prior_var_ = None

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,  # noqa: ARG002 - required by interface
        cache: FeatureCache,
    ) -> None:
        """Learn prior parameters from training data."""
        del boom_frames  # unused but required by interface

        if isinstance(self.feature_idx, str):
            self.feature_idx = get_feature_index(self.feature_idx)

        # Collect all values to establish prior
        all_values = []
        for sim_id in sim_ids:
            features = cache[sim_id]
            all_values.extend(features[:, self.feature_idx])

        self.prior_mean_ = np.mean(all_values)
        self.prior_var_ = np.var(all_values) + 1e-8

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict boom frames using BOCPD."""
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            signal = features[:, self.feature_idx]

            change_probs = self._run_bocpd(signal)

            # Find first significant change point
            # Look for where P(change) exceeds threshold
            crossings = np.where(change_probs > self.threshold)[0]
            if len(crossings) > 0:
                # Skip very early detections (likely noise)
                valid = crossings[crossings > 50]
                if len(valid) > 0:
                    predictions.append(int(valid[0]))
                else:
                    predictions.append(int(crossings[0]))
            else:
                # Fallback: highest change probability
                predictions.append(int(np.argmax(change_probs)))

        return np.array(predictions)

    def _run_bocpd(self, signal: np.ndarray) -> np.ndarray:
        """Run BOCPD algorithm and return change point probabilities."""
        T = len(signal)

        # Run length distribution: R[t, r] = P(run_length=r at time t)
        # For efficiency, we only track the marginal P(change at t)

        # Sufficient statistics for Gaussian with unknown mean
        # Using Normal-Gamma conjugate prior
        mu0 = self.prior_mean_
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = self.prior_var_

        # Initialize run length distribution
        max_run = min(T, 500)  # Limit for memory
        R = np.zeros((T + 1, max_run))
        R[0, 0] = 1.0

        # Sufficient statistics per run length
        sum_x = np.zeros(max_run)
        sum_x2 = np.zeros(max_run)
        counts = np.zeros(max_run)

        change_probs = np.zeros(T)

        for t in range(T):
            x = signal[t]

            # Compute predictive probabilities for each run length
            pred_probs = np.zeros(min(t + 1, max_run))

            for r in range(min(t + 1, max_run)):
                if counts[r] == 0:
                    # Use prior
                    mu = mu0
                    var = beta0 / alpha0
                else:
                    # Posterior predictive (Student-t)
                    n = counts[r]
                    mean_x = sum_x[r] / n
                    kappa_n = kappa0 + n
                    mu_n = (kappa0 * mu0 + sum_x[r]) / kappa_n
                    alpha_n = alpha0 + n / 2
                    beta_n = beta0 + 0.5 * (sum_x2[r] - sum_x[r]**2 / n)
                    beta_n += 0.5 * kappa0 * n * (mean_x - mu0)**2 / kappa_n

                    mu = mu_n
                    var = beta_n * (kappa_n + 1) / (alpha_n * kappa_n) + 1e-8

                # Gaussian predictive probability
                pred_probs[r] = stats.norm.pdf(x, mu, np.sqrt(var))

            # Growth probabilities
            growth_probs = R[t, :len(pred_probs)] * pred_probs * (1 - self.hazard_rate)

            # Change point probability
            cp_prob = np.sum(R[t, :len(pred_probs)] * pred_probs * self.hazard_rate)
            change_probs[t] = cp_prob

            # Update run length distribution
            R[t + 1, 0] = cp_prob
            R[t + 1, 1:len(growth_probs) + 1] = growth_probs

            # Normalize
            total = R[t + 1, :].sum()
            if total > 0:
                R[t + 1, :] /= total

            # Update sufficient statistics
            new_sum_x = np.zeros(max_run)
            new_sum_x2 = np.zeros(max_run)
            new_counts = np.zeros(max_run)

            new_sum_x[0] = x
            new_sum_x2[0] = x * x
            new_counts[0] = 1

            new_sum_x[1:] = sum_x[:-1] + x
            new_sum_x2[1:] = sum_x2[:-1] + x * x
            new_counts[1:] = counts[:-1] + 1

            sum_x = new_sum_x
            sum_x2 = new_sum_x2
            counts = new_counts

        return change_probs


class MultiFeatureCUSUM:
    """
    CUSUM applied to multiple features, combined by voting or averaging.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_features: int = 5,
        combination: str = 'median',  # 'median', 'mean', 'vote'
        **cusum_kwargs,
    ):
        """
        Args:
            feature_names: Specific features to use (if None, auto-select)
            n_features: Number of top features to use if auto-selecting
            combination: How to combine predictions from multiple features
            **cusum_kwargs: Passed to CUSUMDetector
        """
        self.feature_names = feature_names
        self.n_features = n_features
        self.combination = combination
        self.cusum_kwargs = cusum_kwargs
        self.detectors_: list[CUSUMDetector] = []
        self.selected_features_: list[int] = []

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit CUSUM detectors for each feature."""
        if self.feature_names:
            # Use specified features
            self.selected_features_ = [
                get_feature_index(name) for name in self.feature_names
            ]
        else:
            # Auto-select features with highest variance change at boom
            self.selected_features_ = self._select_features(sim_ids, boom_frames, cache)

        # Create and fit a detector for each feature
        self.detectors_ = []
        for feat_idx in self.selected_features_:
            detector = CUSUMDetector(feature_idx=feat_idx, **self.cusum_kwargs)
            detector.fit(sim_ids, boom_frames, cache)
            self.detectors_.append(detector)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict by combining multiple CUSUM detectors."""
        all_preds = np.array([d.predict(sim_ids, cache) for d in self.detectors_])

        if self.combination == 'median':
            return np.median(all_preds, axis=0).astype(int)
        elif self.combination == 'mean':
            return np.mean(all_preds, axis=0).astype(int)
        elif self.combination == 'vote':
            # Histogram voting
            predictions = []
            for i in range(all_preds.shape[1]):
                preds = all_preds[:, i]
                # Round to nearest 10 frames and vote
                rounded = (preds / 10).astype(int) * 10
                values, counts = np.unique(rounded, return_counts=True)
                predictions.append(int(values[np.argmax(counts)]))
            return np.array(predictions)
        else:
            raise ValueError(f"Unknown combination: {self.combination}")

    def _select_features(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> list[int]:
        """Select features with highest variance change at boom."""
        # Compute variance ratio (post-boom / pre-boom) for each feature
        n_features = cache[sim_ids[0]].shape[1]
        variance_ratios = np.zeros(n_features)

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]
            boom = int(boom)

            if boom < 20 or boom > len(features) - 20:
                continue

            pre_var = np.var(features[:boom], axis=0) + 1e-8
            post_var = np.var(features[boom:], axis=0) + 1e-8
            variance_ratios += post_var / pre_var

        # Select top features by variance ratio
        top_indices = np.argsort(variance_ratios)[-self.n_features:]
        return list(top_indices)
