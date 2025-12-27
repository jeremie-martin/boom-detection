"""
Feature extraction for boom detection.

All features are designed to be resolution-invariant: they aggregate over
pendulums to produce per-frame statistics that don't depend on pendulum count.

Usage:
    from boom_detection.features import extract_features, FeatureExtractor

    # Extract features from a simulation
    features = extract_features(simulation)  # shape: (frames, n_features)

    # Or use the class for more control
    extractor = FeatureExtractor()
    features = extractor.transform(simulation)

    # For efficiency, use FeatureCache to extract once and reuse
    from boom_detection.features import FeatureCache

    cache = FeatureCache()
    cache.extract_all(dataset)  # extracts features for all simulations
    features = cache.get(simulation_id)  # instant lookup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np

from .loader import Simulation, X1, Y1, X2, Y2, TH1, TH2, W1, W2

if TYPE_CHECKING:
    from .loader import Dataset


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def variance_features(data: np.ndarray) -> np.ndarray:
    """
    Compute variance of each field across pendulums for each frame.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - variance of each field
    """
    return np.var(data, axis=1)


def iqr_features(data: np.ndarray) -> np.ndarray:
    """
    Compute interquartile range of each field across pendulums.

    More robust to outliers than variance.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - IQR of each field
    """
    q75 = np.percentile(data, 75, axis=1)
    q25 = np.percentile(data, 25, axis=1)
    return q75 - q25


def range_features(data: np.ndarray) -> np.ndarray:
    """
    Compute range (max - min) of each field across pendulums.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - range of each field
    """
    return np.max(data, axis=1) - np.min(data, axis=1)


def mean_features(data: np.ndarray) -> np.ndarray:
    """
    Compute mean of each field across pendulums.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - mean of each field
    """
    return np.mean(data, axis=1)


def std_features(data: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation of each field across pendulums.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - std of each field
    """
    return np.std(data, axis=1)


def skewness_features(data: np.ndarray) -> np.ndarray:
    """
    Compute skewness of each field across pendulums.

    Measures asymmetry of the distribution.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - skewness of each field
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std = np.where(std < 1e-9, 1e-9, std)  # avoid division by zero
    return np.mean(((data - mean) / std) ** 3, axis=1)


def kurtosis_features(data: np.ndarray) -> np.ndarray:
    """
    Compute excess kurtosis of each field across pendulums.

    Measures "tailedness" of the distribution.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 8) - excess kurtosis of each field
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std = np.where(std < 1e-9, 1e-9, std)
    return np.mean(((data - mean) / std) ** 4, axis=1) - 3


def tip_spread_features(data: np.ndarray) -> np.ndarray:
    """
    Compute spread metrics specific to tip positions.

    This captures the visual "spread" that humans perceive.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 3) - [area, max_dist, mean_dist_from_centroid]
    """
    x2 = data[:, :, X2]  # (frames, pendulums)
    y2 = data[:, :, Y2]

    # Bounding box area
    x_range = np.max(x2, axis=1) - np.min(x2, axis=1)
    y_range = np.max(y2, axis=1) - np.min(y2, axis=1)
    area = x_range * y_range

    # Max pairwise distance (approximated via max dist from centroid * 2)
    cx = np.mean(x2, axis=1, keepdims=True)
    cy = np.mean(y2, axis=1, keepdims=True)
    dist_from_centroid = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    max_dist = np.max(dist_from_centroid, axis=1) * 2
    mean_dist = np.mean(dist_from_centroid, axis=1)

    return np.stack([area, max_dist, mean_dist], axis=1)


def angular_spread_features(data: np.ndarray) -> np.ndarray:
    """
    Compute spread of angles using circular statistics.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 4) - [th1_spread, th2_spread, th1_circular_std, th2_circular_std]
    """
    th1 = data[:, :, TH1]
    th2 = data[:, :, TH2]

    # Linear spread
    th1_spread = np.max(th1, axis=1) - np.min(th1, axis=1)
    th2_spread = np.max(th2, axis=1) - np.min(th2, axis=1)

    # Circular standard deviation
    # R = |mean(exp(i*theta))|, circular_std = sqrt(-2*log(R))
    def circular_std(angles):
        mean_cos = np.mean(np.cos(angles), axis=1)
        mean_sin = np.mean(np.sin(angles), axis=1)
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        R = np.clip(R, 1e-9, 1.0)  # avoid log(0)
        return np.sqrt(-2 * np.log(R))

    th1_cstd = circular_std(th1)
    th2_cstd = circular_std(th2)

    return np.stack([th1_spread, th2_spread, th1_cstd, th2_cstd], axis=1)


def velocity_features(data: np.ndarray) -> np.ndarray:
    """
    Compute velocity statistics.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 6) - stats on w1, w2
    """
    w1 = data[:, :, W1]
    w2 = data[:, :, W2]

    return np.stack([
        np.var(w1, axis=1),
        np.var(w2, axis=1),
        np.mean(np.abs(w1), axis=1),
        np.mean(np.abs(w2), axis=1),
        np.max(np.abs(w1), axis=1) - np.min(np.abs(w1), axis=1),
        np.max(np.abs(w2), axis=1) - np.min(np.abs(w2), axis=1),
    ], axis=1)


# =============================================================================
# Caustic / Angular Distribution Features
# =============================================================================

def _wrap_2pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 2π)."""
    return angles % (2 * np.pi)


def _gini_coefficient(counts: np.ndarray) -> np.ndarray:
    """
    Compute Gini coefficient of histogram counts.

    Vectorized for multiple frames: counts shape (frames, n_bins).

    Returns:
        Shape (frames,) - Gini coefficient per frame
    """
    # Sort counts ascending along bin axis
    sorted_counts = np.sort(counts, axis=1)
    n_bins = counts.shape[1]
    total = counts.sum(axis=1, keepdims=True)

    # Avoid division by zero
    total = np.where(total < 1e-9, 1e-9, total)

    # Gini formula: sum of (2*rank - n - 1) * x / (n * sum)
    ranks = np.arange(1, n_bins + 1)  # 1-indexed ranks
    weights = 2 * ranks - n_bins - 1
    numerator = (sorted_counts * weights).sum(axis=1)

    return numerator / (n_bins * total.squeeze())


def _mean_resultant_length(angles: np.ndarray) -> np.ndarray:
    """
    Compute mean resultant length (circular concentration).

    Args:
        angles: Shape (frames, pendulums)

    Returns:
        Shape (frames,) - R value in [0, 1], high = concentrated
    """
    mean_cos = np.mean(np.cos(angles), axis=1)
    mean_sin = np.mean(np.sin(angles), axis=1)
    return np.sqrt(mean_cos**2 + mean_sin**2)


def _angular_histogram(angles: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Compute histogram of angles across pendulums for each frame.

    Args:
        angles: Shape (frames, pendulums)
        n_bins: Number of angular bins (36 = 10° bins)

    Returns:
        Shape (frames, n_bins) - counts per bin
    """
    n_frames = angles.shape[0]
    wrapped = _wrap_2pi(angles)

    # Bin edges from 0 to 2π
    bin_width = 2 * np.pi / n_bins
    bin_indices = np.clip((wrapped / bin_width).astype(int), 0, n_bins - 1)

    # Count per bin per frame (vectorized via advanced indexing)
    counts = np.zeros((n_frames, n_bins), dtype=np.float64)
    frame_indices = np.arange(n_frames)[:, None]  # (frames, 1)

    # Use np.add.at for unbuffered in-place addition
    np.add.at(counts, (np.broadcast_to(frame_indices, bin_indices.shape), bin_indices), 1)

    return counts


def _birthday_corrected_coverage(counts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Compute birthday-corrected coverage (occupied bins / expected occupied).

    Args:
        counts: Shape (frames, n_bins) - histogram counts
        n_samples: Number of samples (pendulums)

    Returns:
        Shape (frames,) - normalized coverage in [0, 1+]
    """
    n_bins = counts.shape[1]
    occupied = (counts > 0).sum(axis=1)
    raw_coverage = occupied / n_bins

    # Expected coverage if uniformly random: 1 - (1 - 1/M)^N
    expected = 1 - (1 - 1/n_bins) ** n_samples

    if expected > 0.01:
        return np.minimum(1.0, raw_coverage / expected)
    else:
        return raw_coverage


def _causticness_from_angles(angles: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Core causticness computation: coverage × gini.

    High when angles cover many bins AND are unevenly distributed (spiky).

    Args:
        angles: Shape (frames, pendulums)
        n_bins: Number of angular bins

    Returns:
        Shape (frames,) - causticness values
    """
    n_pendulums = angles.shape[1]
    counts = _angular_histogram(angles, n_bins)
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    gini = _gini_coefficient(counts)
    return coverage * gini


def caustic_features(data: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Compute caustic/angular distribution features.

    These capture angular clustering patterns that simple statistics miss.

    Features:
    1. angular_causticness: coverage × gini of (θ1 + θ2)
    2. tip_causticness: coverage × gini of atan2(x2, y2)
    3. joint_concentration: causticness(θ1) × causticness(θ2)
    4. organization_causticness: (1 - R1*R2) × coverage

    Args:
        data: Shape (frames, pendulums, 8)
        n_bins: Number of angular bins (36 = 10° bins)

    Returns:
        Shape (frames, 4) - the four caustic metrics
    """
    th1 = data[:, :, TH1]
    th2 = data[:, :, TH2]
    x2 = data[:, :, X2]
    y2 = data[:, :, Y2]
    n_pendulums = data.shape[1]

    # 1. Angular causticness: tip direction from angles
    tip_angles_from_theta = th1 + th2
    angular_caust = _causticness_from_angles(tip_angles_from_theta, n_bins)

    # 2. Tip causticness: tip direction from positions (atan2)
    tip_angles_from_pos = np.arctan2(x2, y2)  # angle from vertical
    tip_caust = _causticness_from_angles(tip_angles_from_pos, n_bins)

    # 3. Joint concentration: both angles individually show caustic behavior
    th1_caust = _causticness_from_angles(th1, n_bins)
    th2_caust = _causticness_from_angles(th2, n_bins)
    joint_conc = th1_caust * th2_caust

    # 4. Organization causticness: spread (low concentration) × coverage
    R1 = _mean_resultant_length(th1)
    R2 = _mean_resultant_length(th2)
    counts = _angular_histogram(tip_angles_from_theta, n_bins)
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    org_caust = (1 - R1 * R2) * coverage

    return np.stack([angular_caust, tip_caust, joint_conc, org_caust], axis=1)


def temporal_derivatives(features: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compute temporal derivatives of features.

    Args:
        features: Shape (frames, n_features)
        order: Derivative order (1 or 2)

    Returns:
        Shape (frames, n_features) - derivatives (padded to same length)
    """
    result = features.copy()
    for _ in range(order):
        deriv = np.diff(result, axis=0)
        # Pad to maintain length
        result = np.vstack([deriv, deriv[-1:]])
    return result


# =============================================================================
# Enhanced Temporal Features (Phase 2)
# =============================================================================

def rolling_features(features: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling window statistics (mean and std).

    Fully vectorized using uniform_filter1d.

    Args:
        features: Shape (frames, n_features)
        window: Window size

    Returns:
        Shape (frames, n_features * 2) - rolling mean and std
    """
    from scipy.ndimage import uniform_filter1d

    # Rolling mean - uniform_filter1d is fully vectorized C code
    rolling_mean = uniform_filter1d(features, size=window, axis=0, mode='nearest')

    # Rolling std via E[X^2] - E[X]^2
    rolling_mean_sq = uniform_filter1d(features ** 2, size=window, axis=0, mode='nearest')
    rolling_var = np.maximum(rolling_mean_sq - rolling_mean ** 2, 0)
    rolling_std = np.sqrt(rolling_var)

    return np.hstack([rolling_mean, rolling_std])


def lag_features(features: np.ndarray, lag: int) -> np.ndarray:
    """
    Compute lag features (difference from N frames ago).

    Vectorized implementation.

    Args:
        features: Shape (frames, n_features)
        lag: Number of frames to look back

    Returns:
        Shape (frames, n_features) - difference from lag frames ago
    """
    result = np.zeros_like(features)
    if lag < features.shape[0]:
        result[lag:] = features[lag:] - features[:-lag]
    return result


def relative_features(features: np.ndarray) -> np.ndarray:
    """
    Compute relative features: ratio to max and percentile position.

    Vectorized implementation using argsort for percentiles.

    Args:
        features: Shape (frames, n_features)

    Returns:
        Shape (frames, n_features * 2) - ratio_to_max, percentile_position
    """
    n_frames = features.shape[0]

    # Ratio to max (per feature) - already vectorized
    max_vals = np.max(features, axis=0, keepdims=True)
    max_vals = np.where(max_vals < 1e-9, 1e-9, max_vals)
    ratio_to_max = features / max_vals

    # Percentile position via argsort (vectorized)
    # For each feature, rank the frames and convert to percentile
    ranks = np.argsort(np.argsort(features, axis=0), axis=0)
    percentile_pos = ranks / (n_frames - 1) if n_frames > 1 else np.zeros_like(features)

    return np.hstack([ratio_to_max, percentile_pos])


# =============================================================================
# Feature Extractor Class
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    include_variance: bool = True
    include_std: bool = True
    include_iqr: bool = True
    include_range: bool = True
    include_mean: bool = False  # usually not informative
    include_skewness: bool = True
    include_kurtosis: bool = True
    include_tip_spread: bool = True
    include_angular_spread: bool = True
    include_velocity: bool = True
    include_caustic: bool = False  # caustic/angular distribution features
    caustic_bins: int = 36  # number of angular bins (36 = 10° bins)
    include_derivatives: bool = True
    derivative_orders: tuple[int, ...] = (1, 2)
    # Subsampling for speed and resolution invariance testing
    max_pendulums: int | None = None  # None = use all, e.g. 2000 for fast extraction
    subsample_seed: int = 42  # for reproducibility
    # Enhanced temporal features (Phase 2)
    include_rolling: bool = False  # rolling window statistics
    rolling_windows: tuple[int, ...] = (10, 25, 50)  # window sizes
    include_lag: bool = False  # lag features (diff from N frames ago)
    lag_steps: tuple[int, ...] = (5, 10, 25)  # lag amounts
    include_relative: bool = False  # relative features (ratio to max, percentile)


# Default configuration
DEFAULT_CONFIG = FeatureConfig()

# Enhanced configuration with temporal features (Phase 2)
ENHANCED_CONFIG = FeatureConfig(
    max_pendulums=2000,
    include_rolling=True,
    rolling_windows=(10, 25),  # reduced for speed
    include_lag=True,
    lag_steps=(5, 15),  # reduced for speed
    include_relative=True,
)

# Configuration with caustic/angular distribution features
CAUSTIC_CONFIG = FeatureConfig(
    max_pendulums=2000,
    include_caustic=True,
)

# Feature names for each extraction function
FEATURE_GROUPS = {
    'variance': ['var_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'std': ['std_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'iqr': ['iqr_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'range': ['range_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'mean': ['mean_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'skewness': ['skew_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'kurtosis': ['kurt_' + n for n in ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']],
    'tip_spread': ['tip_area', 'tip_max_dist', 'tip_mean_dist'],
    'angular_spread': ['th1_spread', 'th2_spread', 'th1_cstd', 'th2_cstd'],
    'velocity': ['var_w1', 'var_w2', 'mean_abs_w1', 'mean_abs_w2', 'range_abs_w1', 'range_abs_w2'],
    'caustic': ['angular_causticness', 'tip_causticness', 'joint_concentration', 'organization_causticness'],
}


class FeatureExtractor:
    """
    Extracts resolution-invariant features from simulations.

    Features are per-frame statistics that aggregate over pendulums.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        if self._feature_names is None:
            self._feature_names = self._compute_feature_names()
        return self._feature_names

    def _compute_feature_names(self) -> list[str]:
        """Compute feature names based on config."""
        names = []
        cfg = self.config

        base_groups = []
        if cfg.include_variance:
            base_groups.append('variance')
        if cfg.include_std:
            base_groups.append('std')
        if cfg.include_iqr:
            base_groups.append('iqr')
        if cfg.include_range:
            base_groups.append('range')
        if cfg.include_mean:
            base_groups.append('mean')
        if cfg.include_skewness:
            base_groups.append('skewness')
        if cfg.include_kurtosis:
            base_groups.append('kurtosis')
        if cfg.include_tip_spread:
            base_groups.append('tip_spread')
        if cfg.include_angular_spread:
            base_groups.append('angular_spread')
        if cfg.include_velocity:
            base_groups.append('velocity')
        if cfg.include_caustic:
            base_groups.append('caustic')

        for group in base_groups:
            names.extend(FEATURE_GROUPS[group])

        if cfg.include_derivatives:
            base_names = names.copy()
            for order in cfg.derivative_orders:
                names.extend([f"d{order}_{n}" for n in base_names])

        return names

    def transform(self, simulation: Simulation) -> np.ndarray:
        """
        Extract features from a simulation.

        Args:
            simulation: Simulation object

        Returns:
            Shape (frames, n_features) array of features
        """
        data = simulation.data
        cfg = self.config

        # Subsample pendulums if configured (for speed and resolution invariance)
        if cfg.max_pendulums is not None and data.shape[1] > cfg.max_pendulums:
            rng = np.random.RandomState(cfg.subsample_seed)
            indices = rng.choice(data.shape[1], cfg.max_pendulums, replace=False)
            data = data[:, indices, :]

        features_list = []

        # Base statistical features
        if cfg.include_variance:
            features_list.append(variance_features(data))
        if cfg.include_std:
            features_list.append(std_features(data))
        if cfg.include_iqr:
            features_list.append(iqr_features(data))
        if cfg.include_range:
            features_list.append(range_features(data))
        if cfg.include_mean:
            features_list.append(mean_features(data))
        if cfg.include_skewness:
            features_list.append(skewness_features(data))
        if cfg.include_kurtosis:
            features_list.append(kurtosis_features(data))

        # Specialized features
        if cfg.include_tip_spread:
            features_list.append(tip_spread_features(data))
        if cfg.include_angular_spread:
            features_list.append(angular_spread_features(data))
        if cfg.include_velocity:
            features_list.append(velocity_features(data))
        if cfg.include_caustic:
            features_list.append(caustic_features(data, cfg.caustic_bins))

        # Combine base features
        base_features = np.hstack(features_list)

        # Add temporal derivatives
        if cfg.include_derivatives:
            all_features = [base_features]
            for order in cfg.derivative_orders:
                all_features.append(temporal_derivatives(base_features, order))
            base_features = np.hstack(all_features)

        # Enhanced temporal features (Phase 2)
        enhanced = [base_features]

        if cfg.include_rolling:
            for window in cfg.rolling_windows:
                enhanced.append(rolling_features(base_features, window))

        if cfg.include_lag:
            for lag in cfg.lag_steps:
                enhanced.append(lag_features(base_features, lag))

        if cfg.include_relative:
            enhanced.append(relative_features(base_features))

        if len(enhanced) > 1:
            return np.hstack(enhanced)

        return base_features

    def transform_batch(
        self,
        simulations: Sequence[Simulation],
        verbose: bool = False
    ) -> list[np.ndarray]:
        """
        Extract features from multiple simulations.

        Args:
            simulations: List of Simulation objects
            verbose: Print progress

        Returns:
            List of feature arrays, one per simulation
        """
        results = []
        for i, sim in enumerate(simulations):
            if verbose and (i + 1) % 10 == 0:
                print(f"Extracting features: {i + 1}/{len(simulations)}")
            results.append(self.transform(sim))
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_features(
    simulation: Simulation,
    config: FeatureConfig | None = None
) -> np.ndarray:
    """
    Extract features from a simulation.

    Convenience function that creates an extractor and transforms.

    Args:
        simulation: Simulation object
        config: Optional feature configuration

    Returns:
        Shape (frames, n_features) array
    """
    extractor = FeatureExtractor(config)
    return extractor.transform(simulation)


def get_feature_names(config: FeatureConfig | None = None) -> list[str]:
    """Get list of feature names for a configuration."""
    return FeatureExtractor(config).feature_names


# =============================================================================
# Feature Cache
# =============================================================================

class FeatureCache:
    """
    Caches extracted features to avoid redundant computation.

    Extract features once for all simulations, then look them up instantly.
    Optionally saves to disk for persistence across runs.

    Usage:
        # In-memory only:
        cache = FeatureCache()
        cache.extract_all(dataset, verbose=True)

        # With disk persistence (recommended for iteration):
        cache = FeatureCache(cache_dir='./feature_cache')
        cache.extract_all(dataset)  # Saves to disk
        # Next run: features loaded from disk instantly!

        # Access features
        features = cache["run_20251226_110631"]
    """

    def __init__(self, config: FeatureConfig | None = None, cache_dir: str | None = None):
        self.config = config
        self.cache_dir = cache_dir
        self.extractor = FeatureExtractor(config)
        self._cache: dict[str, np.ndarray] = {}
        self._id_to_sim: dict[str, 'Simulation'] = {}

        # Create cache directory if specified
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            self._config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """Compute a hash of the config to detect changes."""
        import hashlib
        import json
        from dataclasses import asdict
        config_dict = asdict(self.config) if self.config else {}
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_cache_path(self, sim_id: str) -> str:
        """Get the cache file path for a simulation."""
        import os
        # Include config hash to invalidate cache when config changes
        filename = f"{sim_id}_{self._config_hash}.npy"
        return os.path.join(self.cache_dir, filename)

    def _load_from_disk(self, sim_id: str) -> np.ndarray | None:
        """Try to load features from disk cache."""
        if not self.cache_dir:
            return None
        import os
        path = self._get_cache_path(sim_id)
        if os.path.exists(path):
            return np.load(path)
        return None

    def _save_to_disk(self, sim_id: str, features: np.ndarray) -> None:
        """Save features to disk cache."""
        if not self.cache_dir:
            return
        path = self._get_cache_path(sim_id)
        np.save(path, features)

    def extract_all(
        self,
        dataset: 'Dataset',
        verbose: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        """
        Extract and cache features for all simulations in a dataset.

        If cache_dir was specified, features are loaded from disk if available,
        and newly extracted features are saved to disk.

        Args:
            dataset: Dataset object with loaded simulations
            verbose: Print progress
            n_jobs: Number of parallel workers (None = number of CPUs)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        # First, try to load from disk cache
        loaded_from_disk = 0
        to_process = []

        for ann in dataset.annotations:
            sim_id = ann.id
            if sim_id in self._cache:
                continue  # Already in memory

            # Try disk cache first
            disk_features = self._load_from_disk(sim_id)
            if disk_features is not None:
                self._cache[sim_id] = disk_features
                loaded_from_disk += 1
                continue

            # Need to extract - requires the simulation data
            sim = dataset.simulations[sim_id]
            self._id_to_sim[sim_id] = sim
            to_process.append((sim_id, sim))

        if loaded_from_disk > 0 and verbose:
            print(f"Loaded {loaded_from_disk} simulations from disk cache")

        if not to_process:
            if verbose and loaded_from_disk == 0:
                print("All features already cached")
            return

        total = len(to_process)
        if n_jobs is None:
            n_jobs = min(os.cpu_count() or 1, total)

        if verbose:
            print(f"Extracting features for {total} simulations using {n_jobs} workers...")

        def extract_one(item):
            sim_id, sim = item
            features = self.extractor.transform(sim)
            return sim_id, features

        if n_jobs == 1:
            # Sequential (for debugging)
            for i, (sim_id, sim) in enumerate(to_process):
                if verbose:
                    print(f"  {i+1}/{total}: {sim_id}")
                features = self.extractor.transform(sim)
                self._cache[sim_id] = features
                self._save_to_disk(sim_id, features)
        else:
            # Parallel extraction
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(extract_one, item): item[0] for item in to_process}
                done = 0
                for future in as_completed(futures):
                    sim_id, features = future.result()
                    self._cache[sim_id] = features
                    self._save_to_disk(sim_id, features)
                    done += 1
                    if verbose and done % 10 == 0:
                        print(f"  {done}/{total} done")

        if verbose:
            print(f"Cached features for {len(self._cache)} simulations")

    def extract_single(self, sim_id: str, simulation: 'Simulation') -> np.ndarray:
        """Extract and cache features for a single simulation."""
        if sim_id not in self._cache:
            # Try disk cache first
            disk_features = self._load_from_disk(sim_id)
            if disk_features is not None:
                self._cache[sim_id] = disk_features
            else:
                features = self.extractor.transform(simulation)
                self._cache[sim_id] = features
                self._save_to_disk(sim_id, features)
            self._id_to_sim[sim_id] = simulation
        return self._cache[sim_id]

    def load_from_disk(self, sim_ids: list[str], verbose: bool = True) -> int:
        """
        Load features from disk cache without needing the dataset.

        This is the fastest way to iterate: extract once, then load from disk.

        Args:
            sim_ids: List of simulation IDs to load
            verbose: Print progress

        Returns:
            Number of simulations successfully loaded
        """
        if not self.cache_dir:
            raise ValueError("No cache_dir specified. Create FeatureCache with cache_dir.")

        loaded = 0
        missing = []
        for sim_id in sim_ids:
            if sim_id in self._cache:
                loaded += 1
                continue
            features = self._load_from_disk(sim_id)
            if features is not None:
                self._cache[sim_id] = features
                loaded += 1
            else:
                missing.append(sim_id)

        if verbose:
            print(f"Loaded {loaded}/{len(sim_ids)} simulations from disk cache")
            if missing:
                print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        return loaded

    def get(self, sim_id: str) -> np.ndarray:
        """Get cached features by simulation ID."""
        if sim_id not in self._cache:
            # Try disk cache as fallback
            disk_features = self._load_from_disk(sim_id)
            if disk_features is not None:
                self._cache[sim_id] = disk_features
                return disk_features
            raise KeyError(f"No cached features for {sim_id}. Call extract_all() first.")
        return self._cache[sim_id]

    def __getitem__(self, sim_id: str) -> np.ndarray:
        return self.get(sim_id)

    def __contains__(self, sim_id: str) -> bool:
        if sim_id in self._cache:
            return True
        # Check disk cache
        disk_features = self._load_from_disk(sim_id)
        if disk_features is not None:
            self._cache[sim_id] = disk_features
            return True
        return False

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def feature_names(self) -> list[str]:
        return self.extractor.feature_names
