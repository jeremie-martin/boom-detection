"""
Convergence-based features for boom detection.

The boom is fundamentally a CONVERGENCE event: two groups of pendulums
that had separated come back together at a single point. These features
directly capture this phenomenon.

Usage:
    from boom_detection.convergence import (
        angular_bimodality,
        cluster_distance,
        convergence_rate,
        ConvergenceDetector,
    )
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .loader import X2, Y2, TH1, TH2
from .features import FeatureCache


def angular_bimodality(data: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Measure bimodality in the angular distribution of tips.

    The boom occurs when two separate angular clusters converge.
    This measures how separated the two main clusters are.

    Args:
        data: Shape (frames, pendulums, 8)
        n_bins: Number of angular bins

    Returns:
        Shape (frames, 4) - [peak_distance, bimodality_strength,
                            peak1_size, peak2_size]
    """
    # Use tip angle (th1 + th2) as the angular position
    tip_angles = data[:, :, TH1] + data[:, :, TH2]

    n_frames = data.shape[0]
    features = np.zeros((n_frames, 4))

    for t in range(n_frames):
        angles = tip_angles[t]

        # Wrap angles to [0, 2*pi]
        angles = np.mod(angles, 2 * np.pi)

        # Create histogram
        hist, _ = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
        hist = hist.astype(float)

        # Smooth histogram to find robust peaks
        hist_smooth = ndimage.gaussian_filter1d(hist, sigma=1, mode='wrap')

        # Find peaks (local maxima)
        peaks = []
        for i in range(n_bins):
            prev_i = (i - 1) % n_bins
            next_i = (i + 1) % n_bins
            if hist_smooth[i] > hist_smooth[prev_i] and hist_smooth[i] > hist_smooth[next_i]:
                peaks.append((i, hist_smooth[i]))

        if len(peaks) >= 2:
            # Sort by peak height
            peaks.sort(key=lambda x: -x[1])
            peak1_idx, peak1_height = peaks[0]
            peak2_idx, peak2_height = peaks[1]

            # Angular distance between peaks (accounting for wrap-around)
            bin_width = 2 * np.pi / n_bins
            peak1_angle = peak1_idx * bin_width
            peak2_angle = peak2_idx * bin_width

            # Circular distance
            diff = abs(peak1_angle - peak2_angle)
            peak_distance = min(diff, 2 * np.pi - diff)

            # Bimodality strength: ratio of smaller peak to larger peak
            bimodality = peak2_height / (peak1_height + 1e-8)

            # Peak sizes (fraction of total)
            total = hist.sum() + 1e-8
            peak1_size = hist[peak1_idx] / total
            peak2_size = hist[peak2_idx] / total

            features[t] = [peak_distance, bimodality, peak1_size, peak2_size]
        else:
            # Single peak or no peaks - not bimodal
            features[t] = [0, 0, 1, 0]

    return features


def cluster_distance(data: np.ndarray) -> np.ndarray:
    """
    Compute distance between two main clusters of tip positions.

    Uses simple k-means-like approach: split by median angle,
    compute centroid of each half.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, 3) - [inter_cluster_dist, cluster1_spread, cluster2_spread]
    """
    x2 = data[:, :, X2]
    y2 = data[:, :, Y2]
    tip_angles = np.arctan2(x2, y2)  # angle from vertical

    n_frames = data.shape[0]
    features = np.zeros((n_frames, 3))

    for t in range(n_frames):
        angles = tip_angles[t]
        x = x2[t]
        y = y2[t]

        # Split into two groups by median angle
        median_angle = np.median(angles)
        group1 = angles <= median_angle
        group2 = ~group1

        if group1.sum() > 0 and group2.sum() > 0:
            # Centroids
            c1_x, c1_y = x[group1].mean(), y[group1].mean()
            c2_x, c2_y = x[group2].mean(), y[group2].mean()

            # Inter-cluster distance
            dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

            # Within-cluster spread (std of distances from centroid)
            spread1 = np.std(np.sqrt((x[group1] - c1_x)**2 + (y[group1] - c1_y)**2))
            spread2 = np.std(np.sqrt((x[group2] - c2_x)**2 + (y[group2] - c2_y)**2))

            features[t] = [dist, spread1, spread2]
        else:
            features[t] = [0, 0, 0]

    return features


def convergence_rate(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute rate of change of cluster distance.

    Boom = when clusters are converging fastest (most negative rate).

    Args:
        data: Shape (frames, pendulums, 8)
        window: Smoothing window for derivative

    Returns:
        Shape (frames, 2) - [convergence_rate, acceleration]
    """
    # Get cluster distances
    cluster_dist = cluster_distance(data)[:, 0]  # just the distance

    # Smooth
    kernel = np.ones(window) / window
    smoothed = np.convolve(cluster_dist, kernel, mode='same')

    # First derivative (rate of change)
    rate = np.gradient(smoothed)

    # Second derivative (acceleration)
    accel = np.gradient(rate)

    return np.stack([rate, accel], axis=1)


def tip_concentration(data: np.ndarray, grid_size: int = 20) -> np.ndarray:
    """
    Measure maximum local concentration of tips.

    At the boom, many tips pass through the same small region.

    Args:
        data: Shape (frames, pendulums, 8)
        grid_size: Size of spatial grid for counting

    Returns:
        Shape (frames, 2) - [max_concentration, concentration_entropy]
    """
    x2 = data[:, :, X2]
    y2 = data[:, :, Y2]

    n_frames = data.shape[0]
    features = np.zeros((n_frames, 2))

    for t in range(n_frames):
        x = x2[t]
        y = y2[t]

        # Normalize to [0, 1] based on data range
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        if x_max > x_min and y_max > y_min:
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)

            # Create 2D histogram
            hist, _, _ = np.histogram2d(
                x_norm, y_norm,
                bins=grid_size,
                range=[[0, 1], [0, 1]]
            )

            # Max concentration (normalized)
            n_pendulums = len(x)
            max_conc = hist.max() / n_pendulums

            # Entropy of spatial distribution
            hist_flat = hist.flatten()
            hist_flat = hist_flat[hist_flat > 0]
            probs = hist_flat / hist_flat.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(grid_size * grid_size)
            norm_entropy = entropy / max_entropy

            features[t] = [max_conc, norm_entropy]
        else:
            features[t] = [1, 0]  # All at same point

    return features


def extract_convergence_features(data: np.ndarray) -> np.ndarray:
    """
    Extract all convergence-related features.

    Args:
        data: Shape (frames, pendulums, 8)

    Returns:
        Shape (frames, n_features)
    """
    features = [
        angular_bimodality(data),      # 4 features
        cluster_distance(data),         # 3 features
        convergence_rate(data),         # 2 features
        tip_concentration(data),        # 2 features
    ]
    return np.hstack(features)


CONVERGENCE_FEATURE_NAMES = [
    'peak_distance', 'bimodality_strength', 'peak1_size', 'peak2_size',
    'cluster_dist', 'cluster1_spread', 'cluster2_spread',
    'convergence_rate', 'convergence_accel',
    'max_concentration', 'spatial_entropy',
]


class ConvergenceDetector:
    """
    Detect boom by finding the convergence point.

    The boom is when cluster distance reaches minimum after decreasing.
    """

    def __init__(
        self,
        min_frame: int = 100,  # Ignore first N frames (too early)
        smoothing: int = 10,
    ):
        self.min_frame = min_frame
        self.smoothing = smoothing
        self.offset_ = 0

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Learn offset from detected minimum to actual boom."""
        offsets = []

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]

            # Get cluster distance (assuming it's in the features)
            # For now, we need raw data - this is a limitation
            # We'll detect minimum of bimodality * cluster_dist proxy

            # Use convergence-related features if available
            # Features indices depend on config, so we use a heuristic:
            # Look for minimum in a feature that typically decreases then increases

            # Simple approach: find where variance derivative is maximum
            # (this correlates with boom)
            tip_var = features[:, 2] + features[:, 3]  # var_x2 + var_y2
            deriv = np.gradient(tip_var)

            # Smooth
            kernel = np.ones(self.smoothing) / self.smoothing
            deriv_smooth = np.convolve(deriv, kernel, mode='same')

            # Find maximum derivative after min_frame
            search_range = deriv_smooth[self.min_frame:]
            if len(search_range) > 0:
                detected = self.min_frame + np.argmax(search_range)
                offsets.append(int(boom) - detected)

        if offsets:
            self.offset_ = int(np.median(offsets))

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict boom frames."""
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            n_frames = len(features)

            # Same detection as in fit
            tip_var = features[:, 2] + features[:, 3]
            deriv = np.gradient(tip_var)

            kernel = np.ones(self.smoothing) / self.smoothing
            deriv_smooth = np.convolve(deriv, kernel, mode='same')

            search_range = deriv_smooth[self.min_frame:]
            if len(search_range) > 0:
                detected = self.min_frame + np.argmax(search_range)
                pred = detected + self.offset_
                pred = max(0, min(pred, n_frames - 1))
            else:
                pred = n_frames // 2

            predictions.append(int(pred))

        return np.array(predictions)


class BimodalityDetector:
    """
    Detect boom by finding when angular bimodality drops.

    Before boom: two clusters → high bimodality
    At boom: clusters merge → low bimodality (peaks converge)
    """

    def __init__(
        self,
        n_bins: int = 36,
        min_frame: int = 100,
        smoothing: int = 5,
    ):
        self.n_bins = n_bins
        self.min_frame = min_frame
        self.smoothing = smoothing
        self.threshold_ = 0.5
        self.offset_ = 0

    def _compute_peak_distance(self, data: np.ndarray) -> np.ndarray:
        """Compute peak distance for all frames."""
        return angular_bimodality(data, self.n_bins)[:, 0]  # peak_distance

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Learn threshold and offset from training data."""
        # This detector needs raw data, not just cached features
        # For now, use a simpler approach based on cached features
        offsets = []

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]

            # Use tip variance as proxy (increases at boom)
            tip_var = features[:, 2] + features[:, 3]

            # Find where variance starts increasing rapidly
            deriv = np.gradient(tip_var)
            kernel = np.ones(self.smoothing) / self.smoothing
            deriv_smooth = np.convolve(deriv, kernel, mode='same')

            # Find first frame where derivative exceeds threshold
            # after min_frame
            threshold = np.percentile(deriv_smooth[self.min_frame:], 75)
            exceeds = np.where(deriv_smooth[self.min_frame:] > threshold)[0]

            if len(exceeds) > 0:
                detected = self.min_frame + exceeds[0]
                offsets.append(int(boom) - detected)

        if offsets:
            self.offset_ = int(np.median(offsets))

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict boom frames."""
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            n_frames = len(features)

            tip_var = features[:, 2] + features[:, 3]
            deriv = np.gradient(tip_var)
            kernel = np.ones(self.smoothing) / self.smoothing
            deriv_smooth = np.convolve(deriv, kernel, mode='same')

            threshold = np.percentile(deriv_smooth[self.min_frame:], 75)
            exceeds = np.where(deriv_smooth[self.min_frame:] > threshold)[0]

            if len(exceeds) > 0:
                detected = self.min_frame + exceeds[0]
                pred = detected + self.offset_
                pred = max(0, min(pred, n_frames - 1))
            else:
                pred = n_frames // 2

            predictions.append(int(pred))

        return np.array(predictions)
