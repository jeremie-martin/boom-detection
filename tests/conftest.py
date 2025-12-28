"""
Shared fixtures for boom detection tests.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_simulation_data():
    """Create sample simulation data for testing.

    Returns:
        np.ndarray: Shape (frames=100, pendulums=50, features=8)
    """
    np.random.seed(42)
    frames = 100
    pendulums = 50
    # 8 features: x1, y1, x2, y2, th1, th2, w1, w2
    data = np.random.randn(frames, pendulums, 8).astype(np.float32)
    return data


@pytest.fixture
def sample_features():
    """Create sample extracted features for testing.

    Returns:
        np.ndarray: Shape (frames=100, n_features=183)
    """
    np.random.seed(42)
    frames = 100
    n_features = 183  # Default feature count
    return np.random.randn(frames, n_features).astype(np.float32)


@pytest.fixture
def sample_boom_frame():
    """Sample boom frame for a 100-frame simulation."""
    return 50


@pytest.fixture
def sample_quality():
    """Sample quality score."""
    return 0.7


class MockSimulation:
    """Mock simulation object for testing."""

    def __init__(self, data: np.ndarray, sim_id: str = "test_sim"):
        self.data = data
        self.id = sim_id


class MockCache:
    """Mock feature cache for testing."""

    def __init__(self, features_dict: dict[str, np.ndarray]):
        self._cache = features_dict

    def __getitem__(self, sim_id: str) -> np.ndarray:
        return self._cache[sim_id]

    def __contains__(self, sim_id: str) -> bool:
        return sim_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)


@pytest.fixture
def mock_simulation(sample_simulation_data):
    """Create a mock simulation object."""
    return MockSimulation(sample_simulation_data)


@pytest.fixture
def mock_cache(sample_features):
    """Create a mock cache with one simulation."""
    return MockCache({"test_sim": sample_features})


@pytest.fixture
def mock_cache_multi(sample_features):
    """Create a mock cache with multiple simulations."""
    np.random.seed(42)
    features = {}
    for i in range(10):
        # Slightly different features for each simulation
        features[f"sim_{i}"] = sample_features + np.random.randn(*sample_features.shape) * 0.1
    return MockCache(features)
