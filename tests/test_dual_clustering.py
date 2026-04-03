"""
tests/test_dual_clustering.py

TDD test suite for dual_clustering.assign_sensory_clusters.
Uses synthetic numpy matrices that simulate large batches of 1024-d
visual and audio embeddings with known cluster structure.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from dual_clustering import assign_sensory_clusters

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_clustered_embeddings(
    n_clusters: int,
    samples_per_cluster: int,
    dim: int = 1024,
    spread: float = 0.5,
    seed: int = 42,
) -> NDArray[np.float32]:
    """
    Generate a matrix of embeddings with *n_clusters* well-separated
    Gaussian blobs so HDBSCAN can reliably recover them.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-10, 10, size=(n_clusters, dim)).astype(np.float32)
    blocks: list[NDArray[np.float32]] = []
    for centre in centres:
        noise = rng.normal(0, spread, size=(samples_per_cluster, dim))
        blocks.append((centre + noise).astype(np.float32))
    return np.vstack(blocks)


@pytest.fixture()
def three_cluster_visual() -> NDArray[np.float32]:
    """200 samples × 1024-d with 3 well-separated visual clusters."""
    return _make_clustered_embeddings(
        n_clusters=3, samples_per_cluster=200, seed=1
    )


@pytest.fixture()
def two_cluster_audio() -> NDArray[np.float32]:
    """200 samples × 1024-d with 2 well-separated audio clusters."""
    return _make_clustered_embeddings(
        n_clusters=2, samples_per_cluster=200, seed=2
    )


@pytest.fixture()
def large_visual() -> NDArray[np.float32]:
    """1 000 samples × 1024-d with 5 clusters."""
    return _make_clustered_embeddings(
        n_clusters=5, samples_per_cluster=200, seed=10
    )


@pytest.fixture()
def large_audio() -> NDArray[np.float32]:
    """1 000 samples × 1024-d with 4 clusters."""
    return _make_clustered_embeddings(
        n_clusters=4, samples_per_cluster=250, seed=20
    )


# ── Return shape & type ──────────────────────────────────────────────────────


class TestReturnShape:
    """Basic contract: two lists of labels, one per input row."""

    def test_returns_two_lists(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        v_labels, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        assert isinstance(v_labels, list)
        assert isinstance(a_labels, list)

    def test_label_count_matches_input_rows(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        v_labels, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        assert len(v_labels) == three_cluster_visual.shape[0]
        assert len(a_labels) == two_cluster_audio.shape[0]

    def test_labels_are_strings(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        v_labels, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        assert all(isinstance(lbl, str) for lbl in v_labels)
        assert all(isinstance(lbl, str) for lbl in a_labels)


# ── Label format ──────────────────────────────────────────────────────────────


class TestLabelFormat:
    """Visual labels must start with 'V_', audio labels with 'A_'."""

    def test_visual_prefix(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        v_labels, _ = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        for lbl in v_labels:
            assert lbl.startswith("V_"), f"Visual label '{lbl}' missing V_ prefix"

    def test_audio_prefix(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        _, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        for lbl in a_labels:
            assert lbl.startswith("A_"), f"Audio label '{lbl}' missing A_ prefix"

    def test_noise_label_format(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        """HDBSCAN may label outliers as -1; those should become V_-1 / A_-1."""
        v_labels, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        for lbl in v_labels + a_labels:
            prefix, _, suffix = lbl.partition("_")
            assert prefix in ("V", "A")
            # Suffix must be an integer (possibly negative for noise).
            int(suffix)  # raises ValueError if not a valid int


# ── Cluster quality ───────────────────────────────────────────────────────────


class TestClusterQuality:
    """With well-separated blobs HDBSCAN should find the right cluster count."""

    def test_visual_discovers_expected_clusters(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        v_labels, _ = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        unique = {lbl for lbl in v_labels if not lbl.endswith("_-1")}
        # HDBSCAN should find at least 2 of the 3 planted clusters.
        assert len(unique) >= 2

    def test_audio_discovers_expected_clusters(
        self,
        three_cluster_visual: NDArray[np.float32],
        two_cluster_audio: NDArray[np.float32],
    ) -> None:
        _, a_labels = assign_sensory_clusters(
            three_cluster_visual, two_cluster_audio,
        )
        unique = {lbl for lbl in a_labels if not lbl.endswith("_-1")}
        assert len(unique) >= 2

    def test_large_matrices_cluster_correctly(
        self,
        large_visual: NDArray[np.float32],
        large_audio: NDArray[np.float32],
    ) -> None:
        v_labels, a_labels = assign_sensory_clusters(
            large_visual, large_audio,
        )
        v_unique = {lbl for lbl in v_labels if not lbl.endswith("_-1")}
        a_unique = {lbl for lbl in a_labels if not lbl.endswith("_-1")}
        assert len(v_unique) >= 3
        assert len(a_unique) >= 3


# ── Matrices with different row counts ────────────────────────────────────────


class TestUnequalRows:
    """Visual and audio matrices may have different sample counts."""

    def test_different_row_counts(self) -> None:
        vis = _make_clustered_embeddings(2, 150, seed=99)
        aud = _make_clustered_embeddings(3, 100, seed=100)
        v_labels, a_labels = assign_sensory_clusters(vis, aud)
        assert len(v_labels) == 300
        assert len(a_labels) == 300


# ── Parametrized edge cases ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_vis, n_aud",
    [(50, 50), (500, 200), (200, 500)],
    ids=["small-equal", "vis-heavy", "aud-heavy"],
)
def test_various_sizes(n_vis: int, n_aud: int) -> None:
    vis = _make_clustered_embeddings(2, n_vis // 2, seed=77)
    aud = _make_clustered_embeddings(2, n_aud // 2, seed=88)
    v_labels, a_labels = assign_sensory_clusters(vis, aud)
    assert len(v_labels) == n_vis
    assert len(a_labels) == n_aud
