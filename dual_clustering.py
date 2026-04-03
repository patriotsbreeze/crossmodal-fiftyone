"""
dual_clustering.py

Assigns sensory cluster labels to visual and audio embedding matrices
using sklearn's HDBSCAN (automatic cluster count — no hardcoded k).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

# Maximum dimensions to keep before clustering.  Reduces the curse of
# dimensionality for small sample counts while preserving cluster structure.
_MAX_PCA_DIMS = 32


def _reduce_if_needed(
    matrix: NDArray[np.float32],
    max_dims: int = _MAX_PCA_DIMS,
) -> NDArray[np.float32]:
    """Apply PCA when the feature count exceeds *max_dims*."""
    n_samples, n_features = matrix.shape
    if n_features <= max_dims:
        return matrix
    n_components = min(max_dims, n_samples)
    return PCA(n_components=n_components).fit_transform(matrix).astype(np.float32)


def _cluster_and_label(
    matrix: NDArray[np.float32],
    prefix: str,
    min_cluster_size: int = 3,
) -> list[str]:
    """
    Run HDBSCAN on *matrix* and return string labels with the given prefix.

    A PCA step is applied first when embeddings are high-dimensional
    (>32 dims) to help HDBSCAN find density in small datasets.

    Args:
        matrix:           (n_samples, n_features) embedding matrix.
        prefix:           Label prefix — ``"V"`` for visual, ``"A"`` for audio.
        min_cluster_size: Minimum HDBSCAN cluster size (forwarded directly).

    Returns:
        A list of ``"<prefix>_<cluster_id>"`` strings, one per row.
        Noise points receive cluster id ``-1`` (e.g. ``"V_-1"``).
    """
    reduced = _reduce_if_needed(matrix)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels: NDArray[np.intp] = clusterer.fit_predict(reduced)
    return [f"{prefix}_{int(lbl)}" for lbl in labels]


def assign_sensory_clusters(
    visual_matrix: NDArray[np.float32],
    audio_matrix: NDArray[np.float32],
    *,
    min_cluster_size: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Cluster visual and audio embeddings independently with HDBSCAN.

    HDBSCAN automatically determines the optimal number of clusters for
    each modality — no *k* is hardcoded.

    Args:
        visual_matrix:    (n_visual, dim) array of visual embeddings.
        audio_matrix:     (n_audio, dim) array of audio embeddings.
        min_cluster_size: Forwarded to ``sklearn.cluster.HDBSCAN``.

    Returns:
        A tuple ``(visual_labels, audio_labels)`` where each element is a
        list of prefixed cluster-ID strings (e.g. ``["V_0", "V_1", …]``
        and ``["A_0", "A_1", …]``).
    """
    visual_labels = _cluster_and_label(
        visual_matrix, "V", min_cluster_size=min_cluster_size,
    )
    audio_labels = _cluster_and_label(
        audio_matrix, "A", min_cluster_size=min_cluster_size,
    )
    return visual_labels, audio_labels
