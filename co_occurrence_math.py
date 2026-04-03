"""
co_occurrence_math.py

Computes contextual anomaly scores from visual/audio embeddings or cluster labels.

Two strategies are provided:
  - score_audio_drift: temporal drift from a silent baseline (primary, single-video).
  - calculate_contextual_anomalies: co-occurrence P(A|V) (multi-video context).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def score_audio_drift(
    audio_matrix: NDArray[np.float32],
    baseline_count: int = 4,
) -> list[float]:
    """
    Score each segment by how far its audio embedding has drifted from the
    silent baseline established by the first *baseline_count* segments.

    Uses cosine distance so the result is independent of embedding magnitude.
    Scores are normalised to ``[0.0, 1.0]`` relative to the maximum observed
    drift; a score near **1.0** means the audio changed maximally from the
    baseline (i.e. anomalous).

    Args:
        audio_matrix:   ``(n_segments, dim)`` array of audio embeddings in
                        **temporal order** (first row = earliest clip).
        baseline_count: Number of leading clips to treat as the silent
                        baseline.  Clamped to ``len(audio_matrix)``.

    Returns:
        A list of float anomaly scores in ``[0.0, 1.0]``, one per segment.
    """
    n = len(audio_matrix)
    bc = min(baseline_count, n)
    baseline: NDArray[np.float32] = audio_matrix[:bc].mean(axis=0)

    baseline_norm = np.linalg.norm(baseline)
    row_norms = np.linalg.norm(audio_matrix, axis=1)

    # Cosine similarity of each row against the baseline.
    if baseline_norm == 0:
        cosine_sim = np.zeros(n, dtype=np.float32)
    else:
        dots = audio_matrix @ baseline
        denom = row_norms * baseline_norm
        # Avoid division by zero for zero-norm rows.
        safe_denom = np.where(denom == 0, 1.0, denom)
        cosine_sim = np.where(denom == 0, 0.0, dots / safe_denom)

    raw_dist = (1.0 - cosine_sim).astype(np.float32)
    d_max = raw_dist.max()
    if d_max == 0:
        return [0.0] * n
    return (raw_dist / d_max).tolist()


def _build_conditional_prob_matrix(
    visual_clusters: list[str],
    audio_clusters: list[str],
) -> pd.DataFrame:
    """
    Build a P(A|V) conditional-probability matrix via ``pandas.crosstab``.

    Rows are visual clusters, columns are audio clusters.  Each cell holds
    ``P(audio_cluster | visual_cluster)`` — i.e. counts are normalised
    per-row so each row sums to 1.0.

    Args:
        visual_clusters: Visual cluster label per segment (e.g. ``"V_0"``).
        audio_clusters:  Audio cluster label per segment (e.g. ``"A_1"``).

    Returns:
        A DataFrame indexed by visual cluster, columns by audio cluster,
        values are conditional probabilities.
    """
    ct = pd.crosstab(
        pd.Series(visual_clusters, name="visual"),
        pd.Series(audio_clusters, name="audio"),
    )
    return ct.div(ct.sum(axis=1), axis=0)


def calculate_contextual_anomalies(
    visual_clusters: list[str],
    audio_clusters: list[str],
) -> list[float]:
    """
    Score every segment by how surprising its (visual, audio) cluster pair
    is, using ``1 − P(A|V)`` as the anomaly score.

    A score near **0.0** means the audio cluster is the expected companion
    for that visual cluster (common pair).  A score near **1.0** means the
    audio cluster almost never appears with that visual cluster (rare /
    anomalous pair).

    Args:
        visual_clusters: One visual-cluster label per segment.
        audio_clusters:  One audio-cluster label per segment (same length).

    Returns:
        A list of float anomaly scores in ``[0.0, 1.0]``, one per segment.
    """
    prob_matrix = _build_conditional_prob_matrix(visual_clusters, audio_clusters)

    scores: list[float] = []
    for v, a in zip(visual_clusters, audio_clusters):
        p_a_given_v: float = float(prob_matrix.at[v, a])
        scores.append(1.0 - p_a_given_v)

    return scores
