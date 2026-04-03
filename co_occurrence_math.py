"""
co_occurrence_math.py

Computes contextual anomaly scores from paired visual/audio cluster labels
using a P(A|V) conditional-probability crosstab built with pandas.
"""

from __future__ import annotations

import pandas as pd


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
