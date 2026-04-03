"""
tests/test_co_occurrence_math.py

TDD suite for co_occurrence_math.calculate_contextual_anomalies.
Verifies that P(A|V) crosstab probabilities produce the correct anomaly
scores: high-frequency pairs → near 0.0, rare pairs → near 1.0.
"""

from __future__ import annotations

import pytest

from co_occurrence_math import calculate_contextual_anomalies


# ── Helpers ───────────────────────────────────────────────────────────────────


def _repeat(pairs: list[tuple[str, str]], n: int) -> tuple[list[str], list[str]]:
    """Repeat a list of (visual, audio) pairs *n* times and unzip."""
    expanded = pairs * n
    vis = [p[0] for p in expanded]
    aud = [p[1] for p in expanded]
    return vis, aud


# ── Return contract ──────────────────────────────────────────────────────────


class TestReturnContract:

    def test_returns_list_of_floats(self) -> None:
        vis = ["V_0", "V_0", "V_1", "V_1"]
        aud = ["A_0", "A_0", "A_1", "A_1"]
        scores = calculate_contextual_anomalies(vis, aud)
        assert isinstance(scores, list)
        assert all(isinstance(s, float) for s in scores)

    def test_length_matches_input(self) -> None:
        vis = ["V_0"] * 10 + ["V_1"] * 5
        aud = ["A_0"] * 10 + ["A_1"] * 5
        scores = calculate_contextual_anomalies(vis, aud)
        assert len(scores) == 15

    def test_scores_between_0_and_1(self) -> None:
        vis = ["V_0", "V_0", "V_1", "V_1", "V_0"]
        aud = ["A_0", "A_0", "A_1", "A_1", "A_1"]
        scores = calculate_contextual_anomalies(vis, aud)
        for s in scores:
            assert 0.0 <= s <= 1.0


# ── Perfect correlation: every V always maps to the same A ────────────────────


class TestPerfectCorrelation:
    """When each visual cluster ALWAYS co-occurs with the same audio cluster,
    P(A|V) = 1.0 for every pair, so anomaly = 1 - 1.0 = 0.0."""

    def test_all_scores_zero(self) -> None:
        vis = ["V_0"] * 50 + ["V_1"] * 50
        aud = ["A_0"] * 50 + ["A_1"] * 50
        scores = calculate_contextual_anomalies(vis, aud)
        for s in scores:
            assert s == pytest.approx(0.0, abs=1e-9)

    def test_three_clusters_perfect(self) -> None:
        vis = ["V_0"] * 30 + ["V_1"] * 30 + ["V_2"] * 30
        aud = ["A_0"] * 30 + ["A_1"] * 30 + ["A_2"] * 30
        scores = calculate_contextual_anomalies(vis, aud)
        assert all(s == pytest.approx(0.0, abs=1e-9) for s in scores)


# ── Uniform distribution: every A equally likely for a given V ────────────────


class TestUniformDistribution:
    """When V_0 maps to A_0 and A_1 equally, P(A|V) = 0.5,
    so anomaly = 0.5 for every segment."""

    def test_two_audio_uniform(self) -> None:
        vis = ["V_0", "V_0"]
        aud = ["A_0", "A_1"]
        scores = calculate_contextual_anomalies(vis, aud)
        for s in scores:
            assert s == pytest.approx(0.5, abs=1e-9)

    def test_three_audio_uniform(self) -> None:
        vis = ["V_0"] * 30
        aud = (["A_0"] * 10) + (["A_1"] * 10) + (["A_2"] * 10)
        scores = calculate_contextual_anomalies(vis, aud)
        for s in scores:
            assert s == pytest.approx(1.0 - 1.0 / 3.0, abs=1e-9)


# ── Rare pair = high anomaly, common pair = low anomaly ───────────────────────


class TestRareVsCommon:
    """Core behavioural test: dominant pairs should score near 0,
    and the single rare pair should score near 1."""

    def test_single_rare_pair(self) -> None:
        # 99 segments: V_0 → A_0 (dominant)
        # 1 segment:   V_0 → A_1 (rare)
        vis = ["V_0"] * 100
        aud = ["A_0"] * 99 + ["A_1"]
        scores = calculate_contextual_anomalies(vis, aud)

        # Dominant pair: P(A_0|V_0) = 99/100, anomaly = 0.01
        for s in scores[:99]:
            assert s == pytest.approx(0.01, abs=1e-9)

        # Rare pair: P(A_1|V_0) = 1/100, anomaly = 0.99
        assert scores[99] == pytest.approx(0.99, abs=1e-9)

    def test_rare_pair_scores_higher_than_common(self) -> None:
        # V_0: 80× A_0 + 20× A_1
        # V_1: 90× A_2 + 10× A_3
        vis = ["V_0"] * 100 + ["V_1"] * 100
        aud = (
            ["A_0"] * 80 + ["A_1"] * 20
            + ["A_2"] * 90 + ["A_3"] * 10
        )
        scores = calculate_contextual_anomalies(vis, aud)

        common_v0 = scores[0]   # V_0→A_0, P=0.8, anomaly=0.2
        rare_v0 = scores[80]    # V_0→A_1, P=0.2, anomaly=0.8
        common_v1 = scores[100] # V_1→A_2, P=0.9, anomaly=0.1
        rare_v1 = scores[190]   # V_1→A_3, P=0.1, anomaly=0.9

        assert common_v0 < rare_v0
        assert common_v1 < rare_v1
        assert common_v0 == pytest.approx(0.2, abs=1e-9)
        assert rare_v0 == pytest.approx(0.8, abs=1e-9)
        assert common_v1 == pytest.approx(0.1, abs=1e-9)
        assert rare_v1 == pytest.approx(0.9, abs=1e-9)


# ── Multiple visual clusters, mixed distributions ─────────────────────────────


class TestMixedClusters:

    def test_independent_visual_rows(self) -> None:
        """Each visual cluster's P(A|V) is computed from its own row only."""
        # V_0: always A_0 → P(A_0|V_0)=1.0, anomaly=0.0
        # V_1: 50/50 A_0/A_1 → P=0.5, anomaly=0.5
        vis = ["V_0"] * 40 + ["V_1"] * 40
        aud = ["A_0"] * 40 + (["A_0"] * 20 + ["A_1"] * 20)
        scores = calculate_contextual_anomalies(vis, aud)

        # V_0 segments
        for s in scores[:40]:
            assert s == pytest.approx(0.0, abs=1e-9)

        # V_1 segments
        for s in scores[40:]:
            assert s == pytest.approx(0.5, abs=1e-9)


# ── Parametrized edge cases ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vis, aud, expected",
    [
        # Single element — P(A|V) = 1.0
        (["V_0"], ["A_0"], [0.0]),
        # Two identical pairs
        (["V_0", "V_0"], ["A_0", "A_0"], [0.0, 0.0]),
    ],
    ids=["single-element", "two-identical"],
)
def test_edge_cases(
    vis: list[str], aud: list[str], expected: list[float],
) -> None:
    scores = calculate_contextual_anomalies(vis, aud)
    for s, e in zip(scores, expected):
        assert s == pytest.approx(e, abs=1e-9)


# ── Noise labels (cluster -1 from HDBSCAN) ───────────────────────────────────


class TestNoiseLabels:
    """V_-1 / A_-1 from HDBSCAN noise points should be handled normally."""

    def test_noise_labels_treated_as_cluster(self) -> None:
        vis = ["V_-1"] * 10 + ["V_0"] * 10
        aud = ["A_0"] * 10 + ["A_-1"] * 10
        scores = calculate_contextual_anomalies(vis, aud)
        assert len(scores) == 20
        # Each V cluster maps to exactly one A cluster → P=1.0 → anomaly=0.0
        for s in scores:
            assert s == pytest.approx(0.0, abs=1e-9)
