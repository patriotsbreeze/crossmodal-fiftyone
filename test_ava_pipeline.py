"""
test_ava_pipeline.py

Single-video anomaly detection test using trimmed_AVA_data.mp4.

The video has no audio in the first half and engine sounds in the second half.
We expect the engine-sound segments to score high as anomalies.

Scoring strategy: cosine distance from the audio embedding baseline established
by the first few (silent) segments. Co-occurrence scoring is useless for a
single video because every visual cluster maps to exactly one audio cluster
(all clips in the same Twelve Labs API window share identical embeddings).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import fiftyone as fo
from dotenv import load_dotenv

from ingest_and_embed import (
    load_into_fiftyone,
    segment_into_clips,
    initialize_embedding_fields,
    embed_clips_with_twelvelabs,
)
from co_occurrence_math import score_audio_drift

load_dotenv()

VIDEO_PATH = Path("data/epic_kitchens/trimmed_AVA_data.mp4")
ANOMALY_THRESHOLD = 0.9
# Number of clips at the start of the video to treat as the "silent baseline".
BASELINE_CLIP_COUNT = 4


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print(f"\n[test] Using video: {VIDEO_PATH}\n")

    # 1. Load & segment.
    dataset = load_into_fiftyone([VIDEO_PATH], dataset_name="ava_anomaly_test")
    clips = segment_into_clips(dataset)
    initialize_embedding_fields(clips)

    # 2. Embed via Twelve Labs.
    embed_clips_with_twelvelabs(clips, dataset)

    # 3. Collect embeddings in temporal order.
    embedded_ids: list[str] = []
    audio_list: list[list[float]] = []
    supports: list[list[int]] = []

    for clip in clips:
        aud = clip["audio_embedding"]
        if isinstance(aud, list) and len(aud) > 0 and any(v != 0.0 for v in aud):
            audio_list.append(aud)
            embedded_ids.append(clip.id)
            supports.append(clip.support)

    n = len(embedded_ids)
    print(f"\n[test] {n}/{len(clips)} clips have audio embeddings.\n")
    if not embedded_ids:
        raise RuntimeError("No audio embeddings returned — cannot score anomalies.")

    audio_matrix: NDArray[np.float32] = np.array(audio_list, dtype=np.float32)

    # 4. Score by drift from the first BASELINE_CLIP_COUNT clips.
    baseline_count = min(BASELINE_CLIP_COUNT, n)
    scores = score_audio_drift(audio_matrix, baseline_count)

    fps = dataset.first().metadata.frame_rate

    # 5. Write scores back onto TemporalDetection labels.
    score_map = {cid: score for cid, score in zip(embedded_ids, scores)}

    clip_iter = iter(clips)
    for sample in dataset.iter_samples(autosave=True):
        detections = sample["segments"].detections
        for det in detections:
            clip = next(clip_iter)
            if clip.id in score_map:
                det["audio_anomaly_score"] = score_map[clip.id]
        sample["segments"] = fo.TemporalDetections(detections=detections)

    # 6. Print results table.
    print(f"\n[test] Baseline: first {baseline_count} clips (expected: silence)")
    print(f"[test] Anomaly threshold: {ANOMALY_THRESHOLD}\n")
    print("=" * 60)
    print(f"{'Start':>8}  {'End':>8}  {'Score':>8}  {'Flag'}")
    print("=" * 60)

    anomaly_count = 0
    for cid, score, support in zip(embedded_ids, scores, supports):
        start_sec = (support[0] - 1) / fps
        end_sec = support[1] / fps
        flag = "  <<< ANOMALY" if score >= ANOMALY_THRESHOLD else ""
        if score >= ANOMALY_THRESHOLD:
            anomaly_count += 1
        print(f"{start_sec:>7.1f}s  {end_sec:>7.1f}s  {score:>8.3f}{flag}")

    print("=" * 60)
    print(f"\n[result] {anomaly_count}/{n} segments flagged "
          f"(threshold={ANOMALY_THRESHOLD}).\n")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
