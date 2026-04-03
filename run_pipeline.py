"""
run_pipeline.py

End-to-end pipeline: download EPIC-Kitchens samples, embed with Twelve Labs,
cluster with HDBSCAN, score contextual anomalies, and launch the FiftyOne app.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import fiftyone as fo
from dotenv import load_dotenv

from ingest_and_embed import (
    download_epic_kitchens_samples,
    load_into_fiftyone,
    segment_into_clips,
    initialize_embedding_fields,
    embed_clips_with_twelvelabs,
)
from dual_clustering import assign_sensory_clusters
from co_occurrence_math import calculate_contextual_anomalies

load_dotenv()


def main() -> None:
    # ── 1. Ingest & embed ─────────────────────────────────────────────────
    video_paths = download_epic_kitchens_samples()
    dataset = load_into_fiftyone(video_paths)
    clips = segment_into_clips(dataset)
    initialize_embedding_fields(clips)
    embed_clips_with_twelvelabs(clips, dataset)

    # ── 2. Collect embeddings into matrices ───────────────────────────────
    # Only keep clips that received real embeddings from the API.
    embedded_clip_ids: list[str] = []
    visual_list: list[list[float]] = []
    audio_list: list[list[float]] = []
    for clip in clips:
        vis = clip["visual_embedding"]
        aud = clip["audio_embedding"]
        if (
            isinstance(vis, list) and len(vis) > 0
            and isinstance(aud, list) and len(aud) > 0
            and any(v != 0.0 for v in vis)
        ):
            visual_list.append(vis)
            audio_list.append(aud)
            embedded_clip_ids.append(clip.id)

    print(f"[pipeline] {len(embedded_clip_ids)}/{len(clips)} clips have embeddings.")
    if not embedded_clip_ids:
        raise RuntimeError("No clips received embeddings — cannot continue.")

    visual_matrix: NDArray[np.float32] = np.array(visual_list, dtype=np.float32)
    audio_matrix: NDArray[np.float32] = np.array(audio_list, dtype=np.float32)

    # ── 3. Cluster ────────────────────────────────────────────────────────
    visual_labels, audio_labels = assign_sensory_clusters(
        visual_matrix, audio_matrix,
    )

    # ── 4. Score anomalies ────────────────────────────────────────────────
    scores = calculate_contextual_anomalies(visual_labels, audio_labels)

    # ── 5. Write results back to FiftyOne ─────────────────────────────────
    label_map = {
        cid: (v, a, s)
        for cid, v, a, s in zip(embedded_clip_ids, visual_labels, audio_labels, scores)
    }
    for clip in clips.iter_samples(autosave=True):
        if clip.id in label_map:
            v_lbl, a_lbl, score = label_map[clip.id]
            clip["visual_cluster"] = v_lbl
            clip["audio_cluster"] = a_lbl
            clip["contextual_anomaly_score"] = score

    # ── 6. Launch app sorted by most anomalous ────────────────────────────
    print("\n[pipeline] Done. Launching FiftyOne app …")
    print("[pipeline] Clips sorted by contextual_anomaly_score (highest first).\n")

    anomalous_view = clips.sort_by("contextual_anomaly_score", reverse=True)
    session = fo.launch_app(view=anomalous_view)
    session.wait()


if __name__ == "__main__":
    main()
