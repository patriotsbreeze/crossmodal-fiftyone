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
from co_occurrence_math import score_audio_drift

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

    # ── 4. Score anomalies (temporal audio drift from silent baseline) ────
    scores = score_audio_drift(audio_matrix)

    ANOMALY_THRESHOLD = 0.9

    # ── 5. Write results onto TemporalDetection labels (persists) ──────────
    label_map = {
        cid: (v, a, s)
        for cid, v, a, s in zip(embedded_clip_ids, visual_labels, audio_labels, scores)
    }

    clip_iter = iter(clips)
    for sample in dataset.iter_samples(autosave=True):
        detections = sample["segments"].detections
        for det in detections:
            clip = next(clip_iter)
            if clip.id in label_map:
                v_lbl, a_lbl, score = label_map[clip.id]
                det["visual_cluster"] = v_lbl
                det["audio_cluster"] = a_lbl
                det["audio_anomaly_score"] = score
                det["is_anomaly"] = score >= ANOMALY_THRESHOLD
        sample["segments"] = fo.TemporalDetections(detections=detections)

    # ── 6. Print summary & launch app ─────────────────────────────────────
    print("\n[pipeline] Results written to segment labels.")
    print("[pipeline] Launching FiftyOne app …\n")

    # Print a table so results are visible even outside the app.
    clips = dataset.to_clips("segments")
    for clip in clips:
        det = clip["segments"]
        vc = getattr(det, "visual_cluster", "?")
        ac = getattr(det, "audio_cluster", "?")
        sc = getattr(det, "audio_anomaly_score", "?")
        flag = "  <<< ANOMALY" if isinstance(sc, float) and sc >= ANOMALY_THRESHOLD else ""
        print(f"  {clip.support}  V={vc}  A={ac}  score={sc:.3f}{flag}"
              if isinstance(sc, float) else
              f"  {clip.support}  V={vc}  A={ac}  score={sc}")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
