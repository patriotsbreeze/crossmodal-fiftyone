"""
run_pipeline.py

End-to-end pipeline: download EPIC-Kitchens samples, embed with Twelve Labs,
cluster with HDBSCAN, score cross-modal anomalies, and launch the FiftyOne app.
"""

from __future__ import annotations
from pathlib import Path
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

load_dotenv()


def score_crossmodal(
    visual_matrix: NDArray[np.float32],
    audio_matrix: NDArray[np.float32],
) -> list[float]:
    """
    Score each segment by cosine distance between its visual and audio embedding.
    Both embeddings live in Marengo's shared latent space so this is meaningful.

    Returns divergence scores in [0, 2] — higher means audio and visual
    are more semantically mismatched at that timestamp.
    """
    # Normalise rows to unit vectors
    v_norms = np.linalg.norm(visual_matrix, axis=1, keepdims=True)
    a_norms = np.linalg.norm(audio_matrix, axis=1, keepdims=True)

    v_safe = np.where(v_norms == 0, 1.0, v_norms)
    a_safe = np.where(a_norms == 0, 1.0, a_norms)

    v_norm = visual_matrix / v_safe
    a_norm = audio_matrix / a_safe

    # Cosine similarity per segment (dot product of unit vectors)
    cos_sim = np.sum(v_norm * a_norm, axis=1)

    # Divergence = 1 - similarity, range [0, 2]
    divergence = 1.0 - cos_sim

    return divergence.tolist()


def main() -> None:
    # ── 1. Ingest & embed ─────────────────────────────────────────────────
    # video_paths = download_epic_kitchens_samples()
    local_test_video = Path("trimmed_AVA_data.mp4")
    video_paths = [local_test_video.resolve()]
    dataset = load_into_fiftyone(video_paths)
    clips = segment_into_clips(dataset)
    initialize_embedding_fields(clips)
    embed_clips_with_twelvelabs(clips, dataset)

    # ── 2. Collect embeddings into matrices ───────────────────────────────
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
            and any(v != 0.0 for v in aud)
        ):
            visual_list.append(vis)
            audio_list.append(aud)
            embedded_clip_ids.append(clip.id)

    print(f"[pipeline] {len(embedded_clip_ids)}/{len(clips)} clips have both embeddings.")

    if not embedded_clip_ids:
        raise RuntimeError("No clips received both visual+audio embeddings — cannot score cross-modal anomalies.")

    visual_matrix: NDArray[np.float32] = np.array(visual_list, dtype=np.float32)
    audio_matrix: NDArray[np.float32] = np.array(audio_list, dtype=np.float32)

    # ── 3. Cluster ────────────────────────────────────────────────────────
    visual_labels, audio_labels = assign_sensory_clusters(
        visual_matrix, audio_matrix,
    )

    # ── 4. Score cross-modal divergence ───────────────────────────────────
    scores = score_crossmodal(visual_matrix, audio_matrix)

    scores_array = np.array(scores)
    ANOMALY_THRESHOLD = float(scores_array.mean() + 1.5 * scores_array.std())
    print(f"[pipeline] Dynamic threshold: {ANOMALY_THRESHOLD:.4f} "
          f"(mean={scores_array.mean():.4f}, std={scores_array.std():.4f})")

    # ── 5. Write results onto TemporalDetection labels ────────────────────
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
                det.label = "ANOMALY" if score >= ANOMALY_THRESHOLD else "normal"
                det["visual_cluster"] = v_lbl
                det["audio_cluster"] = a_lbl
                det["crossmodal_divergence"] = score
        sample["segments"] = fo.TemporalDetections(detections=detections)

    # ── 6. Print summary & launch app ─────────────────────────────────────
    print("\n[pipeline] Results written to segment labels.")
    print("[pipeline] Launching FiftyOne app …\n")

    fps = dataset.first().metadata.frame_rate
    clips = dataset.to_clips("segments")

    print(f"{'Start':>8}  {'End':>8}  {'V_cluster':>10}  {'A_cluster':>10}  {'divergence':>12}  flag")
    print("=" * 70)
    for clip in clips:
        support = clip.support
        start_sec = (support[0] - 1) / fps
        end_sec = support[1] / fps

        # Fields are stored on the parent sample's TemporalDetection
        # that matches this clip's support window
        vc, ac, sc = "?", "?", None
        try:
            parent = dataset[clip.sample_id]
            for det in parent["segments"].detections:
                if det.support == support:
                    vc = det.get_field("visual_cluster") or "?"
                    ac = det.get_field("audio_cluster") or "?"
                    sc = det.get_field("crossmodal_divergence")
                    break
        except Exception:
            pass

        flag = "  <<< ANOMALY" if isinstance(sc, float) and sc >= ANOMALY_THRESHOLD else ""
        if isinstance(sc, float):
            print(f"{start_sec:>7.1f}s  {end_sec:>7.1f}s  {vc:>10}  {ac:>10}  {sc:>12.4f}{flag}")
        else:
            print(f"{start_sec:>7.1f}s  {end_sec:>7.1f}s  {vc:>10}  {ac:>10}  {'N/A':>12}")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
