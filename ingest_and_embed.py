"""
ingest_and_embed.py

Downloads EPIC-Kitchens sample videos, loads them into FiftyOne, segments them
into 3-second overlapping clips, and embeds each clip via the Twelve Labs
Embed API (marengo3.0) with both visual and audio scopes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import fiftyone as fo
from huggingface_hub import hf_hub_download
from twelvelabs import TwelveLabs
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SAMPLE_CLIPS = 4  # Hard cap — do not remove
CLIP_DURATION_SEC = 3.0
CLIP_STRIDE_SEC = 1.5  # 50 % overlap
EMBEDDING_DIM = 512  # marengo3.0 produces 512-d vectors
EMBED_MODEL = "marengo3.0"

HF_REPO_ID = "a1raman/epic_kitchens_100"
HF_REPO_TYPE = "dataset"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Pre-selected smallest files in the repo (~40-53 MB each instead of ~6 GB).
SMALLEST_VIDEOS = [
    "P03/videos/P03_15.MP4",  # ~40 MB
    "P03/videos/P03_26.MP4",  # ~43 MB
    "P06/videos/P06_02.MP4",  # ~53 MB
]

# Extra local videos to include alongside the HF downloads.
LOCAL_VIDEOS = [
    "trimmed_AVA_data.mp4",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class SegmentEmbedding:
    """Holds paired visual/audio embeddings for one API-returned segment."""

    start_sec: float
    end_sec: float
    visual: NDArray[np.float32] | None = field(default=None)
    audio: NDArray[np.float32] | None = field(default=None)
    description: str | None = field(default=None)


# ── Download helpers ──────────────────────────────────────────────────────────


def _remux_clean(src: Path) -> Path:
    """
    Re-mux a video keeping only the first video and audio streams.

    EPIC-Kitchens files contain extra data streams that can cause the
    Twelve Labs API to reject them as ``video_file_broken``.  Remuxing
    with ffmpeg strips those streams without re-encoding.

    Returns:
        Path to the cleaned file (replaces the original).
    """
    import subprocess

    tmp = src.with_suffix(".clean.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c", "copy", "-dn", "-map_metadata", "-1",
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    tmp.replace(src)
    return src


def download_epic_kitchens_samples(
    output_dir: str | Path = "data/epic_kitchens",
    max_clips: int = MAX_SAMPLE_CLIPS,
) -> list[Path]:
    """
    Download up to *max_clips* short EPIC-Kitchens videos from Hugging Face.

    The repo layout is ``PXX/videos/PXX_YY.MP4``.  Files are downloaded
    into *output_dir* preserving the repo structure via ``local_dir``,
    then remuxed to strip extra data streams that break the Twelve Labs API.

    Returns:
        Paths to the downloaded video files (at most MAX_SAMPLE_CLIPS).
    """
    max_clips = min(max_clips, MAX_SAMPLE_CLIPS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-use cached downloads — scan recursively for video files.
    existing = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(existing) >= max_clips:
        print(f"[download] {len(existing)} cached clips found — skipping.")
        return existing[:max_clips]

    paths: list[Path] = []

    # Include local videos first (already clean — no remux needed).
    for local_name in LOCAL_VIDEOS:
        if len(paths) >= max_clips:
            break
        local = output_dir / local_name
        if local.exists():
            print(f"[download] Local file: {local}")
            paths.append(local)

    # Fill remaining slots from HuggingFace.
    for remote in SMALLEST_VIDEOS:
        if len(paths) >= max_clips:
            break
        local = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=remote,
                local_dir=str(output_dir),
            )
        )
        print(f"[download] {remote} → {local}")
        print(f"[download] Remuxing {local.name} (strip extra streams) …")
        _remux_clean(local)
        paths.append(local)

    print(f"[download] {len(paths)} clip(s) ready in {output_dir}")
    return paths


# ── FiftyOne dataset ──────────────────────────────────────────────────────────


def load_into_fiftyone(
    video_paths: list[Path],
    dataset_name: str = "epic_kitchens_embedded",
) -> fo.Dataset:
    """Create (or replace) a FiftyOne video dataset from local paths."""
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.media_type = "video"
    dataset.add_samples([fo.Sample(filepath=str(p.resolve())) for p in video_paths])
    dataset.compute_metadata()

    print(f"[fiftyone] Dataset '{dataset_name}' — {len(dataset)} sample(s)")
    return dataset


# ── Clip segmentation ────────────────────────────────────────────────────────


def segment_into_clips(
    dataset: fo.Dataset,
    clip_duration: float = CLIP_DURATION_SEC,
    stride: float = CLIP_STRIDE_SEC,
) -> fo.DatasetView:
    """
    Segment every video into fixed-length overlapping clips using
    ``TemporalDetections`` and return a ``ClipsView``.

    Args:
        dataset:       A FiftyOne video dataset with computed metadata.
        clip_duration: Length of each clip in seconds.
        stride:        Step between successive clip starts (< clip_duration
                       for overlap).

    Returns:
        A ``fo.ClipsView`` backed by temporal detections on each sample.
    """
    for sample in dataset.iter_samples(autosave=True):
        fps: float = sample.metadata.frame_rate
        total_frames: int = sample.metadata.total_frame_count
        duration_sec: float = total_frames / fps

        detections: list[fo.TemporalDetection] = []
        start_sec = 0.0
        while start_sec + clip_duration <= duration_sec:
            end_sec = start_sec + clip_duration
            # FiftyOne frames are 1-indexed.
            start_frame = int(start_sec * fps) + 1
            end_frame = int(end_sec * fps)
            detections.append(
                fo.TemporalDetection(
                    label="segment",
                    support=[start_frame, end_frame],
                )
            )
            start_sec += stride

        sample["segments"] = fo.TemporalDetections(detections=detections)

    clips_view = dataset.to_clips("segments")
    print(f"[clips] {len(clips_view)} clip(s) created "
          f"({clip_duration}s duration, {stride}s stride)")
    return clips_view


# ── Schema initialisation ────────────────────────────────────────────────────


def initialize_embedding_fields(clips_view: fo.DatasetView) -> None:
    """
    Register analysis fields on the clips schema using typed
    defaults so FiftyOne can infer the field type.

    Fields initialised:
        - ``audio_embedding``  (list[float])
        - ``visual_embedding`` (list[float])
        - ``audio_cluster``    (str)
        - ``visual_cluster``   (str)
        - ``contextual_anomaly_score`` (float)
        - ``audio_cluster_name`` (str)
        - ``visual_cluster_name`` (str)
        - ``contextual_anomaly_level`` (str)
        - ``segment_description`` (str)
        - ``contextual_label`` (str)
    """
    # Set typed defaults on the first clip so FiftyOne creates the schema,
    # then clear them back out on all clips.
    first = clips_view.first()
    first["audio_embedding"] = [0.0] * EMBEDDING_DIM
    first["visual_embedding"] = [0.0] * EMBEDDING_DIM
    first["audio_cluster"] = ""
    first["visual_cluster"] = ""
    first["contextual_anomaly_score"] = 0.0
    first["audio_cluster_name"] = ""
    first["visual_cluster_name"] = ""
    first["contextual_anomaly_level"] = ""
    first["segment_description"] = ""
    first["contextual_label"] = ""
    first.save()

    print("[schema] Initialised analysis fields on clips.")


# ── Twelve Labs embedding ────────────────────────────────────────────────────


def _compute_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    """Return the temporal overlap (in seconds) between two intervals."""
    return max(0.0, min(e1, e2) - max(s1, s2))


def _parse_segment_embeddings(
    task_result: object,
) -> list[SegmentEmbedding]:
    """
    Group the raw API segments (which arrive as separate visual-text and
    audio entries) into ``SegmentEmbedding`` objects keyed by time window.
    """
    buckets: dict[tuple[float, float], SegmentEmbedding] = {}

    for seg in task_result.video_embedding.segments:
        key = (seg.start_offset_sec, seg.end_offset_sec)
        if key not in buckets:
            buckets[key] = SegmentEmbedding(start_sec=key[0], end_sec=key[1])

        option = getattr(seg, "embedding_option", "")

        match option:
            case "visual" | "visual-text":
                vector = getattr(seg, "float_", None)
                if vector is None:
                    continue
                buckets[key].visual = np.asarray(vector, dtype=np.float32)
            case "audio":
                vector = getattr(seg, "float_", None)
                if vector is None:
                    continue
                buckets[key].audio = np.asarray(vector, dtype=np.float32)
            case "transcription":
                text = (
                    getattr(seg, "text", None)
                    or getattr(seg, "transcription", None)
                    or getattr(seg, "caption", None)
                )
                if not text and hasattr(seg, "model_extra"):
                    extra = seg.model_extra or {}
                    for key_name in ("text", "transcription", "caption", "content"):
                        value = extra.get(key_name)
                        if isinstance(value, str) and value.strip():
                            text = value.strip()
                            break
                if isinstance(text, str) and text.strip():
                    buckets[key].description = text.strip()

    return sorted(buckets.values(), key=lambda s: s.start_sec)


def _best_match(
    api_segments: list[SegmentEmbedding],
    clip_start: float,
    clip_end: float,
) -> SegmentEmbedding | None:
    """Return the API segment with the greatest overlap to the clip window."""
    best: SegmentEmbedding | None = None
    best_overlap = 0.0
    for seg in api_segments:
        ov = _compute_overlap(clip_start, clip_end, seg.start_sec, seg.end_sec)
        if ov > best_overlap:
            best_overlap = ov
            best = seg
    return best


def embed_clips_with_twelvelabs(
    clips_view: fo.DatasetView,
    dataset: fo.Dataset,
    *,
    api_key: str | None = None,
) -> None:
    """
    For each source video, call the Twelve Labs Embed API once (Marengo,
    visual + audio scopes), then distribute the returned segment embeddings
    to the matching FiftyOne clips.

    Args:
        clips_view: The ``ClipsView`` produced by :func:`segment_into_clips`.
        dataset:    The parent FiftyOne dataset (used to look up metadata).
        api_key:    Twelve Labs key; falls back to ``TWELVELABS_API_KEY`` env var.
    """
    resolved_key = api_key or os.environ.get("TWELVELABS_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Twelve Labs API key required. "
            "Pass api_key= or set TWELVELABS_API_KEY."
        )

    client = TwelveLabs(api_key=resolved_key)

    # Process one source video at a time.
    for filepath in clips_view.distinct("filepath"):
        video_name = Path(filepath).name
        print(f"[embed] Submitting {video_name} to {EMBED_MODEL} …")

        with open(filepath, "rb") as vf:
            task = client.embed.tasks.create(
                model_name=EMBED_MODEL,
                video_file=vf,
                video_embedding_scope=["clip"],
            )

        # Poll until the task is ready.
        while True:
            task_result = client.embed.tasks.retrieve(
                task.id,
                embedding_option=["visual", "audio", "transcription"],
            )
            print(f"[embed] {video_name}: status={task_result.status}")
            if task_result.status == "ready":
                break
            if task_result.status == "failed":
                print(f"[embed] WARNING: {video_name} failed — skipping.")
                break
            time.sleep(5)

        if task_result.status != "ready":
            continue

        api_segments = _parse_segment_embeddings(task_result)
        print(f"[embed] {video_name}: received {len(api_segments)} segment(s)")

        # Look up FPS from the parent sample's metadata.
        parent = dataset.match(
            fo.ViewField("filepath") == filepath
        ).first()
        fps: float = parent.metadata.frame_rate

        # Distribute embeddings to matching clips.
        video_clips = clips_view.match(
            fo.ViewField("filepath") == filepath
        )
        for clip in video_clips.iter_samples(autosave=True):
            support = clip.support  # [first_frame, last_frame] (1-indexed)
            clip_start = (support[0] - 1) / fps
            clip_end = support[1] / fps

            seg = _best_match(api_segments, clip_start, clip_end)
            if seg is None:
                continue

            if seg.visual is not None:
                clip["visual_embedding"] = seg.visual.tolist()
            if seg.audio is not None:
                clip["audio_embedding"] = seg.audio.tolist()
            if seg.description is not None:
                clip["segment_description"] = seg.description

        print(f"[embed] {video_name}: clips populated.")

    print("[embed] All videos embedded.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Download sample videos.
    video_paths = download_epic_kitchens_samples()

    # 2. Load into FiftyOne.
    dataset = load_into_fiftyone(video_paths)

    # 3. Segment into 3-second overlapping clips.
    clips_view = segment_into_clips(dataset)

    # 4. Initialise schema fields.
    initialize_embedding_fields(clips_view)

    # 5. Embed clips via Twelve Labs.
    embed_clips_with_twelvelabs(clips_view, dataset)

    print("[done] Dataset ready. Launch the app with:")
    print("       fo.launch_app(dataset)")


if __name__ == "__main__":
    main()
