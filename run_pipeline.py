"""
run_pipeline.py

End-to-end pipeline: download EPIC-Kitchens samples, embed with Twelve Labs,
generate human-readable segment descriptions, and launch the FiftyOne app.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
import time

import fiftyone as fo
from dotenv import load_dotenv
from twelvelabs import TwelveLabs

from ingest_and_embed import (
    download_epic_kitchens_samples,
    load_into_fiftyone,
    segment_into_clips,
    initialize_embedding_fields,
)

load_dotenv()

MIN_ANALYZE_INTERVAL_SEC = 8.2
MAX_ANALYZE_RETRIES = 5
_RETRY_AFTER_REGEX = re.compile(r"'retry-after':\s*'(?P<seconds>\d+)'")


def _clip_label_text(raw_description: str | None) -> str:
    """Return a guaranteed human-readable clip description."""
    if isinstance(raw_description, str) and raw_description.strip():
        return raw_description.strip()
    return "General kitchen activity in this segment"


def _extract_clip_file(
    source_path: str,
    window_start_sec: float,
    window_end_sec: float,
) -> str:
    """Extract a short clip window to a temporary MP4 file."""
    duration = max(0.1, window_end_sec - window_start_sec)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{window_start_sec:.3f}",
            "-i",
            source_path,
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            "scale=640:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _analysis_window(
    segment_start_sec: float,
    segment_end_sec: float,
    video_duration_sec: float,
    min_duration_sec: float = 4.1,
) -> tuple[float, float]:
    """Expand a segment to satisfy Analyze minimum duration while staying in bounds."""
    segment_duration = max(0.0, segment_end_sec - segment_start_sec)
    if video_duration_sec <= min_duration_sec:
        return 0.0, video_duration_sec

    if segment_duration >= min_duration_sec:
        return segment_start_sec, segment_end_sec

    pad = (min_duration_sec - segment_duration) / 2.0
    window_start = max(0.0, segment_start_sec - pad)
    window_end = min(video_duration_sec, segment_end_sec + pad)

    # If clamped by either edge, shift to preserve required duration.
    current_duration = window_end - window_start
    if current_duration < min_duration_sec:
        if window_start <= 0.0:
            window_end = min(video_duration_sec, min_duration_sec)
        elif window_end >= video_duration_sec:
            window_start = max(0.0, video_duration_sec - min_duration_sec)

    return window_start, window_end


def _describe_clip_with_api(
    client: TwelveLabs,
    clip_path: str,
    last_request_at: float,
) -> tuple[str, float]:
    """Generate a short natural-language description for one clip via TwelveLabs."""
    with open(clip_path, "rb") as f:
        encoded_video = base64.b64encode(f.read()).decode("ascii")

    attempt = 0
    while attempt < MAX_ANALYZE_RETRIES:
        attempt += 1

        elapsed = time.time() - last_request_at
        if elapsed < MIN_ANALYZE_INTERVAL_SEC:
            time.sleep(MIN_ANALYZE_INTERVAL_SEC - elapsed)

        try:
            response = client.analyze(
                prompt=(
                    "Describe what is happening in this short video clip in one concise sentence. "
                    "Focus on visible actions and objects."
                ),
                video={"type": "base64_string", "base64_string": encoded_video},
                max_tokens=80,
            )
            request_time = time.time()
            text = response.data if response is not None else None
            if isinstance(text, str) and text.strip():
                return text.strip(), request_time
            return "Kitchen activity is visible in this segment", request_time
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ != "TooManyRequestsError":
                raise

            message = str(exc)
            match = _RETRY_AFTER_REGEX.search(message)
            retry_after_sec = int(match.group("seconds")) + 1 if match else 30
            print(
                f"[describe] Rate limit hit (attempt {attempt}/{MAX_ANALYZE_RETRIES}). "
                f"Sleeping {retry_after_sec}s before retry..."
            )
            time.sleep(float(retry_after_sec))
            last_request_at = time.time()

    raise RuntimeError(
        "Exceeded maximum retries while requesting clip description from TwelveLabs"
    )


def _describe_clips_with_twelvelabs(
    clips: fo.DatasetView,
    dataset: fo.Dataset,
) -> None:
    """Populate clip descriptions with per-segment TwelveLabs Analyze outputs."""
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise ValueError("TWELVELABS_API_KEY is required for clip description generation")

    client = TwelveLabs(api_key=api_key)

    fps_by_sample: dict[str, float] = {}
    filepath_by_sample: dict[str, str] = {}
    duration_by_sample: dict[str, float] = {}
    last_request_at = 0.0
    for sample in dataset:
        fps_by_sample[sample.id] = float(sample.metadata.frame_rate)
        filepath_by_sample[sample.id] = str(sample.filepath)
        duration_by_sample[sample.id] = (
            float(sample.metadata.total_frame_count) / float(sample.metadata.frame_rate)
        )

    for clip in clips.iter_samples(autosave=True):
        fps = fps_by_sample[clip.sample_id]
        source_path = filepath_by_sample[clip.sample_id]
        video_duration = duration_by_sample[clip.sample_id]
        start_sec = (int(clip.support[0]) - 1) / fps
        end_sec = int(clip.support[1]) / fps
        window_start, window_end = _analysis_window(start_sec, end_sec, video_duration)

        if window_end - window_start < 4.0:
            description = (
                f"Activity in segment from {start_sec:.1f}s to {end_sec:.1f}s"
            )
            clip["segment_description"] = description
            clip["segment_event"] = fo.Classification(label=description)
            clip["contextual_label"] = description
            continue

        tmp_clip_path = _extract_clip_file(source_path, window_start, window_end)
        try:
            description, last_request_at = _describe_clip_with_api(
                client, tmp_clip_path, last_request_at,
            )
        finally:
            try:
                os.remove(tmp_clip_path)
            except OSError:
                pass

        if not description.strip():
            description = (
                f"Activity in segment from {start_sec:.1f}s to {end_sec:.1f}s"
            )

        clip["segment_description"] = description
        clip["segment_event"] = fo.Classification(label=description)
        clip["contextual_label"] = description


def _build_descriptive_segments_view(
    dataset: fo.Dataset,
    clips: fo.DatasetView,
) -> fo.DatasetView:
    """Create temporal labels from clip descriptions and return a clips view."""
    detections_by_sample: dict[str, list[fo.TemporalDetection]] = {}

    for clip in clips:
        label = _clip_label_text(clip["segment_description"])
        detection = fo.TemporalDetection(
            label=label,
            support=[int(clip.support[0]), int(clip.support[1])],
        )
        detections_by_sample.setdefault(clip.sample_id, []).append(detection)

    for sample in dataset.iter_samples(autosave=True):
        sample["descriptive_segments"] = fo.TemporalDetections(
            detections=detections_by_sample.get(sample.id, []),
        )

    return dataset.to_clips("descriptive_segments")


def main() -> None:
    # ── 1. Ingest & embed ─────────────────────────────────────────────────
    video_paths = download_epic_kitchens_samples()
    dataset = load_into_fiftyone(video_paths)
    clips = segment_into_clips(dataset)
    initialize_embedding_fields(clips)
    _describe_clips_with_twelvelabs(clips, dataset)

    # ── 2. Normalize any empty labels (should be rare) ────────────────────
    for clip in clips.iter_samples(autosave=True):
        description = _clip_label_text(clip["segment_description"])
        clip["segment_description"] = description
        clip["segment_event"] = fo.Classification(label=description)
        clip["contextual_label"] = description

    # ── 3. Launch app with visible descriptive labels ─────────────────────
    print("\n[pipeline] Done. Launching FiftyOne app …")
    print("[pipeline] Launching clips with descriptive labels for each segment.\n")

    descriptive_clips = _build_descriptive_segments_view(dataset, clips)
    session = fo.launch_app(view=descriptive_clips)
    session.wait()


if __name__ == "__main__":
    main()
