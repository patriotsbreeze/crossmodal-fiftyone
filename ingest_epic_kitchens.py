"""
ingest_epic_kitchens.py

Downloads a small sample of EPIC-Kitchens videos from Hugging Face, loads them
into a FiftyOne dataset, and indexes them with Twelve Labs (marengo2.6, visual+audio).
"""

from __future__ import annotations

import os
from pathlib import Path

import fiftyone as fo
from huggingface_hub import HfApi, hf_hub_download
from twelvelabs import TwelveLabs
from twelvelabs.models.index import IndexModel

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SAMPLE_CLIPS = 5  # Hard cap — do not remove
HF_REPO_ID = "all-of-us/epic-kitchens-100-samples"
HF_REPO_TYPE = "dataset"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── Download ──────────────────────────────────────────────────────────────────


def list_video_files_in_repo(repo_id: str, repo_type: str) -> list[str]:
    """Return repo-relative paths for all video files in a Hugging Face repo."""
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    return [
        f for f in all_files
        if Path(f).suffix.lower() in VIDEO_EXTENSIONS
    ]


def download_epic_kitchens_samples(
    output_dir: str | Path = "data/epic_kitchens",
    max_clips: int = MAX_SAMPLE_CLIPS,
) -> list[Path]:
    """
    Download up to *max_clips* short video samples from the EPIC-Kitchens
    Hugging Face mirror and return the local file paths.

    Args:
        output_dir: Directory where downloaded videos will be stored.
        max_clips:  Maximum number of clips to download (capped at MAX_SAMPLE_CLIPS).

    Returns:
        List of Path objects pointing to the downloaded video files.
    """
    max_clips = min(max_clips, MAX_SAMPLE_CLIPS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []

    # Collect already-downloaded files so re-runs are idempotent.
    existing = sorted(
        p for p in output_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(existing) >= max_clips:
        print(f"[download] Found {len(existing)} cached clips — skipping download.")
        return existing[:max_clips]

    print(f"[download] Listing videos in {HF_REPO_ID} …")
    remote_files = list_video_files_in_repo(HF_REPO_ID, HF_REPO_TYPE)

    if not remote_files:
        raise RuntimeError(
            f"No video files found in repo '{HF_REPO_ID}'. "
            "Check the repo_id or your Hugging Face credentials."
        )

    for remote_path in remote_files[:max_clips]:
        local_path = output_dir / Path(remote_path).name
        if local_path.exists():
            print(f"[download] Skipping (cached): {local_path.name}")
        else:
            print(f"[download] Downloading: {remote_path}")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=remote_path,
                local_dir=str(output_dir),
            )
            # hf_hub_download may place files in subdirectories; normalise.
            downloaded_path = Path(downloaded)
            if downloaded_path != local_path:
                downloaded_path.rename(local_path)
        video_paths.append(local_path)

    print(f"[download] Ready: {len(video_paths)} clip(s) in {output_dir}")
    return video_paths


# ── FiftyOne ──────────────────────────────────────────────────────────────────


def load_into_fiftyone(
    video_paths: list[Path],
    dataset_name: str = "epic_kitchens_samples",
) -> fo.Dataset:
    """
    Create (or overwrite) a FiftyOne video dataset from a list of local paths.

    Args:
        video_paths:  Local video file paths to ingest.
        dataset_name: Name for the FiftyOne dataset.

    Returns:
        The populated FiftyOne Dataset.
    """
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.media_type = "video"

    samples = [fo.Sample(filepath=str(p.resolve())) for p in video_paths]
    dataset.add_samples(samples)

    print(f"[fiftyone] Dataset '{dataset_name}' created with {len(dataset)} sample(s).")
    return dataset


# ── Twelve Labs ───────────────────────────────────────────────────────────────


def _get_or_create_index(client: TwelveLabs, index_name: str) -> str:
    """
    Return the id of an existing Twelve Labs index with *index_name*, or create
    a new one configured for marengo2.6 with visual and audio embeddings.

    Args:
        client:     Authenticated TwelveLabs client.
        index_name: Human-readable name for the index.

    Returns:
        The Twelve Labs index id (string).
    """
    # Check for an existing index with this name.
    for existing in client.indexes.list():
        if existing.name == index_name:
            print(f"[twelvelabs] Reusing existing index '{index_name}' (id={existing.id})")
            return existing.id

    print(f"[twelvelabs] Creating index '{index_name}' with marengo2.6 (visual + audio) …")
    index = client.indexes.create(
        name=index_name,
        models=[
            IndexModel(
                name="marengo2.6",
                options=["visual", "audio"],
            )
        ],
    )
    print(f"[twelvelabs] Index created: id={index.id}")
    return index.id


def index_multimodal_to_twelvelabs(
    dataset: fo.Dataset,
    *,
    api_key: str | None = None,
    index_name: str = "epic_kitchens_multimodal",
) -> None:
    """
    Index every video sample in *dataset* with Twelve Labs (marengo2.6,
    visual + audio).  The Twelve Labs ``video_id`` is stored as a field on
    each FiftyOne sample.

    Args:
        dataset:    A FiftyOne video dataset whose samples have valid filepaths.
        api_key:    Twelve Labs API key.  Falls back to the ``TWELVELABS_API_KEY``
                    environment variable when omitted.
        index_name: Name of the Twelve Labs index to use (created if absent).
    """
    resolved_key = api_key or os.environ.get("TWELVELABS_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Twelve Labs API key is required. "
            "Pass api_key= or set the TWELVELABS_API_KEY environment variable."
        )

    client = TwelveLabs(api_key=resolved_key)
    index_id = _get_or_create_index(client, index_name)

    # Ensure the custom field exists on the dataset schema.
    if "twelvelabs_video_id" not in dataset.get_field_schema():
        dataset.add_sample_field("twelvelabs_video_id", fo.StringField)

    for sample in dataset.iter_samples(progress=True, autosave=True):
        filepath = sample.filepath
        print(f"[twelvelabs] Submitting task for: {Path(filepath).name}")

        task = client.tasks.create(
            index_id=index_id,
            file=filepath,
        )

        print(f"[twelvelabs] Task {task.id} submitted — waiting for completion …")
        task.wait_for_done(sleep_interval=5)

        if task.status != "ready":
            print(
                f"[twelvelabs] WARNING: task {task.id} finished with status "
                f"'{task.status}' — skipping video_id assignment."
            )
            continue

        sample["twelvelabs_video_id"] = task.video_id
        print(
            f"[twelvelabs] Indexed '{Path(filepath).name}' → video_id={task.video_id}"
        )

    print("[twelvelabs] All samples indexed.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    video_paths = download_epic_kitchens_samples()
    dataset = load_into_fiftyone(video_paths)
    index_multimodal_to_twelvelabs(dataset)


if __name__ == "__main__":
    main()
