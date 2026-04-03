# crossmodal-fiftyone

A multimodal anomaly detection pipeline that finds **audio-visual mismatches** in video — moments where what you see and what you hear don't belong together.

Built on [FiftyOne](https://voxel51.com/fiftyone/), [Twelve Labs Marengo](https://www.twelvelabs.io/), and scikit-learn HDBSCAN clustering.

---

## How it works

```
Video files
    └─► Segment into 3-second overlapping clips (50% overlap)
            └─► Twelve Labs Marengo 3.0 (visual + audio embeddings, 512-d each)
                    └─► HDBSCAN clustering (independent per modality, PCA-reduced)
                            └─► Cosine divergence scoring (visual vs audio embedding)
                                    └─► Anomaly labels written to FiftyOne dataset
                                            └─► FiftyOne App for interactive exploration
```

Each video clip receives two 512-dimensional embeddings from Marengo's shared latent space — one for the visual stream and one for the audio stream. The **cross-modal divergence score** is the cosine distance between the two vectors. Clips above a dynamic threshold (`mean + 1.5 × std`) are flagged as `ANOMALY`.

---

## Project structure

```
crossmodal-fiftyone/
├── run_pipeline.py        # End-to-end pipeline entry point
├── ingest_and_embed.py    # Download, FiftyOne ingest, clip segmentation, Twelve Labs embedding
├── dual_clustering.py     # Independent HDBSCAN clustering for visual and audio embeddings
├── co_occurrence_math.py  # Anomaly scoring: audio drift and co-occurrence P(A|V)
├── __init__.py            # FiftyOne plugin operators
├── fiftyone.yml           # Plugin manifest
├── requirements.txt
└── .env                   # TWELVELABS_API_KEY goes here (not committed)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`ffmpeg` must also be on your PATH (used to remux EPIC-Kitchens downloads).

### 2. Configure your API key

Create a `.env` file in the project root:

```
TWELVELABS_API_KEY=your_key_here
```

### 3. Run the pipeline

```bash
python run_pipeline.py
```

---

## Use this as a FiftyOne plugin

This repository also ships a FiftyOne plugin with two operators defined in `__init__.py` and `fiftyone.yml`:
- `Cross-Modal: Compute Anomalies`
- `Cross-Modal: Show Anomalies`

### 1. Install the plugin locally

From this repository root, run:

```bash
fiftyone plugins create .
```

Then verify it is installed:

```bash
fiftyone plugins list
```

### 2. Launch FiftyOne with a video dataset

You can either run the end-to-end script:

```bash
python run_pipeline.py
```

or launch FiftyOne yourself after loading a dataset in Python.

### 3. Run the operators in the App

In the FiftyOne App:
1. Open the **Operators** panel
2. Run **Cross-Modal: Compute Anomalies**
3. Set:
        - `clip_duration` (default: `3.0`)
        - `clip_stride` (default: `1.5`)
        - `threshold_std_multiplier` (default: `1.5`)
4. After processing, run **Cross-Modal: Show Anomalies** to sort and filter to the top anomalous videos

### 4. Configure your API key for plugin runs

The `compute_anomalies` operator expects `TWELVELABS_API_KEY` from FiftyOne secrets or the environment.

Set it in your shell before launching FiftyOne:

```bash
export TWELVELABS_API_KEY=your_key_here
```

On Windows PowerShell:

```powershell
$env:TWELVELABS_API_KEY="your_key_here"
```

This will:
1. Load video(s) from disk into a FiftyOne dataset
2. Segment each video into 3-second overlapping clips
3. Submit each video to the Twelve Labs Embed API (Marengo 3.0)
4. Cluster visual and audio embeddings independently with HDBSCAN
5. Score each clip by cosine divergence between its visual and audio embedding
6. Write `ANOMALY` / `normal` labels and cluster IDs onto the FiftyOne dataset
7. Print a per-clip summary table and launch the FiftyOne App

---

## Output

The pipeline prints a table like:

```
  Start      End   V_cluster   A_cluster   divergence  flag
======================================================================
    0.0s    3.0s         V_0         A_-1       0.1234
    1.5s    4.5s         V_0         A_-1       0.1401
    3.0s    6.0s         V_1         A_0        0.9821  <<< ANOMALY
```

Each clip in the FiftyOne dataset gets these fields:
- `visual_cluster` — HDBSCAN cluster label for the visual stream (`V_0`, `V_1`, …)
- `audio_cluster` — HDBSCAN cluster label for the audio stream (`A_0`, `A_-1`, …)
- `crossmodal_divergence` — cosine distance score in `[0, 2]`
- `label` — `"ANOMALY"` or `"normal"`

---

## Using EPIC-Kitchens data

To download sample clips from EPIC-Kitchens instead of using a local file, edit `run_pipeline.py` and swap:

```python
# Current: local file
local_test_video = Path("trimmed_AVA_data.mp4")
video_paths = [local_test_video.resolve()]

# Switch to: HuggingFace download (up to 4 clips)
video_paths = download_epic_kitchens_samples()
```

Downloads are cached in `data/epic_kitchens/` and remuxed automatically with `ffmpeg`.

---

## Anomaly scoring details

Two scoring strategies are available in `co_occurrence_math.py`:

| Strategy | Function | When to use |
|---|---|---|
| **Cross-modal divergence** (primary) | `score_crossmodal` in `run_pipeline.py` | Single video — cosine distance between visual and audio embeddings |
| **Audio drift** | `score_audio_drift` | Single video with a known-silent baseline at the start |
| **Co-occurrence probability** | `calculate_contextual_anomalies` | Multi-video corpus — uses `P(audio_cluster \| visual_cluster)` |

The pipeline uses cross-modal divergence by default with a dynamic threshold of `mean + 1.5 × std` across all clips in the video.

---

## Requirements

- Python 3.10+
- ffmpeg (on PATH)
- Twelve Labs API key ([twelvelabs.io](https://www.twelvelabs.io/))
