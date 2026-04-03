"""
__init__.py

FiftyOne plugin operators for Cross-Modal Anomaly Explorer.
Wraps the run_pipeline logic into two operators:
  - compute_anomalies  : embed + score + write back to dataset
  - show_anomalies     : filter dataset to surface most anomalous segments
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone as fo

from .ingest_and_embed import (
    load_into_fiftyone,
    segment_into_clips,
    initialize_embedding_fields,
    embed_clips_with_twelvelabs,
)
from .dual_clustering import assign_sensory_clusters


def _score_crossmodal(visual_matrix, audio_matrix):
    v_norms = np.linalg.norm(visual_matrix, axis=1, keepdims=True)
    a_norms = np.linalg.norm(audio_matrix, axis=1, keepdims=True)
    v_norm = visual_matrix / np.where(v_norms == 0, 1.0, v_norms)
    a_norm = audio_matrix / np.where(a_norms == 0, 1.0, a_norms)
    cos_sim = np.sum(v_norm * a_norm, axis=1)
    return (1.0 - cos_sim).tolist()


# ---------------------------------------------------------------------------
# Operator 1: Compute Anomalies
# ---------------------------------------------------------------------------

class ComputeAnomalies(foo.Operator):

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_anomalies",
            label="Cross-Modal: Compute Anomalies",
            description=(
                "Embeds video segments with Twelve Labs Marengo, clusters visual "
                "and audio embeddings independently with HDBSCAN, and scores "
                "cross-modal divergence per segment."
            ),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.float(
            "clip_duration",
            label="Clip duration (seconds)",
            description="Length of each segment sent to Twelve Labs.",
            default=3.0,
            required=True,
        )

        inputs.float(
            "clip_stride",
            label="Clip stride (seconds)",
            description="Step between clips. Less than duration = overlapping clips.",
            default=1.5,
            required=True,
        )

        inputs.float(
            "threshold_std_multiplier",
            label="Threshold multiplier",
            description=(
                "Anomaly threshold = mean + (multiplier × std). "
                "Higher = stricter, fewer anomalies flagged."
            ),
            default=1.5,
            required=True,
        )

        n = len(ctx.view)
        inputs.view(
            "info",
            types.Notice(
                label=f"Will process {n} video(s). Each will be uploaded to Twelve Labs."
            ),
        )

        return types.Property(inputs, view=types.View(label="Compute Cross-Modal Anomalies"))

    def execute(self, ctx):
        clip_duration = ctx.params.get("clip_duration", 3.0)
        clip_stride = ctx.params.get("clip_stride", 1.5)
        std_mult = ctx.params.get("threshold_std_multiplier", 1.5)

        api_key = (
            ctx.secrets.get("TWELVELABS_API_KEY")
            or os.environ.get("TWELVELABS_API_KEY")
        )

        dataset = ctx.dataset
        stats = {"processed": 0, "skipped": 0, "anomalies_found": 0}

        for sample in ctx.view.iter_samples(progress=True):
            try:
                video_path = Path(sample.filepath)

                # Create a temp single-video dataset for embedding
                tmp_dataset = fo.Dataset(
                    name=f"_tmp_crossmodal_{sample.id}",
                    overwrite=True,
                )
                tmp_dataset.media_type = "video"
                tmp_dataset.add_sample(fo.Sample(filepath=str(video_path)))
                tmp_dataset.compute_metadata()

                clips = segment_into_clips(
                    tmp_dataset,
                    clip_duration=clip_duration,
                    stride=clip_stride,
                )
                initialize_embedding_fields(clips)
                embed_clips_with_twelvelabs(clips, tmp_dataset, api_key=api_key)

                # Collect embeddings
                embedded_ids, visual_list, audio_list = [], [], []
                for clip in clips:
                    vis = clip["visual_embedding"]
                    aud = clip["audio_embedding"]
                    if (
                        isinstance(vis, list) and any(v != 0.0 for v in vis)
                        and isinstance(aud, list) and any(v != 0.0 for v in aud)
                    ):
                        visual_list.append(vis)
                        audio_list.append(aud)
                        embedded_ids.append(clip.id)

                if not embedded_ids:
                    stats["skipped"] += 1
                    fo.delete_dataset(f"_tmp_crossmodal_{sample.id}")
                    continue

                visual_matrix = np.array(visual_list, dtype=np.float32)
                audio_matrix = np.array(audio_list, dtype=np.float32)

                # Cluster
                visual_labels, audio_labels = assign_sensory_clusters(
                    visual_matrix, audio_matrix,
                )

                # Score
                scores = _score_crossmodal(visual_matrix, audio_matrix)
                scores_array = np.array(scores)
                threshold = float(scores_array.mean() + std_mult * scores_array.std())

                # Write back to the original sample
                label_map = {
                    cid: (v, a, s)
                    for cid, v, a, s in zip(embedded_ids, visual_labels, audio_labels, scores)
                }

                clip_iter = iter(clips)
                orig_sample = dataset[sample.id]
                # Copy segment detections from tmp to original sample
                tmp_sample = tmp_dataset.first()
                detections = tmp_sample["segments"].detections

                for det in detections:
                    clip = next(clip_iter)
                    if clip.id in label_map:
                        v_lbl, a_lbl, score = label_map[clip.id]
                        det.label = "ANOMALY" if score >= threshold else "normal"
                        det["visual_cluster"] = v_lbl
                        det["audio_cluster"] = a_lbl
                        det["crossmodal_divergence"] = round(float(score), 4)

                orig_sample["segments"] = fo.TemporalDetections(detections=detections)
                orig_sample["max_crossmodal_divergence"] = round(float(scores_array.max()), 4)
                orig_sample["mean_crossmodal_divergence"] = round(float(scores_array.mean()), 4)
                orig_sample["anomaly_threshold"] = round(threshold, 4)
                orig_sample["anomaly_segment_count"] = sum(
                    1 for s in scores if s >= threshold
                )
                orig_sample.save()

                stats["processed"] += 1
                stats["anomalies_found"] += orig_sample["anomaly_segment_count"]

                fo.delete_dataset(f"_tmp_crossmodal_{sample.id}")

            except Exception as e:
                print(f"Error processing {sample.filepath}: {e}")
                stats["skipped"] += 1

        ctx.ops.reload_dataset()
        return stats

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("processed", label="Videos processed")
        outputs.int("skipped", label="Videos skipped")
        outputs.int("anomalies_found", label="Anomalous segments found")
        return types.Property(outputs)


# ---------------------------------------------------------------------------
# Operator 2: Show Anomalies
# ---------------------------------------------------------------------------

class ShowAnomalies(foo.Operator):

    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_anomalies",
            label="Cross-Modal: Show Anomalies",
            description="Sorts dataset by max cross-modal divergence and surfaces the top results.",
            dynamic=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.int(
            "top_n",
            label="Show top N most anomalous videos",
            default=10,
            required=True,
        )
        return types.Property(inputs, view=types.View(label="Show Anomalies"))

    def execute(self, ctx):
        top_n = ctx.params.get("top_n", 10)
        view = (
            ctx.dataset
            .exists("max_crossmodal_divergence")
            .sort_by("max_crossmodal_divergence", reverse=True)
            .limit(top_n)
        )
        ctx.ops.set_view(view)
        return {"shown": len(view)}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("shown", label="Videos shown")
        return types.Property(outputs)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def register(p):
    p.register(ComputeAnomalies)
    p.register(ShowAnomalies)
