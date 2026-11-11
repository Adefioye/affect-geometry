"""Portable, checksummed sample-level raw and neutral-projected EXP3 activations."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.analysis.cohorts import Cohorts
from affect_geometry.analysis.geometry_preprocessing import preprocess
from affect_geometry.analysis.representations import compute_emotion_centroids
from affect_geometry.embeddings.geometry import UnsupportedGeometry
from affect_geometry.logging import progress
from affect_geometry.persistence.correctness_geometry import GeometryStore, checksum, fingerprint

COHORT_NAMES = ("correct", "incorrect", "all")
BUNDLE_FORMAT = "rq3_cohort_activations_v1"


def _write_matrix(path, shape, dtype, batches):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    for start, values in batches:
        array[start : start + len(values)] = values
    array.flush()
    del array
    os.replace(temporary, path)
    return path


def export_cohort_bundle(
    root,
    *,
    model,
    run_id,
    cohorts,
    source_manifest,
    source_audit,
    prompt_samples,
    layer_paths,
    prompt_metadata,
    prompt_checksums,
    seed,
    projected_dtype="float32",
    batch_size=2048,
    tolerance=1e-10,
    budget=None,
):
    """Export raw cohorts and independently fitted cohort/block neutral projections.

    All shares raw rows with both subsets but has its own neutral transform.
    Unsupported projections retain the raw matrix and an explicit failure record.
    """
    if projected_dtype not in ("float32", "float64") or batch_size < 1:
        raise ValueError("Projection storage must be float32/float64 with positive batch size")
    if prompt_samples.sample_id.duplicated().any():
        raise ValueError("Duplicate prompt activation row IDs")
    manifest = dict(
        format=BUNDLE_FORMAT,
        model=model,
        run_id=run_id,
        model_id=source_manifest["model_id"],
        source=source_manifest,
        source_audit=source_audit,
        seed=seed,
        emotion_order=list(cohorts.labels),
        n_layers=len(layer_paths),
        hidden_size=prompt_metadata["hidden_size"],
        prompt_metadata=prompt_metadata,
        prompt_checksums=prompt_checksums,
        projected_dtype=projected_dtype,
        neutral_variance_threshold=0.5,
        tolerance=tolerance,
        projection="(x - equal_weight_non_neutral_centroid_mean) @ (I - components.T @ components)",
        projected_states_are_unit_normalized=False,
        input_condition="classification_prompt_before_generation",
        code_hash=fingerprint(
            {
                str(p.name): checksum(p)
                for p in [
                    Path(__file__),
                    Path(__file__).parents[1] / "analysis/representations.py",
                    Path(__file__).parents[1] / "analysis/geometry_preprocessing.py",
                ]
            }
        ),
    )
    store = GeometryStore(root, manifest)
    if store.valid("run", store.manifest):
        return PreparedActivations.open(root)
    store.csv("cohorts/support.csv", cohorts.support)
    store.csv("cohorts/counts.csv", cohorts.counts)
    store.csv("cohorts/unresolved.csv", cohorts.unresolved)
    positions = {s: i for i, s in enumerate(prompt_samples.sample_id)}
    for cohort in progress(
        COHORT_NAMES, description=f"{model}: export cohort activations", unit="cohort"
    ):
        samples = (
            cohorts.samples.loc[cohorts.samples.cohort.eq(cohort)]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        if not set(samples.sample_id) <= set(positions):
            raise ValueError("Selected cohort IDs missing from prompt activations")
        rows = np.array([positions[s] for s in samples.sample_id], dtype=int)
        if "input_token_count" in samples:
            samples["activation_token_position"] = samples.input_token_count.astype(int) - 1
        store.csv(f"cohorts/{cohort}/samples.csv", samples)
        for layer, source in enumerate(
            progress(layer_paths, description=f"{cohort}: neutral projection", unit="layer")
        ):
            if budget is not None:
                budget.check()
            stage = f"cohorts/{cohort}/layer_{layer:03d}"
            provenance = dict(
                manifest=fingerprint(store.manifest),
                sample_ids=samples.sample_id.tolist(),
                layer=layer,
                cohort=cohort,
            )
            if store.valid(stage, provenance):
                continue
            prefix = f"cohorts/{cohort}"
            raw_path = store.root / f"{prefix}/last_token/layer_{layer:03d}.npy"
            projected_path = store.root / f"{prefix}/neutral_projected/layer_{layer:03d}.npy"
            source_array = np.load(source, mmap_mode="r")
            if source_array.shape != (len(prompt_samples), manifest["hidden_size"]):
                raise ValueError("Prompt activation shape mismatch")
            raw = _write_matrix(
                raw_path,
                (len(rows), manifest["hidden_size"]),
                source_array.dtype,
                (
                    (i, source_array[rows[i : i + batch_size]])
                    for i in range(0, len(rows), batch_size)
                ),
            )
            paths = [raw]
            status = dict(
                cohort=cohort,
                layer=layer,
                sample_count=len(rows),
                sample_ids_sha256=fingerprint(samples.sample_id.tolist()),
                emotion_order=list(cohorts.labels),
            )
            try:
                values = np.load(raw_path, mmap_mode="r")
                prep = preprocess(values, samples, cohorts.labels, tolerance)

                def projected_batches():
                    for start in range(0, len(values), batch_size):
                        centered = (
                            np.asarray(values[start : start + batch_size], float)
                            - prep["grand_mean"]
                        )
                        yield (
                            start,
                            centered - (centered @ prep["components"].T) @ prep["components"],
                        )

                paths.append(
                    _write_matrix(
                        projected_path, values.shape, projected_dtype, projected_batches()
                    )
                )
                projected = np.load(projected_path, mmap_mode="r")
                # Downstream centroids use the saved states, including storage rounding.
                prep["R"] = compute_emotion_centroids(
                    projected, samples.emotion, labels=cohorts.labels
                ).values
                norms = np.linalg.norm(prep["R"], axis=1)
                if np.any(norms <= tolerance):
                    raise UnsupportedGeometry("near_zero_saved_projected_centroid")
                prep["U"] = prep["R"] / norms[:, None]
                prep["S"] = np.clip(prep["U"] @ prep["U"].T, -1, 1)
                prep["D"] = np.clip(1 - prep["S"], 0, 2)
                np.fill_diagonal(prep["D"], 0)
                paths.append(store.arrays(f"{prefix}/preprocessing/layer_{layer:03d}.npz", **prep))
                status.update(status="ok", reason="")
            except UnsupportedGeometry as exc:
                status.update(status="unsupported", reason=str(exc))
                projected_path.unlink(missing_ok=True)
                paths = [raw]
            paths.append(store.json(f"{prefix}/status/layer_{layer:03d}.json", status))
            store.complete(stage, provenance, paths)
    store.finish()
    return PreparedActivations.open(root)


class PreparedActivations:
    """Validated frozen cohort rows and precomputed sample-level transformations."""

    def __init__(self, root, manifest, cohorts, artifacts):
        self.root, self.manifest, self.cohorts, self.artifacts = root, manifest, cohorts, artifacts

    def __len__(self):
        return self.manifest["n_layers"]

    @classmethod
    def open(cls, root, expected_model=None):
        root = Path(root).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError("Expected an EXP3 raw/projected cohort activation bundle")
        if expected_model is not None and manifest["model"] != expected_model:
            raise ValueError("Activation bundle belongs to a different model")
        if manifest["input_condition"] != "classification_prompt_before_generation":
            raise ValueError("Wrong activation input condition")
        record = json.loads((root / "checkpoints/run.json").read_text())
        if record["provenance"] != fingerprint(manifest):
            raise ValueError("Bundle manifest differs from completion checkpoint")
        artifacts = record["artifacts"]
        required_metadata = {
            "manifest.json",
            "cohorts/support.csv",
            "cohorts/counts.csv",
            "cohorts/unresolved.csv",
        }
        if not required_metadata <= set(artifacts):
            raise ValueError("Completion record must cover all bundle metadata")
        for name, sha in artifacts.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or checksum(path) != sha:
                raise ValueError(f"Bundle artifact missing or checksum mismatch: {name}")
        frames = []
        for cohort in COHORT_NAMES:
            path = f"cohorts/{cohort}/samples.csv"
            if path not in artifacts:
                raise ValueError(f"Missing cohort sample manifest: {cohort}")
            frame = pd.read_csv(root / path, dtype={"sample_id": str})
            if frame.sample_id.duplicated().any() or frame.sample_id.isna().any():
                raise ValueError("Duplicate/null cohort sample IDs")
            if frame.sample_id.tolist() != sorted(frame.sample_id):
                raise ValueError("Cohort rows must use canonical sample-ID order")
            if not frame.cohort.eq(cohort).all():
                raise ValueError("Cohort row membership mismatch")
            frames.append(frame)
            for layer in range(manifest["n_layers"]):
                status_path = f"cohorts/{cohort}/status/layer_{layer:03d}.json"
                raw_path = f"cohorts/{cohort}/last_token/layer_{layer:03d}.npy"
                required = [status_path, raw_path]
                if status_path not in artifacts:
                    raise ValueError("Missing cohort/layer status record")
                status = json.loads((root / status_path).read_text())
                if (
                    status["sample_ids_sha256"] != fingerprint(frame.sample_id.tolist())
                    or status["emotion_order"] != manifest["emotion_order"]
                ):
                    raise ValueError("Cohort/layer row identity mismatch")
                if status["status"] == "ok":
                    required += [
                        f"cohorts/{cohort}/neutral_projected/layer_{layer:03d}.npy",
                        f"cohorts/{cohort}/preprocessing/layer_{layer:03d}.npz",
                    ]
                elif status["status"] != "unsupported" or not status.get("reason"):
                    raise ValueError("Invalid projection status")
                for name in required:
                    if name not in artifacts:
                        raise ValueError(f"Unchecked required bundle artifact: {name}")
                    if name.endswith(".npy"):
                        array = np.load(root / name, mmap_mode="r")
                        if array.shape != (len(frame), manifest["hidden_size"]):
                            raise ValueError("Cohort matrix/sample shape mismatch")
        correct, incorrect, all_rows = frames
        if set(correct.sample_id) & set(incorrect.sample_id) or set(correct.sample_id) | set(
            incorrect.sample_id
        ) != set(all_rows.sample_id):
            raise ValueError("Frozen cohort IDs violate the exact all-union identity")
        if (
            not correct.compat_correct.eq(True).all()
            or not incorrect.compat_correct.eq(False).all()
        ):
            raise ValueError("Frozen compatibility-correctness partition is invalid")
        union = pd.concat([correct, incorrect]).sort_values("sample_id")
        for column in ("emotion", "compat_correct", "prompt_hash", "source_split"):
            if union[column].tolist() != all_rows[column].tolist():
                raise ValueError(f"All cohort differs from its subsets: {column}")
        for frame in frames:
            if set(frame.emotion) != set(manifest["emotion_order"]):
                raise ValueError("Cohorts must have identical supported emotion sets")
        if not correct.groupby("emotion").size().equals(incorrect.groupby("emotion").size()):
            raise ValueError("Frozen per-emotion cohort balance is invalid")
        support = pd.read_csv(root / "cohorts/support.csv")
        for label in manifest["emotion_order"]:
            entry = support.loc[support.emotion.eq(label)].iloc[0]
            if entry.correct <= 100 or entry.sampled_per_subset != int(
                correct.emotion.eq(label).sum()
            ):
                raise ValueError("Frozen eligibility/support does not match sampled rows")
        cohorts = Cohorts(
            pd.concat(frames, ignore_index=True),
            support,
            pd.read_csv(root / "cohorts/unresolved.csv"),
            pd.read_csv(root / "cohorts/counts.csv"),
            tuple(manifest["emotion_order"]),
        )
        return cls(root, manifest, cohorts, artifacts)

    def preprocessing(self, cohort, layer):
        status = json.loads(
            (self.root / f"cohorts/{cohort}/status/layer_{layer:03d}.json").read_text()
        )
        if status["status"] != "ok":
            raise UnsupportedGeometry(status["reason"])
        with np.load(self.root / f"cohorts/{cohort}/preprocessing/layer_{layer:03d}.npz") as arrays:
            return dict(arrays)

    def projected(self, cohort, layer):
        self.preprocessing(cohort, layer)
        return np.load(
            self.root / f"cohorts/{cohort}/neutral_projected/layer_{layer:03d}.npy", mmap_mode="r"
        )

    def reference(self):
        return dict(
            root=str(self.root),
            manifest=self.manifest,
            manifest_sha256=checksum(self.root / "manifest.json"),
            checksums=self.artifacts,
            metadata=dict(n_layers=len(self)),
        )
