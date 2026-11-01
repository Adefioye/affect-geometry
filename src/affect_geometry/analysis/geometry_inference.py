"""Shared NRC-row schedules and separate aggregate/layerwise max-stat families."""

import json

import numpy as np

from affect_geometry.analysis.geometry_validation import evaluate
from affect_geometry.embeddings.geometry import UnsupportedGeometry
from affect_geometry.logging import progress
from affect_geometry.persistence.correctness_geometry import fingerprint


def permutation_schedule(n, repetitions=2000, seed=17, smoke=False):
    if repetitions != 2000 and not smoke:
        raise ValueError("Inferential runs require exactly 2,000 permutations")
    if repetitions < 1:
        raise ValueError("At least one permutation is required")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(repetitions)])


def infer(
    cells,
    targets,
    layers,
    schedule,
    selection="nested",
    store=None,
    prefix="inference",
    budget=None,
):
    """cells[(cohort, method, layer)] contains all permitted setting coordinates.

    Every family member is retained; a failed member invalidates family correction.
    """
    keys = sorted(cells)
    null = np.full((len(schedule), len(keys)), np.nan)
    choices = []
    failures = []
    invariant = fingerprint(
        dict(
            targets=targets,
            selection=selection,
            cells={str(k): v for k, v in cells.items()},
            run=None if store is None else store.manifest,
        )
    )
    for b, perm in enumerate(
        progress(schedule, description="Permuting NRC rows", unit="permutation")
    ):
        if budget is not None:
            budget.check()
        selected, transformations, permutation_failures = [], [], []
        provenance = dict(
            permutation=perm.tolist(),
            invariant=invariant,
        )
        stage = f"{prefix}/permutation-{b:04d}"
        artifact = f"{prefix}/permutations/{b:04d}.json"
        if store is not None and store.valid(stage, provenance):
            record = json.loads((store.root / artifact).read_text())
            null[b] = [np.nan if x is None else x for x in record["null"]]
            choices.append(record["selections"])
            failures.extend(record["failures"])
            continue
        for j, key in enumerate(keys):
            try:
                result = evaluate(cells[key], targets[perm], selection=selection)
                null[b, j] = result["cv_r2"]
                selected.append([f["setting"] for f in result["folds"]])
                transformations.append(
                    dict(
                        cell=list(key),
                        folds=result["folds"],
                        full_transform=result["full_transform"],
                    )
                )
            except UnsupportedGeometry as exc:
                selected.append(None)
                permutation_failures.append(dict(permutation=b, cell=list(key), reason=str(exc)))
        failures.extend(permutation_failures)
        choices.append(selected)
        if store is not None:
            path = store.json(
                artifact,
                dict(
                    null=null[b],
                    selections=selected,
                    transformations=transformations,
                    failures=permutation_failures,
                ),
            )
            store.complete(stage, provenance, [path])
    groups = sorted({key[:2] for key in keys})
    aggregate = np.full((len(schedule), len(groups)), np.nan)
    for j, group in enumerate(groups):
        indices = [keys.index((*group, layer)) for layer in layers]
        aggregate[:, j] = null[:, indices].mean(axis=1)
    return dict(
        keys=keys,
        groups=groups,
        layerwise_null=null,
        aggregate_null=aggregate,
        selections=choices,
        failures=failures,
    )


def corrected_pvalues(observed, null):
    observed, null = np.asarray(observed), np.asarray(null)
    if null.ndim != 2 or null.shape[1] != len(observed):
        raise ValueError("Inference family shape mismatch")
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        return np.full(len(observed), np.nan), np.full(len(observed), np.nan)
    raw = (1 + (null >= observed).sum(axis=0)) / (len(null) + 1)
    corrected = (1 + (null.max(axis=1)[:, None] >= observed).sum(axis=0)) / (len(null) + 1)
    return raw, corrected
