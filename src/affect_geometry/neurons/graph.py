"""Sparse census using Goodfire's pinned Curveball null and exact stabilized score.

Curveball adaptation: Copyright (c) 2026 Goodfire, MIT license.
See third_party/goodfire-neuron-communities/LICENSE and census.py for the source/notice.
Unlike upstream, only observed candidate pairs are materialized, never a full width-squared tile.
"""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy import sparse

from affect_geometry.data.neuron_transfer import child_seed
from affect_geometry.logging import progress
from affect_geometry.neurons import UPSTREAM_CENSUS_SHA, UPSTREAM_COMMIT

EDGE_COLUMNS = [
    "layer_i",
    "channel_i",
    "layer_j",
    "channel_j",
    "count",
    "null_mean",
    "null_std",
    "null_scale",
    "excess",
    "frequency_i",
    "frequency_j",
]


def _within(counts):
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return np.arange(ptr[-1]) - np.repeat(ptr[:-1], counts), ptr


def curveball(matrix, rng, sweeps=20):
    """Exact pinned shuffle, with original global state made explicit and widths generalized."""
    matrix = matrix.tocsr()
    n, d = matrix.shape
    indptr = matrix.indptr.astype(np.int64)
    lens = np.diff(indptr)
    ind = matrix.indices.astype(np.int32).copy()
    for _ in range(sweeps):
        perm = rng.permutation(n)
        half = n // 2
        a, b = perm[:half], perm[half : 2 * half]
        la, lb = lens[a], lens[b]
        total = la + lb
        wa, _ = _within(la)
        wb, _ = _within(lb)
        _, seg_ptr = _within(total)
        m = seg_ptr[-1]
        if not m:
            continue
        item = np.empty(m, dtype=np.int32)
        item[np.repeat(seg_ptr[:-1], la) + wa] = ind[np.repeat(indptr[a], la) + wa]
        item[np.repeat(seg_ptr[:-1] + la, lb) + wb] = ind[np.repeat(indptr[b], lb) + wb]
        seg = np.repeat(np.arange(half, dtype=np.int64), total)
        order = np.argsort(seg * (d + 1) + item, kind="stable")
        sv, ss = item[order], seg[order]
        dup = np.zeros(m, dtype=bool)
        dup[:-1] = (sv[:-1] == sv[1:]) & (ss[:-1] == ss[1:])
        common = dup.copy()
        common[1:] |= dup[:-1]
        common_count = np.bincount(ss[dup], minlength=half)
        keep_a = la - common_count
        pool_seg, pool_val = ss[~common], sv[~common]
        po = np.lexsort((rng.random(len(pool_seg)), pool_seg))
        ps, pv = pool_seg[po], pool_val[po]
        rank, _ = _within(np.bincount(ps, minlength=half))
        to_a = rank < keep_a[ps]
        cw, _ = _within(common_count)
        ind[np.repeat(indptr[a], common_count) + cw] = sv[dup]
        ind[np.repeat(indptr[b], common_count) + cw] = sv[dup]
        aw, _ = _within(keep_a)
        ind[np.repeat(indptr[a] + common_count, keep_a) + aw] = pv[to_a]
        nb = lb - common_count
        bw, _ = _within(nb)
        ind[np.repeat(indptr[b] + common_count, nb) + bw] = pv[~to_a]
    return sparse.csr_matrix((np.ones(len(ind), np.int64), ind, indptr), shape=(n, d))


def incidence(attributions, sample_ids, widths):
    index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    result = {}
    for layer, width in sorted(widths.items()):
        part = attributions[attributions.layer.eq(layer)] if len(attributions) else attributions
        if len(part) and part.duplicated(["sample_id", "layer", "channel"]).any():
            raise ValueError("Duplicate neuron selection within a prompt")
        rows = [index[v] for v in part.get("sample_id", [])]
        cols = list(part.get("channel", []))
        result[layer] = sparse.csr_matrix(
            (np.ones(len(rows), np.int64), (rows, cols)), shape=(len(sample_ids), width)
        )
    return result


def census(matrices, config, on_tile=None, cached_tile=None):
    frames, tile_metadata = [], []
    tiles = list(combinations_with_replacement(sorted(matrices), 2))
    for a, b in progress(tiles, description="Counting neuron co-attribution tiles", unit="tile"):
        sa, sb = matrices[a], matrices[b]
        observed = (sa.T @ sb).tocoo()
        keep = observed.data >= config.minimum_count
        if a == b:
            keep &= observed.row < observed.col
        i, j, counts = observed.row[keep], observed.col[keep], observed.data[keep]
        if not len(i):
            continue
        cached = cached_tile(a, b) if cached_tile else None
        combined = sa if a == b else sparse.hstack((sa, sb), format="csr")
        total = np.zeros(len(i), np.float64)
        square = np.zeros(len(i), np.float64)
        seeds = [child_seed(config.seed, "null", a, b, r) for r in range(config.null_replicates)]
        for seed in seeds if cached is None else []:
            shuffled = curveball(combined, np.random.default_rng(seed), config.null_sweeps)
            left = shuffled if a == b else shuffled[:, : sa.shape[1]]
            right = shuffled if a == b else shuffled[:, sa.shape[1] :]
            values = np.asarray((left.T @ right).tocsr()[i, j]).ravel().astype(np.float64)
            total += values
            square += values**2
        mean = total / len(seeds)
        std = np.sqrt(np.maximum(square / len(seeds) - mean**2, 0))
        scale = np.maximum(std, np.sqrt(np.maximum(mean, 0.25)))
        z = (counts - mean) / scale
        freq_a = np.asarray(sa.sum(0)).ravel()
        freq_b = np.asarray(sb.sum(0)).ravel()
        frame = pd.DataFrame(
            dict(
                layer_i=a,
                channel_i=i,
                layer_j=b,
                channel_j=j,
                count=counts,
                null_mean=mean,
                null_std=std,
                null_scale=scale,
                excess=z,
                frequency_i=freq_a[i],
                frequency_j=freq_b[j],
            )
        )
        frame = frame[frame.excess.ge(config.edge_threshold)].reset_index(drop=True)
        if cached is not None:
            frame = cached
        elif on_tile is not None:
            on_tile(a, b, frame)
        frames.append(frame)
        tile_metadata.append(dict(layers=[a, b], seeds=seeds, pairs_examined=len(i)))
    metadata = dict(
        upstream_commit=UPSTREAM_COMMIT,
        upstream_blob_sha=UPSTREAM_CENSUS_SHA,
        algorithm="curveball_combined_layer_rows",
        sweeps=config.null_sweeps,
        replicates=config.null_replicates,
        scale="max(population_std, sqrt(max(null_mean, 0.25)))",
        sparse_observed_pairs_only=True,
        tiles=tile_metadata,
    )
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EDGE_COLUMNS),
        metadata,
    )
