"""Pairwise cross-layer co-occurrence census: one (layer_a, layer_b) tile per invocation.

Statistics
  * The world holds one CSR matrix per layer (``merged/L{layer:02d}.npz``: prompts x neurons, a 1 where the neuron is in
    the prompt's top-k attribution set) and the deduplicated prompt rows (``dedup_kept.npz``). The tile stacks the
    kept rows of layer A (and layer B, column offset D_MLP) and computes exact co-occurrence counts C: the full
    d x d matrix for a diagonal tile (A == B), only the A x B cross block otherwise.
  * The null is popularity preserving: ``--replicates`` curveball shuffles of the binary matrix (every prompt keeps
    its number of active neurons, every neuron keeps its number of prompts), run in parallel worker processes, give
    the per-pair null mean and std; z = (C - mean) / max(std, sqrt(max(mean, 0.25))).
  * A pair is stored when C >= C_MIN and z >= Z_STORE. Diagonal tiles store the strict upper triangle only.

Output: ``<census-dir>/pairs_L{A:02d}_L{B:02d}.npz`` with
  i, j      tile-local neuron indices (int32); on a cross tile j carries the D_MLP offset
  z         z-score against the null (float32)
  c, mean   observed count and null mean of the pair
  freq_i, freq_j   both neurons' prompt counts (so a reader can weight edges by lift = C N / (n_i n_j))
  n_rows, layers, replicates

Tiles are scheduled as an array: ``--n-tiles`` prints the number of (a, b) pairs with a <= b over the configured layer
tile, ``--tile k`` selects the k-th; an existing tile file is skipped unless ``--force``.

  python -m neuron_communities census --world <world> --layer-a 19 --layer-b 21 --workers 10
  python -m neuron_communities census --world <world> --tile $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

from neuron_communities import config


# tile state shared with the null-replicate workers (set in run() before the Pool forks)
MD: Path
A: int
B: int
n: int
d: int
indptr: np.ndarray
indices: np.ndarray
lens: np.ndarray
kept: np.ndarray


def tiles(layers: tuple[int, int]) -> list[tuple[int, int]]:
    """All (a, b) layer pairs with lo <= a <= b <= hi: the tiles the discovery window needs."""
    lo, hi = layers
    return [(a, b) for a in range(lo, hi + 1) for b in range(a, hi + 1)]


def load_csr(L: int, col_offset: int):
    z = np.load(MD / "merged" / f"L{L:02d}.npz")
    indptr, idx = z["indptr"].astype(np.int64), z["indices"].astype(np.int32)
    starts, ends = indptr[kept], indptr[kept + 1]
    lens_ = ends - starts
    out_ptr = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum(lens_, out=out_ptr[1:])
    src = np.repeat(starts, lens_) + (np.arange(out_ptr[-1]) - np.repeat(out_ptr[:-1], lens_))
    return out_ptr, idx[src] + col_offset


def cooc_block(ind: np.ndarray) -> np.ndarray:
    """Dense co-occurrence counts: full d x d for a diagonal tile, the A x B cross block otherwise."""
    S = sparse.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=(n, d))
    if B == A:
        return (S.T @ S).toarray()
    Sa, Sb = S[:, :config.D_MLP], S[:, config.D_MLP:]
    return (Sa.T @ Sb).toarray()


def _within(counts):
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return np.arange(ptr[-1]) - np.repeat(ptr[:-1], counts), ptr


def curveball_cpu(ind0: np.ndarray, rng: np.random.Generator, sweeps=20) -> np.ndarray:
    """Curveball shuffle: `sweeps` rounds of pairing rows at random and redistributing their non-shared items."""
    ind = ind0.copy()
    for _ in range(sweeps):
        perm = rng.permutation(n)
        half = n // 2
        a, b = perm[:half], perm[half: 2 * half]
        la_, lb_ = lens[a], lens[b]
        tot = la_ + lb_
        wa, _ = _within(la_); wb, _ = _within(lb_); _, seg_ptr = _within(tot)
        m = seg_ptr[-1]
        item = np.empty(m, dtype=np.int32)
        item[np.repeat(seg_ptr[:-1], la_) + wa] = ind[np.repeat(indptr[a], la_) + wa]
        item[np.repeat(seg_ptr[:-1] + la_, lb_) + wb] = ind[np.repeat(indptr[b], lb_) + wb]
        seg = np.repeat(np.arange(half, dtype=np.int64), tot)
        order = np.argsort(seg * (d + 1) + item, kind="stable")
        sv, ss = item[order], seg[order]
        dup = np.zeros(m, dtype=bool)
        dup[:-1] = (sv[:-1] == sv[1:]) & (ss[:-1] == ss[1:])
        is_common = dup.copy(); is_common[1:] |= dup[:-1]
        c_cnt = np.bincount(ss[dup], minlength=half)
        keep_a = la_ - c_cnt
        pool_seg, pool_val = ss[~is_common], sv[~is_common]
        po = np.lexsort((rng.random(len(pool_seg)), pool_seg))
        ps, pv = pool_seg[po], pool_val[po]
        rank, _ = _within(np.bincount(ps, minlength=half))
        to_a = rank < keep_a[ps]
        cw, _ = _within(c_cnt)
        ind[np.repeat(indptr[a], c_cnt) + cw] = sv[dup]
        ind[np.repeat(indptr[b], c_cnt) + cw] = sv[dup]
        aw, _ = _within(keep_a)
        ind[np.repeat(indptr[a] + c_cnt, keep_a) + aw] = pv[to_a]
        nb = lb_ - c_cnt
        bw, _ = _within(nb)
        ind[np.repeat(indptr[b] + c_cnt, nb) + bw] = pv[~to_a]
    return ind


def worker(seeds):
    """Accumulate sum and sum of squares of the null co-occurrence block over the given replicate seeds."""
    tot = None; tot_sq = None
    for s in seeds:
        Cn = cooc_block(curveball_cpu(indices, np.random.default_rng(s)))
        if tot is None:
            tot = Cn.astype(np.float64); tot_sq = tot * tot
        else:
            tot += Cn; tot_sq += Cn.astype(np.float64) ** 2
        del Cn
    return tot, tot_sq


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world", type=Path, required=True, help="world dir holding dedup_kept.npz and merged/L??.npz")
    parser.add_argument("--census-dir", type=Path, default=None, help="output dir (default <world>/census)")
    parser.add_argument("--layer-a", type=int, default=None)
    parser.add_argument("--layer-b", type=int, default=None)
    parser.add_argument("--tile", type=int, default=None, help="tile index into the (a, b) pairs of --layers instead of --layer-a/--layer-b")
    parser.add_argument("--layers", default=f"{config.LAYER_TILE[0]}-{config.LAYER_TILE[1]}",
                        help="layer range 'lo-hi' the tiles cover (default the configured discovery window)")
    parser.add_argument("--n-tiles", action="store_true", help="print the number of tiles of --layers and exit")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=config.N_REPLICATES, help="curveball null replicates")
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--c-min", type=int, default=config.C_MIN, help="minimum observed co-occurrence count of a stored pair")
    parser.add_argument("--z-store", type=float, default=config.Z_STORE, help="minimum z of a stored pair")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--force", action="store_true", help="recompute a tile whose output file exists")


def run(args: argparse.Namespace) -> None:
    global MD, A, B, n, d, indptr, indices, lens, kept
    lo, hi = (int(x) for x in args.layers.split("-"))
    all_tiles = tiles((lo, hi))
    if args.n_tiles:
        print(len(all_tiles))
        return
    if args.tile is not None:
        A, B = all_tiles[args.tile]
    elif args.layer_a is not None and args.layer_b is not None:
        A, B = args.layer_a, args.layer_b
    else:
        raise SystemExit("give --tile k or both --layer-a and --layer-b")
    MD = args.world
    out = args.census_dir or MD / "census"
    final = out / f"pairs_L{A:02d}_L{B:02d}{args.out_suffix}.npz"
    if final.exists() and not args.force:
        print(f"tile L{A}-L{B} present: {final}")
        return
    t0 = time.monotonic()
    kept = np.load(MD / "dedup_kept.npz")["kept"]

    ptr_a, idx_a = load_csr(A, 0)
    if B != A:
        ptr_b, idx_b = load_csr(B, config.D_MLP)
        la, lb = np.diff(ptr_a), np.diff(ptr_b)
        indptr = np.zeros(len(kept) + 1, dtype=np.int64)
        np.cumsum(la + lb, out=indptr[1:])
        indices = np.empty(indptr[-1], dtype=np.int32)
        indices[np.repeat(indptr[:-1], la) + (np.arange(la.sum()) - np.repeat(ptr_a[:-1], la))] = idx_a
        indices[np.repeat(indptr[:-1] + la, lb) + (np.arange(lb.sum()) - np.repeat(ptr_b[:-1], lb))] = idx_b
        d = 2 * config.D_MLP
    else:
        indptr, indices, d = ptr_a, idx_a, config.D_MLP
    n = len(kept)
    lens = np.diff(indptr).astype(np.int64)
    freq = np.bincount(indices, minlength=d).astype(np.int64)
    print(f"L{A}xL{B}: S ({n:,}, {d}), slots {indptr[-1]:,} ({time.monotonic() - t0:.0f}s)", flush=True)

    C = cooc_block(indices)
    print(f"observed counts done ({time.monotonic() - t0:.0f}s); {args.replicates} null replicates on {args.workers} workers", flush=True)
    seeds = [args.seed_base + r for r in range(args.replicates)]
    chunks = [seeds[k::args.workers] for k in range(args.workers) if seeds[k::args.workers]]
    total = np.zeros(C.shape, np.float64); total_sq = np.zeros(C.shape, np.float64)
    with Pool(len(chunks)) as pool:
        for tot, tot_sq in pool.imap_unordered(worker, chunks):
            total += tot; total_sq += tot_sq
    mean = total / args.replicates
    std = np.sqrt(np.clip(total_sq / args.replicates - mean * mean, 0.0, None))
    del total, total_sq
    z = (C - mean) / np.maximum(std, np.sqrt(np.clip(mean, 0.25, None)))
    del std
    print(f"null done ({time.monotonic() - t0:.0f}s)", flush=True)

    sig = (C >= args.c_min) & (z >= args.z_store)
    if B == A:
        sig &= np.triu(np.ones(C.shape, dtype=bool), 1)
    i_idx, j_idx = np.nonzero(sig)
    j_out = j_idx if B == A else j_idx + config.D_MLP           # cross-tile j indices carry the D_MLP offset
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".pairs_L{A:02d}_L{B:02d}{args.out_suffix}.tmp.npz"
    np.savez_compressed(tmp, i=i_idx.astype(np.int32), j=j_out.astype(np.int32),
                        z=z[sig].astype(np.float32), c=C[sig].astype(np.int32), mean=mean[sig].astype(np.float32),
                        freq_i=freq[i_idx].astype(np.int32), freq_j=freq[j_out].astype(np.int32),
                        n_rows=np.array([n], np.int64), layers=np.array([A, B]), replicates=np.array([args.replicates]))
    os.replace(tmp, final)
    print(f"L{A}xL{B}: {int(sig.sum()):,} pairs stored (z >= {args.z_store:g}, c >= {args.c_min}); "
          f"z >= {config.CPM_EDGE_Z:g}: {int((sig & (z >= config.CPM_EDGE_Z)).sum()):,}  ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
