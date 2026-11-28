"""Countable fixtures and exact parity against the pinned Goodfire reference."""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from affect_geometry.experiments.neuron_transfer_config import NeuronTransferConfig
from affect_geometry.neurons import UPSTREAM_CENSUS_SHA
from affect_geometry.neurons.communities import (
    control_members,
    discover_communities,
    select_communities,
)
from affect_geometry.neurons.graph import census, curveball


def test_pinned_curveball_preserves_marginals_and_matches_reference():
    path = Path(__file__).parents[1] / "third_party/goodfire-neuron-communities/census.py"
    data = path.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == UPSTREAM_CENSUS_SHA
    )
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    tree = ast.parse(data)
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"_within", "curveball_cpu"}
    ]
    namespace = dict(
        np=np,
        n=matrix.shape[0],
        d=matrix.shape[1],
        indptr=matrix.indptr.astype(np.int64),
        lens=np.diff(matrix.indptr),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    expected = namespace["curveball_cpu"](matrix.indices, np.random.default_rng(42), sweeps=7)
    actual = curveball(matrix, np.random.default_rng(42), sweeps=7)
    assert np.array_equal(actual.indices, expected)
    assert np.array_equal(actual.sum(0), matrix.sum(0))
    assert np.array_equal(actual.sum(1), matrix.sum(1))
    assert set(actual.data) == {1}


def test_sparse_counts_and_stabilized_null_match_dense_calculation(tmp_path):
    config = NeuronTransferConfig(
        str(tmp_path),
        "test",
        minimum_count=1,
        edge_threshold=0.01,
        null_replicates=3,
        null_sweeps=2,
    )
    a = sparse.csr_matrix(np.array([[1, 1, 0]] * 8 + [[0, 0, 1]] * 8))
    b = sparse.csr_matrix(np.array([[1, 0]] * 8 + [[0, 1]] * 8))
    edges, metadata = census({0: a, 1: b}, config)
    assert len(edges)
    for row in edges.itertuples():
        left, right = {0: a, 1: b}[row.layer_i], {0: a, 1: b}[row.layer_j]
        expected_count = int((left.T @ right)[row.channel_i, row.channel_j])
        assert row.count == expected_count
        assert row.null_scale == pytest.approx(max(row.null_std, np.sqrt(max(row.null_mean, 0.25))))
        assert row.excess == pytest.approx((row.count - row.null_mean) / row.null_scale)
    assert metadata["scale"].startswith("max(population_std")


def test_leiden_pools_exact_groups_and_controls_match_layer_counts(tmp_path):
    pytest.importorskip("leidenalg")
    config = NeuronTransferConfig(str(tmp_path), "test", resolutions=(1, 2))
    edges = pd.DataFrame(
        [
            dict(layer_i=0, channel_i=0, layer_j=1, channel_j=1, excess=30),
            dict(layer_i=0, channel_i=2, layer_j=1, channel_j=3, excess=30),
        ]
    )
    attrs = pd.DataFrame(
        [
            dict(sample_id=f"s{i}", layer=layer, channel=channel, importance=float(channel + 1))
            for i in range(3)
            for layer, channel in ((0, 0), (1, 1))
        ]
    )
    candidates = discover_communities(edges, attrs, {"revision": "pinned"}, config)
    assert len(candidates) == 2
    assert all(c["resolutions"] == [1, 2] for c in candidates)
    means = pd.DataFrame(
        [dict(layer=layer, channel=j, importance=float(j)) for layer in (0, 1) for j in range(8)]
    )
    controls = control_members(candidates[0], {0: 8, 1: 8}, means, 42)
    for group in controls.values():
        assert sorted(v[0] for v in group) == [0, 1]
        assert not set(map(tuple, group)) & set(map(tuple, candidates[0]["members"]))
    assert controls == control_members(candidates[0], {0: 8, 1: 8}, means, 42)


def test_selection_uses_signed_effects_diversity_and_training_only(tmp_path):
    config = NeuronTransferConfig(
        str(tmp_path), "test", community_budget=2, minimum_effect_examples=2
    )
    candidates = [
        dict(community_id=name, members=members)
        for name, members in (
            ("a", [[0, 0], [0, 1]]),
            ("a-copy", [[0, 0], [0, 1]]),
            ("b", [[0, 2], [0, 3]]),
            ("negative", [[0, 4], [0, 5]]),
        )
    ]
    effects = pd.DataFrame(
        [
            dict(
                community_id=c["community_id"],
                emotion=e,
                sample_id=str(i),
                net_effect=(
                    -2
                    if c["community_id"] == "negative"
                    else 2
                    if (c["community_id"].startswith("a") and e == "joy")
                    or (c["community_id"] == "b" and e == "fear")
                    else 0
                ),
            )
            for c in candidates
            for e in ("joy", "fear")
            for i in range(3)
        ]
    )
    chosen, support = select_communities(candidates, effects, 1e-5, config)
    assert [c["community_id"] for c in chosen] == ["a", "b"]
    assert not support.loc[support.community_id.eq("negative"), "qualifies"].any()
