import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from affect_geometry.analysis.cohorts import balanced_cohorts
from affect_geometry.analysis.geometry_comparisons import linear_cka
from affect_geometry.analysis.geometry_inference import corrected_pvalues, permutation_schedule
from affect_geometry.analysis.geometry_preprocessing import preprocess
from affect_geometry.analysis.geometry_validation import evaluate, fit_similarity, pooled_r2
from affect_geometry.analysis.nonlinearity import fidelity
from affect_geometry.analysis.topology import persistence
from affect_geometry.embeddings.diffusion_maps import diffusion_maps
from affect_geometry.embeddings.geometry import (
    UnsupportedGeometry,
    isomap_decomposition,
    mds_decomposition,
)
from affect_geometry.persistence.correctness_geometry import GeometryStore


def source_rows(labels=("anger", "joy", "neutral", "sadness", "fear", "love")):
    rows = []
    for label in labels:
        for correct, count in [(True, 102), (False, 8)]:
            for i in range(count):
                rows.append(
                    dict(
                        sample_id=f"{label}-{correct}-{i:03d}",
                        emotion=label,
                        compat_correct=correct,
                        status="completed",
                        source_split=("train" if i % 2 else "test"),
                        text="example",
                        text_length=7,
                        compat_valid_label=correct,
                        prompt_hash="fixture",
                        correct=False,
                        valid_label=False,
                    )
                )
    return pd.DataFrame(rows)


def test_cohorts_exact_partition_invalids_and_boundary():
    rows = source_rows()
    boundary = rows.loc[rows.emotion.eq("anger")].copy()
    boundary["emotion"] = "boundary"
    boundary.sample_id = "boundary-" + boundary.sample_id
    boundary = boundary.drop(boundary.loc[boundary.compat_correct].index[:2])
    unresolved = rows.iloc[:1].assign(sample_id="error", status="error", compat_correct=None)
    rows = pd.concat([rows, boundary, unresolved], ignore_index=True)
    labels = sorted(rows.emotion.unique())
    a = balanced_cohorts(rows, labels, 42)
    b = balanced_cohorts(rows.sample(frac=1, random_state=5), labels, 42)
    assert a.samples.sample_id.tolist() == b.samples.sample_id.tolist()
    assert "boundary" not in a.labels
    assert len(a.unresolved) == 1
    correct = set(a.samples.loc[a.samples.cohort.eq("correct"), "sample_id"])
    incorrect = set(a.samples.loc[a.samples.cohort.eq("incorrect"), "sample_id"])
    union = set(a.samples.loc[a.samples.cohort.eq("all"), "sample_id"])
    assert not correct & incorrect and correct | incorrect == union
    assert len(correct) == len(incorrect) == 8 * 6
    assert not a.samples.loc[a.samples.cohort.eq("incorrect"), "compat_valid_label"].any()


def test_projection_cohort_and_neutral_retention():
    rows = source_rows()
    rng = np.random.default_rng(3)
    values = rng.normal(size=(len(rows), 9))
    labels = sorted(rows.emotion.unique())
    pre = preprocess(values, rows, labels)
    assert len(pre["R"]) == len(labels) and "neutral" in labels
    assert np.allclose(pre["R"] @ pre["components"].T, 0, atol=1e-10)
    assert np.allclose(np.linalg.norm(pre["U"], axis=1), 1)
    assert np.allclose(pre["D"], 1 - pre["S"])
    mask = rows.compat_correct.to_numpy()
    other = preprocess(values[mask], rows.loc[mask], labels)
    assert not np.allclose(pre["neutral_mean"], other["neutral_mean"])
    with pytest.raises(UnsupportedGeometry):
        preprocess(np.zeros_like(values), rows, labels)


def test_mds_strict_rank_and_cosine_double_centering():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 5))
    x /= np.linalg.norm(x, axis=1)[:, None]
    d = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(d, 0)
    result = mds_decomposition(d)
    j = np.eye(8) - np.ones((8, 8)) / 8
    assert np.allclose(result.arrays["gram"], -0.5 * j @ (d * d) @ j)
    assert np.allclose(result.coordinates(1)[:, 0], result.coordinates(2)[:, 0])
    line = np.arange(5)[:, None]
    with pytest.raises(UnsupportedGeometry):
        mds_decomposition(squareform(pdist(line))).coordinates(2)


def test_isomap_rejects_disconnected_and_preserves_zero_edges():
    x = np.array([[0], [0], [10], [11]], float)
    d = squareform(pdist(x))
    with pytest.raises(UnsupportedGeometry, match="disconnected"):
        isomap_decomposition(d, 1, list("abcd"))
    full = isomap_decomposition(d, 3, list("abcd"))
    assert full.reference[0, 1] == 0
    theta = np.linspace(0, np.pi, 20)
    d = squareform(pdist(np.c_[np.cos(theta), np.sin(theta)]))
    result = isomap_decomposition(d, 2, [str(i) for i in range(20)])
    assert result.reference[0, -1] > d[0, -1]
    assert np.corrcoef(result.coordinates(1)[:, 0], theta)[0, 1] ** 2 > 0.99


def test_diffusion_markov_stationary_full_spectrum_and_row_invariance():
    theta = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    x = np.c_[np.cos(theta), np.sin(theta)]
    d = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(d, 0)
    ids = list("abcdefghi")
    result = diffusion_maps(d, 3, 2, 2, ids)
    p, pi, psi = (result.arrays[k] for k in ("transition", "stationary", "markov_eigenvectors"))
    assert np.allclose(p.sum(1), 1)
    assert np.allclose(pi @ p, pi)
    assert np.allclose(psi.T @ np.diag(pi) @ psi, np.eye(9))
    assert np.allclose(p @ psi, psi * result.eigenvalues)
    assert np.allclose(
        result.reference, squareform(pdist(psi[:, 1:] * result.eigenvalues[1:] ** 2))
    )
    assert np.allclose(result.coordinates(1), result.coordinates(2)[:, :1])
    order = np.array([8, 2, 5, 0, 1, 4, 7, 3, 6])
    moved = diffusion_maps(d[order][:, order], 3, 2, 2, [ids[i] for i in order])
    # Non-tied graph construction is invariant; exact distance ties use stable emotion IDs.
    assert np.allclose(moved.reference, result.reference[order][:, order])


def test_alignment_no_target_leakage_and_negative_r2():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(8, 2))
    y = x @ np.array([[0, 1], [-1, 0]]) * 2 + 3
    result = evaluate({"fixed": x}, y, selection="fixed")
    assert result["cv_r2"] == pytest.approx(1)
    assert result["full_r2"] == pytest.approx(1)
    changed = y.copy()
    changed[0] += 100
    candidates = {"a": x, "b": rng.normal(size=(8, 2))}
    second = evaluate(candidates, y)
    third = evaluate(candidates, changed)
    assert second["folds"][0]["setting"] == third["folds"][0]["setting"]
    assert second["folds"][0]["inner_scores"] == third["folds"][0]["inner_scores"]
    assert np.allclose(second["predictions"][0], third["predictions"][0])
    # Single-candidate outer prediction cannot depend on the held-out target.
    altered = evaluate({"fixed": x}, changed, selection="fixed")
    assert np.allclose(result["predictions"][0], altered["predictions"][0])
    assert altered["cv_r2"] < 0
    assert 0 not in result["folds"][0]["training"]
    with pytest.raises(UnsupportedGeometry):
        fit_similarity(np.ones((4, 2)), np.ones((4, 2)))
    with pytest.raises(UnsupportedGeometry):
        pooled_r2(np.ones((4, 1)), np.ones((4, 1)), np.ones((4, 1)))


def test_topology_known_square_and_triangle():
    square = squareform(pdist(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])))
    diagrams, betti = persistence(square, bound=2)
    h1 = diagrams.loc[diagrams.dimension.eq(1) & (diagrams.death > diagrams.birth)]
    assert len(h1) == 1
    assert h1.iloc[0].birth == 1
    assert h1.iloc[0].death == pytest.approx(np.sqrt(2))
    assert betti.loc[betti.filtration.eq(0) & betti.dimension.eq(0), "betti"].iloc[0] == 4
    triangle = np.ones((3, 3)) - np.eye(3)
    diag, _ = persistence(triangle)
    assert not ((diag.dimension == 1) & (diag.death > diag.birth)).any()


def test_stress_scale_cka_and_family_failure():
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    d = squareform(pdist(x))
    metrics, _ = fidelity(d, 10 * x, [1, 2])
    assert metrics["stress"] == pytest.approx(0)
    assert np.isnan(metrics["trustworthiness_k2"])
    assert linear_cka(x, 4 * x + 10) == pytest.approx(1)
    assert np.isnan(linear_cka(np.ones_like(x), x))
    raw, corrected = corrected_pvalues([1, 2], np.array([[0, 0], [2, 1], [3, 3]]))
    assert np.all(corrected >= raw)
    raw, corrected = corrected_pvalues([1, np.nan], np.zeros((3, 2)))
    assert np.isnan(corrected).all()
    with pytest.raises(ValueError):
        permutation_schedule(4, 2)


def test_store_invalidates_corruption_and_provenance(tmp_path):
    store = GeometryStore(tmp_path, {"run": "test"})
    path = store.arrays("geometry/a.npz", x=np.arange(3))
    store.complete("a", {"cohort": "correct"}, [path])
    assert store.valid("a", {"cohort": "correct"})
    assert not store.valid("a", {"cohort": "all"})
    path.write_bytes(b"broken")
    assert not store.valid("a", {"cohort": "correct"})
    with pytest.raises(ValueError):
        GeometryStore(tmp_path, {"run": "changed"})
