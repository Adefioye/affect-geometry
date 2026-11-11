"""Build a descriptive EXP3 report from persisted results; never rerun an experiment.

Run from the repository root with the affect-geometry Python environment.
Report figures live beside the Markdown; their source tables stay under outputs/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affect_geometry.logging import configure_file_logging, get_logger, progress

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "analyses/exp3/analysis.md"
PLOTS = REPORT.parent / "plots"
ASSETS = ROOT / "outputs/exp3/analyses/three-models"
RUNS = {
    "gemma2": ROOT / "outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive",
    "apertus": ROOT / "outputs/exp3/apertus/rq3-analysis-full-v1-descriptive",
    "olmo2": ROOT / "outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive",
}
MODELS = list(RUNS)
COHORTS = ["correct", "incorrect", "all"]
METHODS = ["mds", "isomap", "diffusion_maps"]
MODEL_NAMES = {"gemma2": "Gemma-2", "apertus": "Apertus", "olmo2": "OLMo-2"}
METHOD_NAMES = {"mds": "MDS", "isomap": "Isomap", "diffusion_maps": "DM"}
COLORS = {"mds": "#2864a6", "isomap": "#d78316", "diffusion_maps": "#198c76"}
TABLES: dict[str, pd.DataFrame] = {}
MANIFESTS = {}
AUDIT = []
CATALOG = []
FIGURES = []
parts = []


def relative(path):
    return Path(os.path.relpath(path, REPORT.parent)).as_posix()


def link(path, label=None):
    return f"[{label or Path(path).name}]({relative(path)})"


def add(text):
    parts.append(text.strip() + "\n")


def table(rows, columns):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.3f}"
        return str(value).replace("|", " / ").replace("\n", " ")

    add(
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join("---" for _ in columns)
        + " |\n"
        + "\n".join("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    )


def figure(name, data, render, caption):
    """Serialize first, then plot only data reread from the independent CSV."""
    csv = ASSETS / "plot_data" / f"{name}.csv"
    png = PLOTS / f"{name}.png"
    data.to_csv(csv, index=False)
    fig = render(pd.read_csv(csv, low_memory=False))
    fig.savefig(png, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append((name, csv, png))
    add(f"![{caption}]({relative(png)})\n\n{caption} Source: {link(csv, 'plot-ready CSV')}.")


def panel_curves(data, value="value", y_label="", hline=None, full=False):
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True
    )
    for i, model in enumerate(MODELS):
        for j, cohort in enumerate(COHORTS):
            ax = axes[i, j]
            group = data[(data.model == model) & (data.cohort == cohort)]
            for method in METHODS:
                g = group[group.method == method].sort_values("depth")
                if not len(g):
                    continue
                ax.plot(g.depth, g[value], color=COLORS[method], lw=1.7, label=METHOD_NAMES[method])
                if full:
                    ax.plot(g.depth, g.full_r2, color=COLORS[method], lw=1, ls="--", alpha=0.5)
            if hline is not None:
                ax.axhline(hline, color=".55", lw=0.6)
            ax.set_title(f"{MODEL_NAMES[model]} · {cohort}", fontsize=10)
            ax.grid(alpha=0.18)
            if i == 2:
                ax.set_xlabel("Normalized block depth (layer + 1) / L")
            if j == 0:
                ax.set_ylabel(y_label)
    axes[0, 0].legend(fontsize=8)
    return fig


def label_points(ax, frame):
    """Place emotion names with deterministic collision avoidance and light leaders."""
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    points = ax.transData.transform(frame[["x", "y"]].to_numpy())
    limits = ax.get_window_extent(renderer)
    for point, (_, row) in zip(points, frame.iterrows(), strict=True):
        width = max(12, len(row.emotion) * 4.4)
        height = 10
        candidates = []
        for distance in (5, 17, 29, 41, 53, 65, 77):
            for dx, dy in [
                (distance, distance),
                (distance, -distance),
                (-distance - width, distance),
                (-distance - width, -distance),
                (5, distance),
                (5, -distance),
            ]:
                box = [point[0] + dx, point[1] + dy, point[0] + dx + width, point[1] + dy + height]
                overlap = sum(
                    max(0, min(box[2], b[2]) - max(box[0], b[0]))
                    * max(0, min(box[3], b[3]) - max(box[1], b[1]))
                    for b in occupied
                )
                outside = (
                    max(limits.x0 - box[0], 0)
                    + max(box[2] - limits.x1, 0)
                    + max(limits.y0 - box[1], 0)
                    + max(box[3] - limits.y1, 0)
                )
                candidates.append((overlap * 100 + outside * 1000 + abs(dx) + abs(dy), dx, dy, box))
        _, dx, dy, box = min(candidates, key=lambda v: v[0])
        occupied.append([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2])
        ax.annotate(
            row.emotion,
            (row.x, row.y),
            fontsize=6.5,
            xytext=(dx * 72 / ax.figure.dpi, dy * 72 / ax.figure.dpi),
            textcoords="offset points",
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.35, color=".65"),
        )


def load():
    names = [
        "scores",
        "aggregates",
        "selected_layer_scores",
        "cka",
        "contrasts",
        "paired_errors",
        "metrics",
        "fidelity_aggregates",
        "persistence",
        "betti",
        "spectra",
        "coordinates",
        "graph_ratios",
        "failures",
        "rdm_contrasts",
        "held_out_predictions",
        "composition",
        "support",
        "counts",
    ]
    frames = {name: [] for name in names}
    for model in progress(MODELS, description="Audit saved EXP3 model results", unit="model"):
        run = RUNS[model]
        manifest = json.loads((run / "manifest.json").read_text())
        MANIFESTS[model] = manifest
        checkpoint = json.loads((run / "checkpoints/run.json").read_text())
        artifacts = checkpoint["artifacts"]
        for name in names:
            path = run / "plot_data" / f"{name}.csv"
            frame = pd.read_csv(path, low_memory=False)
            assert frame.empty or frame.model.eq(model).all(), (model, name, "model mismatch")
            if "run_id" in frame:
                assert frame.empty or frame.run_id.eq(run.name).all(), (model, name, "run mismatch")
            if "emotion_order" in frame:
                assert (
                    frame.empty or frame.emotion_order.eq("|".join(manifest["emotion_order"])).all()
                )
            if "emotion_set_id" in frame:
                assert frame.empty or frame.emotion_set_id.nunique() == 1
            expected = artifacts[path.relative_to(run).as_posix()]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            frames[name].append(frame)
            CATALOG.append((model, name, len(frame), path))
        samples = pd.read_csv(
            run / "cohorts/samples.csv",
            usecols=[
                "cohort",
                "sample_id",
                "emotion",
                "source_split",
                "text_length",
                "compat_correct",
                "compat_valid_label",
                "correct",
                "duplicate_text_group",
                "input_token_count",
                "activation_token_position",
            ],
        )
        s = samples[samples.cohort == "correct"]
        w = samples[samples.cohort == "incorrect"]
        a = samples[samples.cohort == "all"]
        assert not set(s.sample_id) & set(w.sample_id)
        assert set(a.sample_id) == set(s.sample_id) | set(w.sample_id)
        assert samples.duplicated(["cohort", "sample_id"]).sum() == 0
        assert s.compat_correct.all() and not w.compat_correct.any()
        assert (samples.activation_token_position == samples.input_token_count - 1).all()
        support = frames["support"][-1]
        assert int(support.correct.sum() + support.incorrect.sum()) == 45446
        assert support.eligible.eq(support.correct > 100).all()
        assert support.loc[support.status.eq("ok"), "emotion"].tolist() == manifest["emotion_order"]
        count = samples.groupby(["cohort", "emotion"]).size()
        for emotion in manifest["emotion_order"]:
            n = int(support.set_index("emotion").loc[emotion, "sampled_per_subset"])
            assert count["correct", emotion] == count["incorrect", emotion] == n
            assert count["all", emotion] == 2 * n
        scores = frames["scores"][-1]
        assert not scores.duplicated(["cohort", "layer", "method", "dimension"]).any()
        assert len(scores) == len(manifest["layers"]) * 18
        assert scores.status.eq("ok").all() and np.isfinite(scores.cv_r2).all()
        assert scores.corrected_p.isna().all()
        target = pd.read_csv(run / "validation/targets.csv")
        assert target.emotion.tolist() == manifest["emotion_order"]
        mapping = pd.DataFrame(manifest["nrc"]["mapping"]).set_index("emotion")
        assert np.allclose(
            target[["valence", "arousal"]], mapping.loc[target.emotion, ["valence", "arousal"]]
        )
        for _, row in frames["aggregates"][-1].iterrows():
            group = scores[
                (scores.cohort == row.cohort)
                & (scores.method == row.method)
                & (scores.dimension == row.dimension)
            ]
            assert set(group.layer) == set(manifest["layers"])
            assert np.isclose(group.cv_r2.mean(), row.A_M, atol=1e-12)
            baseline = frames["aggregates"][-1]
            baseline = baseline[
                (baseline.cohort == row.cohort)
                & (baseline.dimension == row.dimension)
                & (baseline.method == "mds")
            ].iloc[0]
            assert np.isclose(row.A_M - baseline.A_M, row.delta_A_M, atol=1e-12)
        cka = frames["cka"][-1]
        for _, group in cka.groupby(["cohort", "dimension"]):
            matrix = (
                group.pivot(index="method_left", columns="method_right", values="cka")
                .reindex(index=METHODS, columns=METHODS)
                .to_numpy()
            )
            assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
            assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
        for cohort in COHORTS:
            g = samples[samples.cohort == cohort]
            AUDIT.append(
                dict(
                    model=model,
                    cohort=cohort,
                    n=len(g),
                    mean_length=g.text_length.mean(),
                    invalid=int((~g.compat_valid_label).sum()),
                    strict_rescued=int((g.compat_correct & ~g.correct).sum()),
                    train=(g.source_split == "train").mean(),
                    test=(g.source_split == "test").mean(),
                    validation=(g.source_split == "validation").mean(),
                    duplicate_groups=g.duplicate_text_group.nunique(),
                    nrc_K=len(manifest["emotion_order"]),
                )
            )
        get_logger().info(
            "audit.complete model=%s layers=%d scores=%d cohort_identity=ok",
            model,
            len(manifest["layers"]),
            len(scores),
        )
    for name, values in frames.items():
        TABLES[name] = pd.concat(values, ignore_index=True)
    # Keep chosen-setting diagnostics separate from the full candidate grid.
    keys = ["model", "cohort", "layer", "method", "dimension", "setting"]
    TABLES["chosen_fidelity"] = TABLES["metrics"].merge(
        TABLES["scores"][keys], on=keys, how="inner", validate="many_to_one"
    )


def get_aggregate(model, cohort, method, dimension):
    f = TABLES["aggregates"]
    return f[
        (f.model == model)
        & (f.cohort == cohort)
        & (f.method == method)
        & (f.dimension == dimension)
    ].iloc[0]


def aggregate_rows(dimension):
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            a = [get_aggregate(model, cohort, m, dimension) for m in METHODS]
            rows.append(
                [MODEL_NAMES[model], cohort, *[x.A_M for x in a], a[1].delta_A_M, a[2].delta_A_M]
            )
    return rows


def run_section():
    add("""# EXP3: correctness-conditioned affective geometry across three models

Analysis of saved results, aligned to the supplied **EXP3 implementation design**
(working draft).
The complete descriptive runs cover all/correct/incorrect cohorts, MDS, Isomap and
Diffusion Maps
(DM), 1D valence and 2D valence–arousal (VA), at every transformer block. Tables below
summarize
every saved result family; detailed layer, setting, emotion and pair observations remain
linked
to their canonical CSVs. Rounding is for presentation only.

**Main finding.** Correct cohorts have higher mean layerwise VA recovery for every
model–method
pair. Incorrect cohorts retain substantial valence and VA recovery. DM improves mean VA
recovery
over MDS in all nine model–cohort cells, but its advantage often accompanies weak
arousal
prediction and does not survive every layer-selection comparison. The geometry therefore
supports
correctness-associated affective organization, with descriptive nonlinear-method
advantages;
it does **not** establish a nonlinear, intrinsically two-dimensional VA manifold.

**Inference status.** These are full-layer, non-smoke **descriptive** runs. Each
requested 2,000
permutations but explicitly completed zero in both target dimensions
(`user_requested_skip`).
No corrected p-values, first significant layers, significant intervals, prompt-sampling
confidence
intervals, or significance claims are available. The unfinished Gemma/Apertus
inferential runs
are not substituted for the completed descriptive runs.

## 1. Scope, provenance and design alignment

The question is whether emotion-centroid representations recover human valence and
arousal,
whether nonlinear embeddings improve on classical MDS, and how recovery differs with
classification
correctness. Native layer indices are zero-based; normalized depth is `(layer + 1) / L`.
""")
    rows = []
    for model in MODELS:
        d = MANIFESTS[model]
        rows.append(
            [
                MODEL_NAMES[model],
                d["source"]["model_id"],
                RUNS[model].name,
                f"0–{max(d['layers'])} ({len(d['layers'])})",
                len(d["emotion_order"]),
                f"{len(d['layers']) * 18}/{len(d['layers']) * 18}",
            ]
        )
    table(
        rows,
        ["Model", "Model identifier", "Run ID", "Block layers", "Emotions K", "Valid score cells"],
    )
    add("""All sources contain 45,446 completed, single-label GoEmotions examples (36,308 train,
4,548 validation, 4,590 test), with zero unresolved outcomes. Cohort membership uses
saved
`compat_correct` from `choi_weber_first_word_v1`, not strict correctness. Seed 42
determines
one balanced draw per model. Representations are final non-padding classification-prompt
states
before answer generation, with separate model/layer/cohort neutral PCA fits at variance
threshold
0.5 and equal-weight non-neutral-centroid centering. Neutral remains in all analyses.
Cosine dissimilarity `1 − cosine_similarity` is the only distance.
Persisted projected sample states use float32 (unnormalized); downstream centroids are
normalized.
Gemma has hidden width 3,584; Apertus and OLMo have width 4,096. The selected OLMo run
uses the
completed `bs8` activation bundle, not an earlier incomplete extraction.

NRC-VAD v2.1 supplies unrescaled `[-1, 1]` valence and arousal, with exact mappings for
all
28 source labels, including neutral. All three runs share reference SHA-256
`5021c35285b626e19ba06974d14d0782c07ea4b7a381ed9415f529455dd035dd`.
The reference provenance records a university-hosted copy after the official download
returned
HTTP 406. Model revisions and source dataset revision are retained in the manifest links
below.

Outer evaluation hides one emotion's complete NRC row; hyperparameter selection uses
inner emotion
folds. Activation points and pooled neutral preprocessing remain transductive. R²
predicts
held-out **ratings**, not unseen prompts or unseen activation points. Procrustes permits
translation,
rotation/reflection and scale, so recovered axes need not be native model axes.
""")
    table(
        [
            [
                "Cohort eligibility/balancing",
                ">100 correct; min(correct, incorrect) per subset; exact union",
                "Verified from saved IDs and counts",
            ],
            [
                "Coverage",
                "3 models × 3 cohorts × 3 methods × 2 targets; every block",
                "1,908/1,908 valid score cells; 54/54 aggregates",
            ],
            [
                "Selection",
                "Nested LOEO settings; separate nested layer-and-setting scores",
                "Saved for every model/cohort/method/target",
            ],
            [
                "Graphs",
                "Union kNN; k ∈ {3,5,7,10}; deterministic emotion-ID ties",
                "Disconnected candidates explicitly rejected",
            ],
            [
                "Diffusion",
                "k ∈ {3,5,7,10}; scale rank ∈ {3,5}; t ∈ {1,2,4}; α=1",
                "24 primary settings; no α sensitivity runs",
            ],
            [
                "Diagnostics",
                "Cosine/native fidelity; spectral dimensions; H0/H1; full CKA",
                "Available; stability scope discussed below",
            ],
            [
                "Permutation inference",
                "2,000 complete NRC-row permutations; separate max-stat families",
                "Skipped: 0/2,000 for each target in each model",
            ],
            [
                "Cross-model matched support",
                "Common eligible emotion intersection",
                "16 common labels identified; no saved matched-support comparison run",
            ],
        ],
        ["EXP3 requirement", "Policy", "Saved-result assessment"],
    )
    add("""The supplied draft is an implementation guide, not a frozen preregistration. The
manifests mark
settings frozen and `pilot_verified=false`. This report audits saved outputs; it does
not certify
extraction numerics or re-execute the pipeline. Selected source/plot tables were checked
against
the run checkpoint hashes, and primary means were independently checked from saved layer
scores.
""")


def cohorts_section():
    add("## 2. Cohort support, exclusions and composition")
    rows = []
    for model in MODELS:
        s = TABLES["support"].query("model == @model")
        a = next(x for x in AUDIT if x["model"] == model and x["cohort"] == "incorrect")
        neutral = s[s.emotion == "neutral"].iloc[0]
        rows.append(
            [
                MODEL_NAMES[model],
                f"{int(s.correct.sum()):,}",
                f"{s.correct.sum() / (s.correct.sum() + s.incorrect.sum()):.2%}",
                len(MANIFESTS[model]["emotion_order"]),
                f"{a['n']:,}",
                f"{2 * a['n']:,}",
                int(neutral.sampled_per_subset),
                f"{a['invalid']:,} ({a['invalid'] / a['n']:.2%})",
            ]
        )
    table(
        rows,
        [
            "Model",
            "Source compat-correct",
            "Source accuracy",
            "K",
            "Each subset n",
            "Union n",
            "Neutral/subset",
            "Compat-invalid in incorrect",
        ],
    )
    add("""Source accuracy is calculated before eligibility/downsampling across all 28 labels. The
geometry
cohorts are deliberately 50:50 correct/incorrect **within every retained emotion**;
their proportions
cannot estimate population accuracy. `all` means the post-downsampling union and
overlaps both subsets.
The union excludes discarded examples. All retained emotions have positive paired
support and NRC
coverage; neutral is well supported in every model. Gemma's large invalid-output
contribution
(35.98% of its incorrect subset) is a material difference in cohort composition.
""")
    rows = []
    for emotion in sorted(TABLES["support"].emotion.unique()):
        values = []
        for model in MODELS:
            s = TABLES["support"].query("model == @model and emotion == @emotion").iloc[0]
            values.append(f"{int(s.correct)}/{int(s.incorrect)} → {int(s.sampled_per_subset)}")
        rows.append([emotion, *values])
    table(
        rows,
        [
            "Gold emotion",
            "Gemma correct/incorrect → sampled/subset",
            "Apertus correct/incorrect → sampled/subset",
            "OLMo correct/incorrect → sampled/subset",
        ],
    )
    add("""A zero after the arrow denotes exclusion under the >100-correct rule, not a zero
centroid.
Gemma retains 16 emotions; Apertus and OLMo retain those same 16 plus annoyance,
approval and optimism
(K=19). All three exclude caring, desire, embarrassment, grief, nervousness, pride,
realization,
relief and remorse. Gemma disapproval only narrowly qualifies (101 correct); OLMo
annoyance has
109 correct, so those centroids have comparatively limited sample support. Centroid
alignment weights
emotions equally despite unequal per-emotion sample counts.

The common cross-model emotion intersection is the 16 Gemma labels. No persisted
common-support
re-centering/projection/re-embedding comparison is available. Consequently, the
cross-model numbers
below describe each model's primary support; they do not rank models on a matched
16-label experiment.
Model-specific correctness and prompt templates also differ. Optional shared-input
comparisons
would require their own saved sample identities and balance audits.
""")
    table(
        [
            [
                MODEL_NAMES[x["model"]],
                x["cohort"],
                f"{x['mean_length']:.2f}",
                f"{x['train']:.2%}",
                f"{x['validation']:.2%}",
                f"{x['test']:.2%}",
                x["strict_rescued"],
            ]
            for x in AUDIT
        ],
        [
            "Model",
            "Cohort",
            "Mean text length (characters)",
            "Train",
            "Validation",
            "Test",
            "Compat-correct but strict-incorrect",
        ],
    )
    add("""Incorrect examples are longer on average: by 8.12 characters in Gemma, 6.89 in Apertus
and 7.32
in OLMo. Split proportions are close but not matched, as the design specifies. Gemma's
relaxed parser
rescues 3,488 correct-cohort examples that strict correctness would misassign; Apertus
rescues one,
OLMo none. Replacing the compatibility parser would materially change Gemma's
experiment.
Correctness contrasts also include cohort-specific fitted neutral transforms and
residual text
composition; they do not identify a causal effect of classification correctness.
""")


def recovery_section():
    add("""## 3. Primary recovery: mean held-out R² across all layers

`A_M` is the equal-weight mean of layerwise LOEO R², preserving negative scores. It is
not a peak,
not an R² of layer-averaged predictions, and not a direct amount of nonlinearity. All 54
aggregates
have complete fixed-layer coverage. `Δ` subtracts MDS within the same
model/cohort/target.

### 3.1 Primary 2D valence–arousal
""")
    table(
        aggregate_rows(2),
        ["Model", "Cohort", "MDS A", "Isomap A", "DM A", "Δ Isomap−MDS", "Δ DM−MDS"],
    )
    add("""DM's mean VA advantage is positive in all nine cells, from +0.021 (Apertus correct) to
+0.178
(OLMo incorrect). Isomap's advantage is mixed: −0.066 for OLMo correct and effectively
zero
(−0.001 rounded) for Gemma correct, despite improvements elsewhere. Thus the evidence
does not
support a uniform benefit from nonlinear embeddings as a class.

Within each model, every method recovers more VA in correct than incorrect examples. The
largest
correctness differences are Gemma MDS/Isomap (~+0.17 to +0.19) and Apertus Isomap
(+0.222).
Incorrect-cohort DM scores remain 0.272, 0.357 and 0.469 for Gemma, Apertus and OLMo
respectively;
classification failure is compatible with retained affective organization. These are
descriptive
contrasts, with no test of method superiority or correctness effects.
""")
    figure(
        "va-layerwise-recovery",
        TABLES["scores"].query("dimension == 2"),
        lambda d: panel_curves(d, "cv_r2", "2D VA Procrustes R²", 0, True),
        "VA recovery by normalized depth, with models as rows and "
        "correct/incorrect/all as columns. "
        "Colors identify methods; solid lines are held-out-rating R² and "
        "dashed lines full-data R². "
        "The curves show localized peaks and weaker early/late layers; peak "
        "scores overstate depth-wide recovery. "
        "Full-data and held-out curves can diverge substantially.",
    )
    add("""### 3.2 Supporting 1D valence

Valence and VA predict different targets; their R² values are not interchangeable.
""")
    table(
        aggregate_rows(1),
        ["Model", "Cohort", "MDS A", "Isomap A", "DM A", "Δ Isomap−MDS", "Δ DM−MDS"],
    )
    add("""Valence recovery is numerically stronger than joint VA recovery in these runs. DM
improves mean
valence over MDS in every cell, although the Gemma union gain is only +0.001. Isomap
loses against
MDS for Gemma all (−0.059) and Apertus incorrect (−0.015). Gemma provides a clear
exception to a
uniform correctness advantage: incorrect DM valence A=0.634 exceeds correct A=0.583.
OLMo's MDS
valence means are nearly equal across correct/incorrect (0.631/0.622), while its VA
difference is
much larger (0.456/0.291). Correctness is more consistently associated with joint
recovery than
with a single valence coordinate.
""")
    figure(
        "valence-layerwise-recovery",
        TABLES["scores"].query("dimension == 1"),
        lambda d: panel_curves(d, "cv_r2", "1D valence Procrustes R²", 0, True),
        "Valence recovery uses the same panels and line conventions as VA. "
        "Gemma's correct peaks concentrate around blocks 23–24, while its "
        "incorrect DM curve remains useful later. "
        "The high peaks coexist with much lower all-layer averages in the "
        "tables; the plots do not establish significant emergence.",
    )
    add("### 3.3 Correctness contrasts and the pooled cohort")
    rows = []
    for model in MODELS:
        for method in METHODS:
            r = [MODEL_NAMES[model], METHOD_NAMES[method]]
            for dim in (1, 2):
                c, w, a = [get_aggregate(model, x, method, dim).A_M for x in COHORTS]
                r.extend([c - w, c - a, w - a])
            rows.append(r)
    table(
        rows,
        ["Model", "Method", "1D C−I", "1D C−all", "1D I−all", "VA C−I", "VA C−all", "VA I−all"],
    )
    add("""The pooled cohort is often worse than both subsets, most clearly for Apertus (all six
method/target combinations) and OLMo VA (all three methods). For example, Apertus DM VA
is 0.468
correct, 0.357 incorrect, but only 0.232 all. This is possible because averaging
opposite subset
patterns, cohort-specific centering/neutral projection, and centroid normalization all
change the
union geometry. The union score need not equal the average subset score. Its larger
sample count
does not guarantee stronger emotion-centroid alignment, and it is not an independent
replication.
""")


def selection_section():
    add("""## 4. Layer selection, axis recovery and emotion-level errors

### 4.1 Peaks versus nested selection

Cells below are **descriptive peak layer : peak R² / nested-selected R²**. The first
score selects
the maximum from layerwise CV scores and is optimistic as a selected-performance
estimate. The
second comes from separate outer folds that select layer and setting using inner ratings
only.
It estimates that adaptive selection procedure; it is not a score at a single globally
selected
layer and does not replace the primary all-layer mean.
""")
    scores = TABLES["scores"]
    selected = TABLES["selected_layer_scores"]
    for dim, label in [(1, "1D valence"), (2, "2D VA")]:
        add(f"**{label}.**")
        rows = []
        for model in MODELS:
            for cohort in COHORTS:
                row = [MODEL_NAMES[model], cohort]
                for method in METHODS:
                    g = scores[
                        (scores.model == model)
                        & (scores.cohort == cohort)
                        & (scores.method == method)
                        & (scores.dimension == dim)
                    ]
                    peak = g.sort_values(["cv_r2", "layer"], ascending=[False, True]).iloc[0]
                    n = selected[
                        (selected.model == model)
                        & (selected.cohort == cohort)
                        & (selected.method == method)
                        & (selected.dimension == dim)
                    ].iloc[0]
                    row.append(f"{int(peak.layer)} : {peak.cv_r2:.3f} / {n.cv_r2:.3f}")
                rows.append(row)
        table(rows, ["Model", "Cohort", "MDS", "Isomap", "DM"])
    add("""The distinction changes the interpretation. OLMo correct VA DM falls from a descriptive
peak
0.676 to nested-selected 0.439, below nested MDS 0.626. Gemma incorrect VA similarly
favors MDS
after nested selection (0.635 versus DM 0.494). Apertus all VA favors Isomap (nested
0.680), not
DM (0.565), despite DM having the higher all-layer mean. In incorrect valence, selection
is also
unstable: Apertus DM peak 0.830 becomes 0.266. These examples qualify claims based on
attractive
peak maps or a single summary. All nested scores remain descriptive with respect to
permutation
inference; no selection-adjusted p-values were saved.

The observed depth profiles differ: Gemma's strongest correct-cohort recovery clusters
near blocks
21–25 (normalized depth about 0.52–0.62); Apertus correct VA peaks span blocks 13–20
(0.44–0.66); OLMo correct VA peaks differ markedly by method, at blocks 9, 13 and 24.
There is no shared universal peak depth. The union curves show abrupt drops around Gemma
block 25,
Apertus block 14 and OLMo block 14; neutral-transform changes and candidate-setting
changes are
possible contributors, so these transitions cannot be assigned to a model mechanism from
scores alone.
""")
    rows = []
    for model in MODELS:
        emergence = json.loads((RUNS[model] / "metrics/emergence.json").read_text())
        for cohort in COHORTS:
            row = [MODEL_NAMES[model], cohort]
            for method in METHODS:
                cells = []
                for dim in (1, 2):
                    r = next(
                        x
                        for x in emergence
                        if x["cohort"] == cohort and x["method"] == method and x["dimension"] == dim
                    )
                    cells.append(f"{len(r['abrupt_transition_layers'])}/{len(r['decline_layers'])}")
                row.append("; ".join(cells))
            rows.append(row)
    table(
        rows,
        ["Model", "Cohort", "MDS transitions/declines: 1D; VA", "Isomap: 1D; VA", "DM: 1D; VA"],
    )
    add("""These are saved counts of adjacent-layer changes with |ΔR²|≥0.1, with declines defined
by
ΔR²≤−0.1; every decline is included among transitions. They quantify profile
irregularity under
the configured descriptive rule, not significant emergence. The exact transition layers
remain
in `metrics/emergence.json`. Stable significant intervals require at least three
corrected
significant layers and are unavailable because permutations were skipped.

### 4.2 Joint VA scores can conceal poor arousal recovery

The table reports pooled out-of-fold Pearson correlations **at each method's descriptive
VA-peak
layer**; it does not average singleton-fold correlations or treat these selected
correlations as
unbiased peak estimates.
""")
    peaks = scores.query("dimension == 2").sort_values(["cv_r2", "layer"], ascending=[False, True])
    peaks = peaks.groupby(["model", "cohort", "method"], sort=False).first().reset_index()
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            row = [MODEL_NAMES[model], cohort]
            for method in METHODS:
                x = peaks.query("model == @model and cohort == @cohort and method == @method").iloc[
                    0
                ]
                row.append(f"L{int(x.layer)}: {x.valence_pearson:.3f} / {x.arousal_pearson:.3f}")
            rows.append(row)
    table(rows, ["Model", "Cohort", "MDS rV / rA", "Isomap rV / rA", "DM rV / rA"])
    add("""At DM's VA peak, Gemma all has rV=0.947 but rA=0.029; Apertus correct has rV=0.949 but
rA=−0.286; OLMo all has rV=0.954 but rA=−0.703. OLMo correct DM arousal is essentially
unrelated
(rA=0.005). Isomap generally gives positive arousal correlations at its VA peaks,
reaching 0.506
for Gemma correct, 0.600 for Apertus incorrect and 0.565 for OLMo correct. This axis
distinction
is essential: a DM advantage in joint R² does not establish better arousal recovery.

The target scales help explain the discrepancy. Across retained labels, valence variance
is 0.612
versus arousal 0.165 for Gemma, and 0.600 versus 0.148 for Apertus/OLMo. Thus valence
contributes
about 79–80% of the joint target sum of squares under the same LOEO denominator
weighting. Joint
R² can improve by fitting valence while arousal remains weak. The analysis retains the
specified
unrescaled VA target and reports the two axes separately.
""")
    axis_data = scores.query("dimension == 2").melt(
        id_vars=[c for c in scores.columns if c not in ["valence_pearson", "arousal_pearson"]],
        value_vars=["valence_pearson", "arousal_pearson"],
        var_name="axis_metric",
        value_name="axis_r",
    )

    def axes_plot(d):
        fig, axs = plt.subplots(
            3, 3, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True
        )
        for i, model in enumerate(MODELS):
            for j, cohort in enumerate(COHORTS):
                ax = axs[i, j]
                for method in METHODS:
                    for metric, style in [("valence_pearson", "-"), ("arousal_pearson", "--")]:
                        g = d.query(
                            "model == @model and cohort == @cohort and method == @method and "
                            "axis_metric == @metric"
                        ).sort_values("depth")
                        ax.plot(
                            g.depth,
                            g.axis_r,
                            style,
                            color=COLORS[method],
                            lw=1.3,
                            label=f"{METHOD_NAMES[method]} {'V' if style == '-' else 'A'}",
                        )
                ax.axhline(0, color=".5", lw=0.6)
                ax.set_ylim(-1.05, 1.05)
                ax.set_title(f"{MODEL_NAMES[model]} · {cohort}", fontsize=10)
                if i == 2:
                    ax.set_xlabel("Normalized block depth")
                if j == 0:
                    ax.set_ylabel("Pooled OOF Pearson r")
        axs[0, 0].legend(fontsize=6, ncol=2)
        return fig

    figure(
        "va-axis-correlations",
        axis_data,
        axes_plot,
        "Held-out valence (solid) and arousal (dashed) correlations by depth, colored by method. "
        "Positive valence recovery is broad; arousal is weaker and sometimes negative. "
        "The peak-layer correlation table quantifies the discrepancy; neither"
        " axis correlation is a separate significance test.",
    )
    add("### 4.3 Which emotions benefit from the nonlinear methods?")
    e = TABLES["paired_errors"]
    e = e[(e.dimension == 2) & (e.cohort_left == e.cohort_right) & (e.method_right == "mds")]
    summary = (
        e.groupby(["model", "cohort_left", "method_left", "emotion"])
        .delta_error.mean()
        .reset_index()
    )
    rows = []
    for (model, cohort, method), g in summary.groupby(
        ["model", "cohort_left", "method_left"], sort=False
    ):
        order = g.sort_values("delta_error")
        best, worst = order.iloc[0], order.iloc[-1]
        rows.append(
            [
                MODEL_NAMES[model],
                cohort,
                METHOD_NAMES[method],
                f"{best.emotion}: {best.delta_error:+.3f}",
                f"{worst.emotion}: {worst.delta_error:+.3f}",
                int((g.delta_error < 0).sum()),
                len(g),
            ]
        )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "Method vs MDS",
            "Largest error reduction",
            "Largest error increase",
            "Emotions improved",
            "K",
        ],
    )
    add("""These are mean layerwise differences in squared held-out VA error per emotion; negative
values
favor the nonlinear method. The table exposes heterogeneous benefits that a positive ΔA
can hide.
Extrema are exploratory and selected for description; no emotion-level significance
follows.
Paired errors use the common NRC-covered emotion order within each model and the same
folds.
Sadness often has the largest error reduction, especially for DM (OLMo correct −0.830),
while
neutral can become worse (Apertus correct DM +0.787). DM improves mean VA error for
11–16 emotions
per condition, whereas OLMo correct Isomap improves only 5/19, consistent with its
negative ΔA.
""")


def fidelity_section():
    add("""## 5. Distance fidelity and neighborhood preservation

Tables report equal-layer means at each layer's saved **descriptive alignment-selected
setting**.
This describes the embeddings used in the displayed maps, not an independent
fidelity-based
selection experiment. `S` is scale-optimized Stress-1 (lower is better), `NM` is
nonmetric stress,
`ρ` is Shepard Spearman distance correlation, and `T3/C3` are trustworthiness/continuity
at k=3
(higher is better). Psychological-target alignment uses Pearson; Spearman here compares
distances.
Cosine is the common reference for comparing methods. Native references are cosine for
MDS,
graph geodesics for Isomap, and full-spectrum diffusion distances for DM; native scores
must not
be ranked across those different reference objects.
""")
    chosen = TABLES["chosen_fidelity"]
    for dim, name in [(1, "1D"), (2, "2D")]:
        add(f"**{name} fidelity against common cosine dissimilarities.**")
        f = chosen[(chosen.dimension == dim) & (chosen.reference == "cosine")]
        values = f.groupby(["model", "cohort", "method", "metric"]).value.mean()
        rows = []
        for model in MODELS:
            for cohort in COHORTS:
                for method in METHODS:
                    rows.append(
                        [
                            MODEL_NAMES[model],
                            cohort,
                            METHOD_NAMES[method],
                            *[
                                values[model, cohort, method, m]
                                for m in [
                                    "stress",
                                    "nonmetric_stress",
                                    "shepard_spearman",
                                    "trustworthiness_k3",
                                    "continuity_k3",
                                ]
                            ],
                        ]
                    )
        table(rows, ["Model", "Cohort", "Method", "S", "NM", "ρ", "T3", "C3"])
    add("""Two dimensions improve distance reconstruction and neighborhood preservation over one
dimension,
but substantial stress remains. Better psychological prediction does not imply better
preservation
of the original cosine geometry: classical MDS generally preserves that common reference
more
faithfully while DM can improve VA alignment. This separates the human-rating question
from
the geometric reconstruction question. Native DM stress measures truncation relative to
the full
diffusion spectrum, not agreement with original cosine distances.
""")
    rows = []
    native = chosen[(chosen.reference == "native") & chosen.metric.eq("stress")]
    for model in MODELS:
        for cohort in COHORTS:
            row = [MODEL_NAMES[model], cohort]
            for method in METHODS:
                for dim in (1, 2):
                    g = native.query(
                        "model == @model and cohort == @cohort and method == @method and "
                        "dimension == @dim"
                    )
                    row.append(g.value.mean())
            rows.append(row)
    table(
        rows,
        [
            "Model",
            "Cohort",
            "MDS native S1",
            "MDS S2",
            "Isomap native S1",
            "Isomap S2",
            "DM native S1",
            "DM S2",
        ],
    )
    add("""Selected-setting DM native 2D stress is small (0.035–0.138), although its common-cosine
stress is 0.357–0.400. Diffusion time can strongly downweight higher modes, making the
truncated
output reproduce its own full-spectrum reference well while distorting the original
geometry.
The matched t=1 diagnostics below provide a more comparable construction across cohorts.
""")
    # Neighborhood grid sensitivity is summarized without changing the primary k=3 diagnostic.
    rows = []
    common = chosen.query("reference == 'cosine'")
    for method in METHODS:
        for dim in (1, 2):
            row = [METHOD_NAMES[method], dim]
            for prefix in ["trustworthiness", "continuity"]:
                for k in [1, 3, 5]:
                    values = (
                        common[
                            (common.method == method)
                            & (common.dimension == dim)
                            & (common.metric == f"{prefix}_k{k}")
                        ]
                        .groupby(["model", "cohort"])
                        .value.mean()
                    )
                    row.append(f"{values.min():.3f}–{values.max():.3f}")
            rows.append(row)
    table(
        rows,
        ["Method", "Dim", "T1 range", "T3 range", "T5 range", "C1 range", "C3 range", "C5 range"],
    )
    add("""Ranges span the nine separate model/cohort means, not confidence intervals and not a
pooled
model ranking. The complete k=1/3/5 diagnostics and scale factors are retained in
`metrics.csv`.
Fixed-setting trustworthiness and graph-ratio aggregates are also saved: unsupported
sparse
settings retain incomplete-coverage records instead of being averaged over successful
layers only.
""")
    rows = []
    for model in MODELS:
        g = TABLES["fidelity_aggregates"].query("model == @model")
        rows.append(
            [MODEL_NAMES[model], int(g.status.eq("ok").sum()), int(g.status.ne("ok").sum()), len(g)]
        )
    table(rows, ["Model", "Complete fixed-setting aggregates", "Unsupported", "Total"])
    add("""These 1,056 rows per model cover fixed-setting trustworthiness and graph-ratio
summaries,
including all neighborhood/reference variants. Missing sparse-graph layers invalidate
170
Gemma/Apertus aggregates each and 85 OLMo aggregates; they are not silently dropped from
a mean.
""")


def dimensions_section():
    add("""## 6. Spectral dimensions, neutral removal and diffusion diagnostics

The common geometry table summarizes classical-MDS positive spectral mass of squared
cosine
dissimilarities. PR is the participation ratio; d90/d95 are dimensions reaching 90%/95%
positive
mass. These characterize the **emotion-centroid dissimilarity geometry**, not intrinsic
dimension
of the sample-level activation manifold. Means of integer dimensions may be fractional.
""")
    metrics = TABLES["metrics"]
    shared = metrics[(metrics.method == "shared") & metrics.emotion.isna()]
    values = shared.pivot(
        index=["model", "cohort", "layer", "depth"], columns="metric", values="value"
    ).reset_index()
    values["negative_fraction"] = values.negative_mass / (
        values.positive_mass + values.negative_mass
    )
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            g = values.query("model == @model and cohort == @cohort")
            rows.append(
                [
                    MODEL_NAMES[model],
                    cohort,
                    g.participation_ratio.mean(),
                    g.dimensions_90.mean(),
                    g.dimensions_95.mean(),
                    f"{g.retained_energy_1d.mean():.1%}",
                    f"{g.retained_energy_2d.mean():.1%}",
                    f"{g.negative_fraction.mean():.1%}",
                    f"{g.neutral_components_removed.mean():.1f} "
                    f"({int(g.neutral_components_removed.min())}–"
                    f"{int(g.neutral_components_removed.max())})",
                ]
            )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "PR",
            "d90",
            "d95",
            "1D mass",
            "2D mass",
            "Negative / absolute mass",
            "Neutral PCs removed: mean (range)",
        ],
    )
    add("""Across the nine cells, mean PR is 4.55–6.50, d90 is 5.93–7.94 and d95 is 7.24–9.38.
Only 47.0–60.0% of positive mass is retained in two dimensions. Therefore, 2D VA is a
useful target
projection, not an adequate description of all centroid geometry. Correctness does not
consistently
reduce dimensions: Gemma incorrect PR (4.55) is lower than correct (5.70), whereas
Apertus correct
PR (5.41) is slightly below incorrect (5.53). Nonzero negative eigenvalue mass shows
that the
cosine-derived dissimilarities are not exactly realizable by a Euclidean MDS embedding.

The number of removed neutral PCs changes substantially with layer and cohort,
especially in
Gemma (up to 41 correct-cohort components). Since transforms are independently fitted,
cohort
differences jointly reflect sample composition and the estimated neutral subspace.
Spectral mass
is not sample-activation variance, and no auxiliary PCA or inferred dimensionality elbow
is added.
""")
    ctx = TABLES["scores"].drop_duplicates(["model", "cohort", "layer"])[
        [
            "model",
            "cohort",
            "layer",
            "run_id",
            "source_split",
            "representation_level",
            "preprocessing",
            "distance",
            "emotion_set_id",
            "emotion_order",
            "K",
            "draw_id",
            "input_condition",
            "fold_policy",
            "inference_label",
            "n_samples",
        ]
    ]
    dimdata = values.merge(ctx, on=["model", "cohort", "layer"], validate="one_to_one")
    dimdata["method"] = "shared"
    dimdata["setting"] = "original"
    dimdata["dimension"] = 0
    dimdata["target_set"] = "none"
    dimdata["status"] = "ok"

    def dimension_plot(d):
        fig, axes = plt.subplots(
            3, 3, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True
        )
        for i, model in enumerate(MODELS):
            for j, cohort in enumerate(COHORTS):
                ax = axes[i, j]
                g = d.query("model == @model and cohort == @cohort").sort_values("depth")
                for name, label in [
                    ("participation_ratio", "PR"),
                    ("dimensions_90", "d90"),
                    ("dimensions_95", "d95"),
                ]:
                    ax.plot(g.depth, g[name], label=label)
                ax.axhline(2, color=".5", ls="--", lw=0.7)
                ax.set_title(f"{MODEL_NAMES[model]} · {cohort}", fontsize=10)
                if i == 2:
                    ax.set_xlabel("Normalized block depth")
                if j == 0:
                    ax.set_ylabel("Positive-spectrum dimensions")
        axes[0, 0].legend(fontsize=8)
        return fig

    figure(
        "centroid-spectral-dimensions",
        dimdata,
        dimension_plot,
        "Participation ratio, d90 and d95 by depth; the dashed reference is two dimensions. "
        "All nine conditions retain structure beyond two dimensions, "
        "corroborating the spectral-mass table. "
        "The curves are centroid diagnostics and do not estimate a population manifold dimension.",
    )
    add("""### 6.1 Diffusion geometry at matched settings

To compare the same diffusion construction across cohorts, this table fixes k=10, local
scale
rank=5, t=1, α=1 (`k10-s5-t1-a1`) at every layer. This transparent diagnostic slice was
chosen for
comparison, without ranking settings by NRC recovery; it is not an additional primary
analysis.
PR/d90/d95 and retained energy use the nonconstant diffusion contributions `|λ|^(2t)`.
DM Stress-1 uses **all** nonconstant modes for the reference distance.
""")
    fixed = metrics[(metrics.method == "diffusion_maps") & (metrics.setting == "k10-s5-t1-a1")]
    v = fixed.groupby(["model", "cohort", "dimension", "metric"]).value.mean()
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            rows.append(
                [
                    MODEL_NAMES[model],
                    cohort,
                    v[model, cohort, 0, "diffusion_direct_spearman"],
                    v[model, cohort, 0, "participation_ratio"],
                    v[model, cohort, 0, "dimensions_90"],
                    v[model, cohort, 0, "dimensions_95"],
                    f"{v[model, cohort, 0, 'retained_energy_1d']:.1%}",
                    f"{v[model, cohort, 0, 'retained_energy_2d']:.1%}",
                    fixed.query(
                        "model == @model and cohort == @cohort and dimension == 1 and "
                        "reference == 'native' and metric == 'stress'"
                    ).value.mean(),
                    fixed.query(
                        "model == @model and cohort == @cohort and dimension == 2 and "
                        "reference == 'native' and metric == 'stress'"
                    ).value.mean(),
                ]
            )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "Diffusion–cosine ρ",
            "PR",
            "d90",
            "d95",
            "Energy 1D",
            "Energy 2D",
            "Native S1",
            "Native S2",
        ],
    )
    add("""Diffusion and direct cosine distances remain strongly associated at these settings
(ρ=0.896–0.928).
Two coordinates retain only 58.8–72.6% of diffusion energy; d95 is 6.19–7.75, again
exceeding two.
The smallest diffusion PR is Gemma incorrect (3.25), while pooled Apertus/OLMo are
around 4.5.
Diffusion modifies connectivity while retaining much of the original distance ordering;
improved
VA recovery need not imply an entirely different manifold. Energy dimensions depend on
time,
graph and density normalization; they are not commensurate with MDS positive-mass
dimensions.
""")

    def diffusion_plot(d):
        fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, constrained_layout=True)
        settings = [
            ("diffusion_direct_spearman", 0, "Distance Spearman ρ"),
            ("retained_energy_2d", 0, "2D energy fraction"),
            ("stress", 2, "2D native Stress-1"),
        ]
        for i, model in enumerate(MODELS):
            for j, (metric, dim, ylabel) in enumerate(settings):
                ax = axes[i, j]
                for cohort in COHORTS:
                    g = d[
                        (d.model == model)
                        & (d.cohort == cohort)
                        & (d.metric == metric)
                        & (d.dimension == dim)
                    ]
                    if metric == "stress":
                        g = g[g.reference == "native"]
                    g = g.sort_values("depth")
                    ax.plot(g.depth, g.value, label=cohort)
                ax.set_title(f"{MODEL_NAMES[model]} · {ylabel}", fontsize=9)
                if i == 2:
                    ax.set_xlabel("Normalized block depth")
                if j < 2:
                    ax.set_ylim(0, 1.02)
        axes[0, 0].legend(fontsize=8)
        return fig

    figure(
        "matched-diffusion-diagnostics",
        fixed,
        diffusion_plot,
        "At fixed diffusion settings, columns show diffusion–cosine distance "
        "correlation, 2D retained energy, "
        "and native 2D stress; lines identify cohorts. High distance "
        "correspondence coexists with incomplete "
        "2D energy retention and nonzero stress. These curves compare matched"
        " constructions, not rating-selected winners.",
    )


def geometry_section():
    add("## 7. Nonlinearity evidence: graphs, topology, scatter and spectra")
    metrics = TABLES["metrics"]
    add("""### 7.1 Graph dependence and nonmetric cosine dissimilarities

Geodesic/direct ratios are summarized once per graph, not once per embedding method. The
fixed
k=10 graph is connected at every layer in every cohort. A ratio below one is allowed
because
cosine dissimilarity violates the triangle inequality; ratios are not metric path
stretch.
""")
    rows = []
    graph = TABLES["graph_ratios"]
    for model in MODELS:
        for cohort in COHORTS:
            g = metrics.query("model == @model and cohort == @cohort and method == 'shared'")
            f = metrics.query(
                "model == @model and cohort == @cohort and method == 'isomap' and setting == 'k10'"
            )
            ratios = graph.query("model == @model and cohort == @cohort and setting == 'k10'")
            layer_means = ratios.groupby("layer").ratio.mean()
            rows.append(
                [
                    MODEL_NAMES[model],
                    cohort,
                    f"{g[g.metric == 'triangle_violation_fraction'].value.mean():.3%}",
                    g[g.metric == "maximum_triangle_excess"].value.max(),
                    f[f.metric == "graph_density"].value.mean(),
                    f[f.metric == "algebraic_connectivity"].value.min(),
                    layer_means.mean(),
                    f"{(ratios.ratio < 1 - 1e-10).mean():.1%}",
                ]
            )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "Triangle violations (mean)",
            "Maximum excess",
            "k10 density (mean)",
            "Min algebraic connectivity",
            "Mean geodesic/direct",
            "Pairs ratio<1",
        ],
    )
    add("""Triangle violations are infrequent in aggregate (roughly 0.03–0.11%) but have nonzero
maxima,
so metric interpretations remain invalid. The relatively dense k=10 graphs shorten paths
and
can suppress differences from the MDS baseline, especially at K=16. Sparse alternatives
are
not always supported: disconnected candidates are explicitly rejected rather than
repaired.
""")
    fail = TABLES["failures"]
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            g = fail.query("model == @model and cohort == @cohort")
            rows.append(
                [
                    MODEL_NAMES[model],
                    cohort,
                    int((g.method == "isomap").sum()),
                    int((g.method == "diffusion_maps").sum()),
                    ", ".join(sorted(g.reason.unique())) if len(g) else "none",
                ]
            )
    table(
        rows,
        ["Model", "Cohort", "Unsupported Isomap dim-cells", "Unsupported DM dim-cells", "Reason"],
    )
    add("""These counts include separate 1D/2D records and DM time/scale settings; they are not
counts of
independent failed layers. All failures are two-component disconnected graphs. They do
not
invalidate the 1,908 selected score cells, which have valid alternatives at every layer.
However,
they limit whole-grid comparisons and explain unsupported fixed-setting aggregates. No
failed
candidate is interpreted as a zero alignment score.

### 7.2 Centroid H0/H1 persistence

Persistence uses the same nonmetric cosine clique filtration, bound 2.0, with a
101-point Betti
grid. Zero-lifetime intervals are retained in canonical outputs but excluded when
counting
positive-persistence H1 features below. H0 merger thresholds summarize finite deaths;
its one
essential connected component is not counted as a finite death. H1 lifetimes and counts
are
descriptive and have no sampling-null calibration.
""")
    persistence = TABLES["persistence"].copy()
    persistence["lifetime"] = persistence.death - persistence.birth
    summaries = []
    for (model, cohort, layer), g in persistence.groupby(["model", "cohort", "layer"], sort=False):
        h0 = g[(g.dimension == 0) & np.isfinite(g.death)]
        h1 = g[(g.dimension == 1) & (g.lifetime > 1e-10)]
        summaries.append(
            dict(
                model=model,
                cohort=cohort,
                layer=layer,
                max_h0_death=h0.death.max(),
                positive_h1=len(h1),
                max_h1_lifetime=h1.lifetime.max() if len(h1) else 0,
            )
        )
    t = pd.DataFrame(summaries)
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            g = t.query("model == @model and cohort == @cohort")
            betti = TABLES["betti"].query(
                "model == @model and cohort == @cohort and dimension == 1"
            )
            rows.append(
                [
                    MODEL_NAMES[model],
                    cohort,
                    g.max_h0_death.mean(),
                    g.positive_h1.mean(),
                    g.max_h1_lifetime.mean(),
                    g.max_h1_lifetime.max(),
                    int(betti.betti.max()),
                ]
            )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "H0 final merge (mean)",
            "Positive H1/layer (mean)",
            "Longest H1/layer (mean)",
            "Longest H1 overall",
            "Max grid Betti-1",
        ],
    )
    add("""Finite H1 features identify scale-dependent cycles in the centroid dissimilarity graph.
Their
presence does not identify a continuous affective manifold or prove curvature. Lifetimes
are
in cosine-filtration units, depend on K and cohort-specific preprocessing, and lack
resampling
stability estimates. These results complement alignment and spectral diagnostics; the
design
defines no composite nonlinearity score.
Correct cohorts do not uniformly have longer-lived cycles: mean longest H1 lifetime is
0.161 correct versus 0.150 incorrect for Gemma, 0.180 versus 0.187 for Apertus, and
0.182 versus
0.199 for OLMo. Apertus's pooled cohort has the most positive H1 features per layer
(8.59) and
largest grid Betti-1 (9), despite weak union VA recovery. Topology and rating recovery
are distinct.
""")
    ctx = TABLES["scores"].drop_duplicates(["model", "cohort", "layer"])
    t = t.merge(
        ctx[
            [
                "model",
                "cohort",
                "layer",
                "depth",
                "run_id",
                "emotion_set_id",
                "emotion_order",
                "K",
                "n_samples",
                "preprocessing",
                "distance",
                "draw_id",
                "input_condition",
                "fold_policy",
                "source_split",
                "representation_level",
                "inference_label",
            ]
        ],
        on=["model", "cohort", "layer"],
    )
    t["method"] = "shared"
    t["filtration_bound"] = 2.0
    t["status"] = "ok"

    def topology_curves(d):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True, constrained_layout=True)
        for ax, model in zip(axes, MODELS):
            for cohort in COHORTS:
                g = d.query("model == @model and cohort == @cohort").sort_values("depth")
                ax.plot(g.depth, g.max_h1_lifetime, label=cohort)
            ax.set_title(MODEL_NAMES[model])
            ax.set_xlabel("Normalized block depth")
        axes[0].set_ylabel("Longest positive H1 lifetime")
        axes[0].legend(fontsize=8)
        return fig

    figure(
        "topology-layerwise",
        t,
        topology_curves,
        "Longest positive H1 lifetime at each layer, with models as panels and cohorts as lines. "
        "Layer-to-layer changes show that cycles are scale- and "
        "representation-dependent; the table reports "
        "their counts and lifetimes. No confidence interval or topological "
        "significance is implied.",
    )
    add("### 7.3 RDM contrasts and within-emotion scatter")
    rdm = TABLES["rdm_contrasts"]
    rows = []
    for model in MODELS:
        for left, right in [("correct", "incorrect"), ("correct", "all"), ("incorrect", "all")]:
            g = rdm.query("model == @model and cohort_left == @left and cohort_right == @right")
            correlations = g.groupby("layer").apply(
                lambda x: x.distance_left.corr(x.distance_right), include_groups=False
            )
            mae = g.groupby("layer").delta_distance.apply(lambda x: x.abs().mean())
            rows.append(
                [
                    MODEL_NAMES[model],
                    f"{left} / {right}",
                    correlations.mean(),
                    f"{correlations.min():.3f}–{correlations.max():.3f}",
                    mae.mean(),
                ]
            )
    table(
        rows,
        [
            "Model",
            "Cohort pair",
            "RDM Pearson r (mean)",
            "Layerwise r range",
            "Pairwise distance MAE (mean)",
        ],
    )
    add("""RDM correlations compare the same emotion pairs within each model and are descriptive.
The pooled
RDM is generally closer to each subset than the subsets are to each other, consistent
with a mixed
centroid geometry. Even correlated RDMs can give different VA recovery because alignment
depends
on which emotion relationships shift, not only on global distance agreement.
For correct/incorrect RDMs, mean correlations are 0.728 (Gemma), 0.535 (Apertus), and
0.592
(OLMo), with mean absolute cosine-distance changes of 0.154–0.179. These comparisons
quantify
cohort-associated geometry changes on matched within-model emotion support.
""")
    scatter = metrics[metrics.metric == "within_emotion_scatter"]
    rows = []
    for model in MODELS:
        pivot = scatter.query("model == @model").pivot(
            index=["layer", "emotion"], columns="cohort", values="value"
        )
        ratio = pivot["incorrect"] / pivot["correct"]
        rows.append(
            [
                MODEL_NAMES[model],
                ratio.median(),
                f"{(ratio > 1).mean():.1%}",
                f"{ratio.min():.3f}–{ratio.max():.3f}",
            ]
        )
    table(
        rows,
        ["Model", "Median incorrect/correct scatter", "Emotion-layer cells ratio>1", "Ratio range"],
    )
    add("""Scatter is mean squared residual norm around the gold-emotion centroid after that
cohort's
projection; ratios pair the same layer and emotion. Ratios avoid comparing raw
activation units
across models, but still include distinct fitted neutral transforms. They do not
establish that
dispersion causes classification errors. Full emotion-level scatter remains in
`metrics.csv`.
Incorrect scatter is only modestly higher at the median (1.8–3.9%), with both increases
and
decreases at individual emotion/layer cells. Large VA contrasts therefore should not be
reduced
to a uniform increase in within-emotion dispersion.

### 7.4 Basic spectral and coordinate checks

Full method/setting spectra, eigengaps, near-tie flags and native coordinates are saved.
Near ties
use the implementation's absolute 1e−8 threshold; conditional subspace checks use three
symmetric
perturbations (seed 0, magnitude 1e−7). Empty stability CSVs mean no tied group was
tested, not
evidence that all leading coordinates are stable. No independent spectral significance
test,
prompt-bootstrap stability estimate, or residual-variance elbow is present. Harmonics
and
horseshoe-like native patterns can be inspected in the original layerwise figures but do
not
independently establish curvature.
""")
    rows = []
    spectra = TABLES["spectra"]
    for model in MODELS:
        for method in METHODS:
            g = spectra.query("model == @model and method == @method")
            conditions = g[["cohort", "layer", "setting"]].drop_duplicates()
            tied = g[g.near_tie][["cohort", "layer", "setting"]].drop_duplicates()
            rows.append([MODEL_NAMES[model], METHOD_NAMES[method], len(conditions), len(tied)])
    table(
        rows,
        ["Model", "Method", "Successful cohort/layer/setting spectra", "Spectra with any near tie"],
    )
    rows = []
    for _, r in (
        spectra[spectra.near_tie]
        .drop_duplicates(["model", "cohort", "layer", "method", "setting"])
        .iterrows()
    ):
        path = (
            RUNS[r.model]
            / "geometry"
            / r.cohort
            / f"layer_{int(r.layer):03d}"
            / r.method
            / r.setting
            / "decomposition/subspace_stability.csv"
        )
        stability = pd.read_csv(path)
        rows.append(
            [
                MODEL_NAMES[r.model],
                r.cohort,
                int(r.layer),
                r.setting,
                stability.modes.iloc[0],
                f"{stability.minimum_principal_cosine.min():.9f}",
                f"{stability.projector_distance.max():.2e}",
                link(path, "saved checks"),
            ]
        )
    table(
        rows,
        [
            "Model",
            "Cohort",
            "Layer",
            "Isomap setting",
            "Tied modes (zero-based)",
            "Min principal cosine",
            "Max projector distance",
            "Source",
        ],
    )
    add("""Only Gemma correct Isomap k=3 has near ties, at layers 22, 24 and 25, in near-zero modes
8–10 rather than the leading two coordinates. All nine saved perturbation checks
succeed;
principal cosines round to one and maximum projector distance is 1.51×10⁻⁶. This
supports numerical
stability of those null subspaces under the specified tiny perturbations, not robustness
to
new samples, alternative graphs, or target-selection changes.
""")


def cka_section():
    add("""## 8. Cross-method CKA at separately selected best layers

Every method chooses its own highest CV-R² layer for the relevant target, with
earliest-layer
tie-breaking. CKA uses centered **native** coordinates on the same emotion order, not
aligned NRC
coordinates. All 18 required 3×3 matrices are available (162 cells, symmetric,
nondegenerate
diagonals=1). The compact table gives the three unique off-diagonal entries in each
matrix;
figures display every matrix, ordered MDS/Isomap/DM on a shared [0,1] color scale.
""")
    cka = TABLES["cka"]
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            for dim in (1, 2):
                g = cka.query("model == @model and cohort == @cohort and dimension == @dim")
                cells = g.set_index(["method_left", "method_right"])
                layers = "/".join(str(int(cells.loc[(m, m), "layer_left"])) for m in METHODS)
                rows.append(
                    [
                        MODEL_NAMES[model],
                        cohort,
                        dim,
                        layers,
                        cells.loc[("mds", "isomap"), "cka"],
                        cells.loc[("mds", "diffusion_maps"), "cka"],
                        cells.loc[("isomap", "diffusion_maps"), "cka"],
                    ]
                )
    table(rows, ["Model", "Cohort", "Dim", "Layers MDS/Iso/DM", "MDS–Iso", "MDS–DM", "Iso–DM"])
    for dim in (1, 2):

        def plot_cka(d):
            fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
            for i, model in enumerate(MODELS):
                for j, cohort in enumerate(COHORTS):
                    ax = axes[i, j]
                    g = d.query("model == @model and cohort == @cohort")
                    matrix = g.pivot(
                        index="method_left", columns="method_right", values="cka"
                    ).reindex(index=METHODS, columns=METHODS)
                    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
                    diagonals = g[g.method_left == g.method_right].set_index("method_left")
                    labels = [
                        f"{METHOD_NAMES[m]} L{int(diagonals.loc[m, 'layer_left'])}" for m in METHODS
                    ]
                    ax.set_xticks(range(3), labels, fontsize=8)
                    ax.set_yticks(range(3), labels, fontsize=8)
                    ax.set_title(
                        f"{MODEL_NAMES[model]} · {cohort} · K={int(g.K.iloc[0])}", fontsize=9
                    )
                    for a in range(3):
                        for b in range(3):
                            ax.text(
                                b,
                                a,
                                f"{matrix.iloc[a, b]:.3f}",
                                ha="center",
                                va="center",
                                fontsize=9,
                                color="black" if matrix.iloc[a, b] > 0.65 else "white",
                            )
            fig.colorbar(im, ax=axes, shrink=0.75, label="Linear CKA")
            return fig

        figure(
            f"best-layer-cka-{dim}d",
            cka[cka.dimension == dim],
            plot_cka,
            f"Full {dim}D CKA matrices for every model and cohort, "
            "with selected layers and K annotated. "
            "High similarity indicates shared emotion geometry across methods "
            "despite separate layer selection. "
            "The table supplies precise pairwise values; CKA is descriptive and "
            "does not measure NRC recovery or validate a nonlinear manifold.",
        )
    add("""One-dimensional CKA ranges from 0.822 to 0.996; it equals squared Pearson coordinate
correlation
and ignores sign. Two-dimensional agreement ranges from 0.793 to 0.989. The lowest 2D
comparison
is OLMo incorrect MDS–DM (0.793), while Isomap–DM there remains 0.952. Gemma's union
MDS–Isomap
agreement is especially high (0.989 in 2D), consistent with their close VA peaks.
Overall, nonlinear
methods preserve much of the baseline organization; CKA does not establish that their
differences
encode arousal or that a native axis is psychological valence.
""")


def maps_section():
    add("""## 9. Native maps and reproducible visual inspection

The following compact views show the descriptive VA-peak native coordinates for each
method and
cohort. They are chosen for inspection, not performance estimation. Each panel retains
neutral
and draws horizontal/vertical references through its **embedded neutral point**, with
native
units and independent axis limits. There is no NRC overlay and no implied cross-model
coordinate
alignment. Original full-layer maps remain linked in the provenance index.
""")
    coords = TABLES["coordinates"]
    score = (
        TABLES["scores"]
        .query("dimension == 2")
        .sort_values(["cv_r2", "layer"], ascending=[False, True])
    )
    peaks = score.groupby(["model", "cohort", "method"], sort=False).first().reset_index()
    keys = ["model", "cohort", "method", "layer", "dimension", "setting"]
    selected = coords.query("dimension == 2 and coordinate_type == 'native'").merge(
        peaks[keys + ["cv_r2"]], on=keys, validate="many_to_one"
    )
    for model in MODELS:

        def plot_maps(d):
            fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
            for i, cohort in enumerate(COHORTS):
                for j, method in enumerate(METHODS):
                    ax = axes[i, j]
                    g = d.query("cohort == @cohort and method == @method")
                    ax.scatter(g.x, g.y, s=20, color=COLORS[method])
                    neutral = g[g.emotion == "neutral"].iloc[0]
                    ax.axvline(neutral.x, color=".65", ls=":", lw=0.8)
                    ax.axhline(neutral.y, color=".65", ls=":", lw=0.8)
                    ax.margins(0.25)
                    x = g.iloc[0]
                    ax.set_title(
                        f"{cohort} · {METHOD_NAMES[method]} L{int(x.layer)}\n"
                        f"{x.setting} · CV R²={x.cv_r2:.3f}",
                        fontsize=9,
                    )
                    ax.set_xlabel("Native coordinate 1")
                    ax.set_ylabel("Native coordinate 2")
            fig.canvas.draw()
            fig.set_layout_engine(None)
            for i, cohort in enumerate(COHORTS):
                for j, method in enumerate(METHODS):
                    label_points(axes[i, j], d.query("cohort == @cohort and method == @method"))
            return fig

        figure(
            f"{model}-va-peak-native-maps",
            selected[selected.model == model],
            plot_maps,
            f"{MODEL_NAMES[model]} native VA-peak maps, "
            "with cohorts as rows and methods as columns. "
            "Neutral reference lines make its relative position visible. Emotion "
            "separations describe these "
            "selected maps; the fidelity tables and axis correlations qualify any"
            " apparent VA organization.",
        )
    add("""Across these maps, positive and negative emotion groups are often separated along a
leading
coordinate, consistent with the strong valence correlations. Neutral occupies different
positions
by cohort, and the second-coordinate ordering varies across methods. In OLMo's union,
the DM map
still separates positive/negative groups while its arousal prediction is anticorrelated
(rA=−0.703);
this illustrates why a visually organized map is insufficient evidence for VA recovery.
Small
DM coordinate scales reflect diffusion time, not less psychological variability.
""")


def conclusion_section():
    add("""## 10. Robustness, limits and direct answer to EXP3

**Supported descriptive conclusions.** Correct examples have higher depth-averaged VA
recovery in
all nine model–method comparisons, while incorrect examples still encode recoverable
affective
relationships. DM's VA mean exceeds MDS in every cohort; Isomap's advantage is
conditional on model
and cohort. Recovery is concentrated in particular depth regions rather than sustained
uniformly.
The pooled union often weakens recovery despite its larger sample size. Methods'
selected native
geometries remain strongly related by CKA.

**Qualified interpretation of nonlinearity.** Nonlinear-method alignment gains are
suggestive,
but joint gains are often driven by valence, not arousal, and several gains reverse
under nested
layer selection. Native fidelity, diffusion-energy retention, eigenspectra and H1 cycles
describe
different properties and do not constitute independent proof of a nonlinear VA manifold.
The
positive spectrum consistently needs more than two dimensions for high mass retention.

**What cannot be claimed.** No above-chance significance, method-superiority test,
causal correctness
effect, significant first layer, stable significant emergence interval,
architecture/instruction-tuning
effect, or matched-support cross-model ranking follows from these runs. The design
specifies
separate maximum-statistic families for aggregate A and layerwise R², within
model/target across
methods/cohorts (and layers for the latter); neither family was completed. Correction
would not
extend across the entire model panel. An alignment-against-chance p-value, even if
completed,
would not test descriptive method/cohort differences.

**Material limitations.** There is one balanced seed, no prompt-sampling intervals, and
no saved
replication across cohort draws. K is only 16 or 19; equal emotion weights coexist with
unequal
centroid precision. Correct/incorrect transforms differ, texts are not length matched,
and Gemma's
incorrect subset contains many parser-invalid responses. The representations and neutral
fits are
transductive, targets are word-level NRC norms, and valence has substantially greater
target
variance than arousal. Sparse graph failures restrict the candidate grid. Cosine
nonmetricity,
negative MDS mass, selection optimism and truncated-coordinate stress limit native-map
inference.

**Direct answer.** The saved outputs support correctness-associated affective centroid
geometry and
conditional benefits from nonlinear embeddings. They do not yet establish a nonlinear
two-axis VA
manifold. The most defensible pattern is strong valence organization, less reliable
arousal,
and additional centroid structure beyond two dimensions. Completing the specified
permutation
families and a separately persisted common-support comparison would address the
principal missing
inferential and cross-model evidence; this report does not rerun those experiments.
""")


def provenance_section():
    add("""## 11. Source index and reproducibility

This report reads canonical saved outputs and creates only report-specific plots and
their tidy
source tables. Canonical results are unchanged. The Markdown lives under `analyses/` as
requested;
Embedded report figures live in `plots/`; plot-ready CSVs remain under `outputs/`.
""")
    rows = []
    for model in MODELS:
        run = RUNS[model]
        rows.append(
            [
                MODEL_NAMES[model],
                link(run / "manifest.json", "manifest"),
                link(run / "checkpoints/run.json", "checkpoint hashes"),
                link(run / "validation/fold_schedule.json", "fold schedule"),
                link(run / "inference/1d/status.json", "1D status")
                + "; "
                + link(run / "inference/2d/status.json", "VA status"),
                link(run / "plots", "all original figures"),
            ]
        )
    table(
        rows,
        ["Model", "Run provenance", "Integrity", "Folds", "Inference completion", "Original plots"],
    )
    add("""The following index converts each compact result family into the corresponding tables
above.
Counts refer to canonical plot-ready tables, including failed candidate records where
applicable.
Raw pairwise-distance tables are reproduction backing for fidelity, not independent
inferential
observations; all-layer coordinate and Betti tables are similarly summarized rather than
pasted
as hundreds of thousands of Markdown rows.
""")
    sections = {
        "scores": "§§3–4: layer curves, peaks, axis recovery",
        "aggregates": "§3: every A and ΔA",
        "selected_layer_scores": "§4.1: every nested-selected score",
        "cka": "§8: every matrix and unique pair",
        "contrasts": "§3.3: cohort/method differences; underlying layerwise deltas",
        "paired_errors": "§4.3: emotion-specific improvements/regressions",
        "metrics": "§§5–7: all 27 metric names, including scatter and graph diagnostics",
        "fidelity_aggregates": "§5: fixed-setting trustworthiness/graph-ratio coverage",
        "persistence": "§7.2: H0/H1 counts/lifetimes",
        "betti": "§7.2: Betti maxima and layerwise cycles",
        "spectra": "§§6,7.4: dimensionality, eigenvalues, near ties",
        "coordinates": "§9: selected native maps; full-layer archive linked above",
        "graph_ratios": "§7.1: graph-level ratio diagnostics",
        "failures": "§7.1: every failure type and cohort",
        "rdm_contrasts": "§7.3: matched-pair RDM agreement and differences",
        "held_out_predictions": "§4: OOF axis/error evidence; predictions retained",
        "composition": "§2: invalid contributions and source composition",
        "support": "§2: all 28 emotion counts and eligibility",
        "counts": "§2: pre/post sampling support and split diagnostics",
    }
    rows = []
    for name in sections:
        entries = [x for x in CATALOG if x[1] == name]
        rows.append(
            [
                name,
                *[f"{next(x[2] for x in entries if x[0] == model):,}" for model in MODELS],
                sections[name],
                " / ".join(
                    link(next(x[3] for x in entries if x[0] == model), MODEL_NAMES[model])
                    for model in MODELS
                ),
            ]
        )
    table(
        rows,
        [
            "Canonical CSV",
            "Gemma rows",
            "Apertus rows",
            "OLMo rows",
            "Report coverage",
            "CSV links",
        ],
    )
    add("""Additional reproduction backing: each run's `metrics/pairwise_distances.csv` and
`plot_data/pairwise_distances.csv` store the distinct emotion-pair reference/embedded
distances;
`validation/` stores OOF predictions, transforms, settings and nested selected layers;
`preprocessing/` stores neutral transforms and centroids; `geometry/` stores
decomposition
metadata, eigenpairs and conditional subspace-stability CSVs. `metrics/emergence.json`
stores
descriptive peaks, configured ±0.1 transitions/declines and unavailable significance
summaries;
`metrics/timing.json` records execution timing. These are provenance/diagnostics, not
additional
independent results. No setting or failed layer was filled by rerunning extraction or
embeddings.
""")
    table(
        [[name, link(csv, "data"), link(png, "image")] for name, csv, png in FIGURES],
        ["Report figure", "Self-contained plot-ready CSV", "PNG"],
    )
    add("""To rebuild from the same canonical tables, run from the repository root:

```bash
python analyses/exp3/build_report.py
```

The builder uses the project's shared progress/logging utilities. Logs are centralized
under
`logs/three-models-exp3-analysis-report/`. It validates model/run/emotion identities,
cohort sample
partitions, per-emotion balance, prompt-token positions, unique complete score cells,
aggregate
reconstruction and source-table checkpoint hashes before rendering. It does not validate
the
entire multi-gigabyte activation bundle or rerun scientific computations.
""")


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "plot_data").mkdir(parents=True, exist_ok=True)
    configure_file_logging(
        ROOT / "logs/three-models-exp3-analysis-report",
        "build_report",
        models=",".join(MODELS),
        output=REPORT,
    )
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    load()
    sections = [
        run_section,
        cohorts_section,
        recovery_section,
        selection_section,
        fidelity_section,
        dimensions_section,
        geometry_section,
        cka_section,
        maps_section,
        conclusion_section,
        provenance_section,
    ]
    for section in progress(
        sections, description="Write EXP3 evidence tables and figures", unit="section"
    ):
        section()
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    get_logger().info(
        "report.complete path=%s figures=%d tables=%d",
        REPORT,
        len(FIGURES),
        sum(x.startswith("| ") for x in parts),
    )
    print(f"Created {REPORT.relative_to(ROOT)} with {len(FIGURES)} figures")


if __name__ == "__main__":
    main()
