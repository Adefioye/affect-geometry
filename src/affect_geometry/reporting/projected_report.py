"""Curated narrative for the frozen full-space projected run and its default audit."""
# ruff: noqa: E501 -- Markdown prose and tables retain their reviewable paragraph layout.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_projected_report(run: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    root = run / "report"
    base = "."
    metadata = "./metadata"
    runlink = "./metadata/run"
    registry = {row["name"]: row for row in json.loads((root / "figure_index.json").read_text())}
    provenance = json.loads((root / "provenance.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    context = provenance["extended_audit"]["context"]
    audit = provenance["extended_audit"]
    expected = (4999, 999, 200, 1701)
    actual = (
        audit["permutations"],
        audit["eigengap_permutations"],
        audit["repetitions"],
        context["analysis_seed"],
    )
    if actual != expected or context["run_id"] != "lt-neutral-projected-full-space-20261021":
        raise ValueError("This curated narrative requires the documented frozen run/default audit")

    def read(name):
        return pd.read_csv(root / f"tables/{name}.csv")

    def link(name, label=None):
        return f"[{label or name}]({base}/tables/{name}.csv)"

    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def table(name, frame, columns=None, labels=None):
        df = frame[columns].copy() if columns else frame.copy()
        persisted = df.copy()
        for key, value in context.items():
            if key not in persisted:
                persisted[key] = value
        persisted.to_csv(root / f"tables/report_{name}.csv", index=False)
        display = df.rename(columns=labels or {})
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines += [
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in display.itertuples(index=False, name=None)
        ]
        return "\n".join(lines) + f"\n\n*Data: {link('report_' + name)}.*\n"

    def fig(name, caption):
        row = registry[name]
        return (
            f"\n![{row['title']}]({base}/{row['png']})\n\n"
            f"*{caption} [Plot-ready CSV]({base}/{row['csv']}); "
            f"[PDF]({base}/{row['pdf']}).*\n"
        )

    va, vad = read("va_summary"), read("vad_summary")
    axes = read("axis_diagnostics")
    quality = read("quality")
    intrinsic = read("intrinsic")
    source = read("class_balance")
    significance = read("va_recovery")
    vad_significance = read("vad_audit")
    sections = []
    sections.append(f"""# Qwen3-0.6B LT-NeutralProjected: comprehensive geometry analysis

This report analyzes the complete frozen run `{run.name}`: **45,446 single-label comments, 28 labels, 28 decoder blocks, six dissimilarities, and 1,008 embeddings**. Layer numbers are zero based. This directory contains the embedded [figures](figures/), [tables](tables/), [plot-ready data](plot_data/), [metadata](metadata/) and [source snapshot](source/), together with the complete original [plot gallery](exploration.html). Canonical run artifacts are also retained under `outputs/`.

## Main findings

**The most consistent recovered structure is valence, with substantially weaker evidence for arousal and dominance.** Cosine–MDS has the highest mean 2D V–A shape recovery, **R² = 0.4447**, and mean leave-one-emotion-out alignment skill **0.3857**. Cosine–Isomap has the highest selected 2D result, **0.5889 at block 8**, while the best cosine–MDS result is **0.5475 at block 17**. These maxima were selected using the same reference that evaluates them.

The interpretation requires several qualifications:

- **Joint recovery is dominated by valence.** Valence supplies **79.98%** of the V–A target variance. Every distance/method has negative *average* held-out arousal skill in the shared-scale 2D alignment, although specific layers do better.
- **A visible 2D affective organization coexists with higher-dimensional geometry.** MDS participation ratios average **5.53–12.46** across distance choices; retaining 90% of positive spectral mass takes **5–19 dimensions** across individual direct RDMs. Two dimensions are useful for interpretation, with substantial omitted geometry.
- **Isomap trades different notions of quality.** It raises mean V–A fit for five of six distances, but mean 2D trustworthiness improves for only three. The cosine gain in trustworthiness comes with much higher direct-distance stress. Graph choice changes the conclusions quantitatively.
- **Several distances carry largely redundant information.** Energy and RBF-MMD have mean RDM Spearman **0.9955**, compared with **0.4219** for cosine versus crossnobis. Agreement among closely related distances is weaker evidence than agreement across distinct distance families.
- **Unequal sample support affects the apparent strength of the geometry.** Neutral has 16,021 examples; grief has 47. At block 17, balanced cosine–MDS draws have median V–A R² **0.3819**, versus **0.5475** using all samples. The all-split fitted projection also limits the interpretation of split checks.

These are properties of this model, dataset, pooling, transformation and distance definitions. They establish descriptive correspondence with label-level reference ratings, not general emotion recognition accuracy or a causal affective mechanism.

## 1. Scope, provenance and completeness

| Item | Frozen run |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B`; 28 post-block residual-stream layers; 1,024 features |
| Input | Raw GoEmotions comment; no classification prompt or chat template |
| Pooling | Last attended content token; padding/special tokens excluded, with attended-token fallback |
| Coverage | 36,308 train + 4,548 validation + 4,590 test comments; exactly one label |
| Labels | 27 emotion categories plus neutral; every embedding includes all 28 |
| Original activations | Float16; unpinned model revision with resolved commit absent |
| Stored projected activations | Float32, 45,446 × 1,024 at every layer |
| Transformation | Equal-weight non-neutral centroid centering, then orthogonal neutral-subspace removal |
| Neutral rule | Smallest leading neutral PCA subspace reaching at least 50% covariance variance |
| Distance inputs | Full 1,024-dimensional projected vectors; **no post-projection PCA** |
| Main Isomap graph | Union-symmetric 5NN; connectivity expansion allowed |
| Embeddings | Classical MDS and Isomap; 1D, 2D, 3D; spectra and residuals retained |
| Reference | Run-local NRC VAD snapshot, all 28 labels, native [−1, 1] axes |
| Original randomization | Seed 17; 1,000 permutations for each saved reference fit |
| Audit randomization | Seed 1701; 4,999 shared label permutations; 999 eigengap permutations |
| Sampling sensitivity | 200 balanced draws at blocks 0, 7, 14, 17, 21, 27 |

The source configuration, version records and transformation metadata are in [manifest.json]({runlink}/manifest.json), [representation.json]({runlink}/representation.json), and [activation metadata]({runlink}/activations/metadata.json). The frozen reference and its provenance are [NRC VAD CSV]({runlink}/references/goemotions_nrc_vad.csv) and [reference provenance]({runlink}/references/nrc_vad_provenance.json). The configuration's activation dtype describes extraction; the projected store records its actual float32 dtype separately.

The audit inventories **{len(read("source_inventory")):,} source files**, hashes all 168 NPZ RDMs, checks label order and finite symmetric geometry, rebuilds Isomap graphs, and independently recomputes every V–A and VAD score. Maximum score differences are **{validation["va_max_score_error"]:.2e}** for V–A and **{validation["vad_max_score_error"]:.2e}** for VAD; maximum geodesic change is **0**. Sample-derived cosine/Euclidean RDMs at six depths match the saved exact-centroid RDMs within **{validation["centroid_max_error"]:.2e}**. All checks pass. These checks verify the saved downstream computation; they do not establish an unknown source model commit or rerun model inference.

There are **233 repeated exact-text rows**, with 45,446 distinct source IDs. These affect effective independence; globally retaining the first exact text leaves **45,213** comments for the split sensitivity check. See [audit provenance]({metadata}/provenance.json), [validation]({metadata}/validation.json), {link("source_inventory")}, and {link("centroid_reconstruction")}.

## 2. Relation to Choi–Weber

[Choi and Weber, *Latent Structure of Affective Representations in Large Language Models*, §§3–5](https://arxiv.org/html/2604.07382v1) motivates the semantic maps, Procrustes label permutations, eigenspectra/eigengaps, and MDS–Isomap comparisons here.

| Design | Choi–Weber | This run |
| --- | --- | --- |
| Activations | Mean pooling in emotion-classification prompts; correctly classified subset | Last content token of raw comments; all retained labels |
| Main dissimilarity | Pairwise classifier test accuracy | Six centroid/distribution distances after neutral projection |
| Reference alignment | ANEW overlap, 17 classes | NRC V–A/VAD, all 28 labels |
| Isomap choice | Trustworthiness-based graph choice | Fixed main k=5, with an explicit sensitivity sweep |

This report extends those geometry analyses with family-level selection checks, held-out-label alignment, per-axis predictive skill, raw/partial reference RSA, metric-validity checks, categorical neighborhoods, label influence, exact-text split sensitivity, and balanced sampling. The paper's uncertainty calibration, neural-data comparison and steering experiments require observations absent from this run; those findings cannot be tested here. The results therefore provide a related geometry study, not a direct numerical replication. [Source and scope: Choi–Weber](https://arxiv.org/html/2604.07382v1).

## 3. Sample support and what each distance estimates

Every label has equal weight as a node and in reference alignment, while its centroid has very different estimation support. Neutral is **35.25%** of samples and grief **0.10%**. Several rare classes—grief, pride, relief and nervousness—are especially uncertain. Comment length also varies: mean whitespace length ranges from about **10.85** words for excitement to **15.01** for realization; this is not a tokenizer-length or truncation audit.
""")
    sections.append(
        fig(
            "01_sample_support",
            "All retained comments and the actual capped distribution sample; the first horizontal axis is logarithmic.",
        )
    )
    sections.append(
        """| Distance | Quantity used for geometry | Sample use and caution |
| --- | --- | --- |
| Cosine | 1 − cosine of exact projected class centroids | All samples; depends on the centering origin; cosine dissimilarity need not obey triangle inequality |
| Euclidean | Euclidean distance of exact projected centroids | All samples; translation invariant, sensitive to centroid norms |
| Crossnobis | Cross-validated Mahalanobis squared contrast, then √max(raw,0) | All samples; 5 folds; pooled Ledoit–Wolf precision from training-fold residuals |
| RBF-MMD | Unbiased averaged-kernel MMD², then √max(raw,0) | At most 256/class; median bandwidth with multipliers 0.5, 1, 2; negative raw estimates can be clipped |
| Energy | Unbiased two-sample energy statistic, then √max(raw,0) | Same capped sample; clipping can merge distinct nodes |
| Sliced Wasserstein | Mean W1 over 256 seeded unit directions | Same capped sample; distribution discrepancy in projected directions |

The capped sample contains **"""
        + str(int(source.distribution_sample_count.sum()))
        + """** rows. It gives most labels 256 examples, but only 47 for grief, 67 for pride, 103 for relief, 105 for nervousness and 246 for embarrassment. Consequently, centroid and distribution distances differ in both scientific estimand and effective sampling precision. Crossnobis fold separation protects its contrast/precision estimation, while the common neutral transformation was fitted using all splits; it does not create a wholly held-out end-to-end pipeline.

## 4. The neutral transformation is a substantial experimental choice

For class centroid `c_e`, define `g` as the equal-weight average of the **27 non-neutral** centroids. With orthonormal rows `U` fitted to covariance variation among neutral samples, the transformed vector is

`z = (h − g) − ((h − g) Uᵀ) U`.

The neutral-sample mean centers covariance estimation only; it is not an additional translation of the transformed samples. The neutral class remains a node. Euclidean pairwise distances are unaffected by subtracting `g` alone; cosine is affected. Both can change under projection.

Blocks 0 and 1 remove **28** directions each. Blocks **2–23** remove **one** direction each, but the removed variance is not constant: **99.74% at block 2**, **87.64% at block 17**, **50.26% at block 23**. Blocks 24–27 remove **2, 3, 4 and 11** directions. The rule reaches *at least* 50%, and can far exceed it when the first eigenvalue dominates. Thus “50% neutral projection” does not imply removal of exactly half the variance at every layer.
"""
    )
    sections.append(
        fig(
            "02_projection",
            "Number of removed directions and the actual fraction of within-neutral variance removed.",
        )
    )
    sections.append("""Neutral covariance may contain lexical, syntactic, task, frequency and affective variation. These outputs cannot identify which component was removed. Changes near blocks 2 and 24–27 mix model-depth differences with changes in the fitted transformation. The neutral variance fraction is not the fraction of affective centroid signal removed.

## 5. Reference recovery: definitions and statistical scope

Classical MDS eigendecomposes `B = −½ J D² J`. Its axes can rotate or reflect; it minimizes a Gram-matrix criterion, not the distance stress reported below. Isomap applies the same construction to shortest-path distances. The saved cosine dissimilarity itself is squared inside MDS, a choice examined in Section 12.

Similarity Procrustes fits translation, an orthogonal transformation allowing reflection, and **one shared scale**. Joint `R² = 1 − SSE/SST` uses the sum of errors and centered target variation over all labels and axes. Axis R² uses that same fit; per-axis Pearson correlation does not measure calibration and is not axis R².

For **4,999 shared label permutations**, every fit is recomputed. V–A and VAD each have a **336-fit family** (28 layers × 6 distances × 2 methods). The report gives unadjusted randomization p, BH q within each family, and a maximum-statistic p comparing the observed score with the maximum null score across its whole family. Secondary BH correction pools all **672** V–A/VAD tests. These controls address the declared families, including peak-layer selection within them; they do not cover every exploratory choice in this report, the graph sweep, or all possible preprocessing/model variants. The smallest attainable p is **0.0002**, which is not zero.

Leave-one-emotion-out (LOO) fits alignment on 27 target labels, then predicts the omitted target. Skill is `1 − pooled squared prediction error / pooled squared error of the 27-label training-mean baseline`. This is a stricter alignment check than in-sample fit. The geometry still contains all 28 label centroids, so this is **held-out target fitting conditional on a transductive geometry**, not prediction for unseen comments or an entirely unseen emotion class. Layer selection remains exploratory.

## 6. Two-dimensional V–A recovery across depth
""")
    sections.append(
        table(
            "va_ranking",
            va,
            [
                "distance",
                "method",
                "mean_r2",
                "median_r2",
                "peak_r2",
                "peak_layer",
                "final_r2",
                "mean_loo_skill",
                "peak_loo_skill",
            ],
            {
                "mean_r2": "Mean R²",
                "median_r2": "Median R²",
                "peak_r2": "Peak R²",
                "peak_layer": "Peak block",
                "final_r2": "Block 27 R²",
                "mean_loo_skill": "Mean LOO skill",
                "peak_loo_skill": "LOO at peak",
            },
        )
    )
    sections.append(
        fig(
            "03_va_recovery",
            "Each line follows every block; means over layers are descriptive, not independent replications or uncertainty estimates.",
        )
    )
    sections.append("""Cosine–MDS leads in average and median fit. Crossnobis–Isomap is close in mean (**0.4386**) and has a larger median (**0.4675**) than cosine–MDS (**0.4571**). Cosine–Isomap's higher selected maximum (**0.5889**) does not give it the highest across-depth average (**0.4320**). For Euclidean, energy, MMD and sliced Wasserstein, Isomap substantially raises mean reference fit; the changes range from roughly **+0.086 to +0.127**. This says that their graph-derived shapes resemble the reference more closely, not that they preserve input geometry more faithfully.

Early/middle/late phases below are descriptive bins **0–8 / 9–18 / 19–27**, with unequal numbers of blocks. Cosine has a middle-depth advantage and late decline. Crossnobis–Isomap continues to improve on average late, and several direct MDS metrics peak at the final block. There is no distance-independent “emotion layer.”
""")
    sections.append(
        table("depth_phases", va, ["distance", "method", "early_r2", "middle_r2", "late_r2"])
    )
    sections.append(
        table(
            "significance",
            va.merge(vad, on=["distance", "method"], suffixes=("_va", "_vad")),
            [
                "distance",
                "method",
                "significant_bh_va",
                "significant_maxT_va",
                "significant_bh_vad",
                "significant_maxT_vad",
            ],
            {
                "significant_bh_va": "VA BH layers /28",
                "significant_maxT_va": "VA max-stat layers /28",
                "significant_bh_vad": "VAD BH layers /28",
                "significant_maxT_vad": "VAD max-stat layers /28",
            },
        )
    )
    sections.append(f"""Across the full 336-fit families, **{int((significance.bh_q < 0.05).sum())} V–A** and **{int((vad_significance.bh_q < 0.05).sum())} VAD** fits pass BH q<0.05; **{int((significance.maxT_p < 0.05).sum())}** and **{int((vad_significance.maxT_p < 0.05).sum())}**, respectively, exceed the family maximum-statistic threshold. Pooled 672-test BH retains **313 V–A** and **331 VAD** fits. The shared-null maximum-score 95th percentiles are **0.2248** (V–A) and **0.2154** (VAD). Statistical correspondence is widespread, while the effect size and axis decomposition remain essential. Randomizing labels does not measure centroid sampling uncertainty.

Full adjusted results: {link("va_recovery")}, {link("vad_audit")}; null maxima: {link("va_null_maxima")}, {link("vad_null_maxima")}.

### Axis decomposition: valence drives the joint result

Valence and arousal have target variances **0.5459** and **0.1367**. Their shares in joint V–A SST are **0.7998** and **0.2002**. A model can therefore achieve a respectable joint fit while miscalibrating arousal. This is a property of the stated loss and native reference scale, not a claim that arousal is unimportant.
""")
    axis_means = (
        axes[axes.dimensions == 2]
        .groupby(["distance", "method", "axis"], as_index=False)[
            ["fitted_r2", "fitted_pearson", "loo_skill"]
        ]
        .mean()
    )
    sections.append(table("va_axis_means", axis_means))
    sections.append(
        fig(
            "05_va_axis_skill",
            "Predicting a held-out axis worse than the training-label mean gives negative skill.",
        )
    )
    sections.append(
        """Every method has negative average 2D arousal skill; mean arousal skill ranges from **−0.3394** for cosine–MDS to **−0.9369** for Euclidean–Isomap. At their selected peaks, cosine–MDS block 17 has arousal fitted R² **0.1628**, Pearson r **0.5438**, and LOO skill **0.1064**; cosine–Isomap block 8 reaches **0.2388**, **0.5977**, and **0.1875**. These exceptions show that an all-layer average should not be mistaken for failure at every layer.

The secondary affine decoder fits separate linear coefficients for the targets, allowing anisotropic scaling and shear rather than preserving reference shape. Mean 2D cosine–MDS joint LOO skill rises to **0.4902**, with arousal **0.0306**; 3D cosine–Isomap arousal decoding reaches **0.1207** on average. Some arousal information may be linearly accessible without being well expressed by a 2D similarity geometry. The affine results are exploratory and conditional on the same frozen geometry; see """
        + link("affine_decoding")
        + """.

## 7. Semantic maps, neutral anchoring and rank 1/3 views

The maps below show **unaligned** embedding coordinates translated so neutral is at the origin. Red/blue/gold/gray use the independent GoEmotions positive/negative/ambiguous/neutral taxonomy. Their 5NN background is a visualization fitted to those same nodes, not an out-of-sample classifier evaluation. Panel layers maximize the evaluated V–A fit within each distance/method; the fixed-depth rank-1/3 panels avoid that layer selection.
"""
    )
    sections.append(
        fig(
            "07_neutral_centered_maps",
            "All 28 labels appear in every panel; zoom the PDF for dense clusters. Quadrants are geometric coordinates, not named V/A states.",
        )
    )
    sections.append("""Positive and negative labels often separate along the first direction, while ambiguous labels and neutral occupy transition regions. In cosine–MDS block 17, the positive group lies mostly to the left and the negative group to the right. Cosine–Isomap block 8 reverses that sign and changes the second direction substantially. Such sign changes are arbitrary eigenvector choices. They cannot support conclusions about positive/negative arousal, nor can quadrant counts be compared across layers without an orientation convention.

Many non-cosine panels give grief, sadness and remorse a long arm, while gratitude or amusement anchor another direction. These are not identical to a clean symmetric V–A parabola: they can reflect sample support, centroid norm, distribution variation and embedding distortion. The reference-aligned panel below directly shows the mismatch rather than inferring reference axes from a raw map.
""")
    sections.append(
        fig(
            "08_rank1",
            "Block 17, all six distances and both methods. Each row is normalized by its own maximum absolute coordinate; vertical position is deterministic label jitter.",
        )
    )
    sections.append(
        fig(
            "09_rank3",
            "Fixed block 17, MDS. The viewing angle and occlusion limit interpretation; all exact values are in the CSV and original gallery.",
        )
    )
    sections.append(
        fig(
            "10_aligned_reference",
            "Separate evaluation view: black crosses are NRC labels, colored points are aligned embeddings, and segments show residuals. These aligned maps are not the neutral-centered exploration coordinates.",
        )
    )
    quadrant = pd.read_csv(run / "analyses/neutral_quadrant_counts.csv").query(
        'layer == 17 & distance == "cosine" & method == "mds"'
    )
    quadrant = (
        quadrant.pivot(index="quadrant", columns="emotion_group", values="emotion_count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    sections.append(
        "### Quadrants in the selected cosine–MDS map\n\n"
        "These reproduce the original block-17 counts in its unaligned orientation. "
        "All 12 positive labels are on the negative-x side; 8 of 11 negative labels "
        "are on the positive-x side. The coordinate signs are descriptive.\n"
    )
    sections.append(table("quadrants_cosine17", quadrant))
    sections.append("""### Categorical structure before embedding

A direct-RDM check restricts neighborhoods to the **12 positive and 11 negative** labels. It measures the fraction of each label's five nearest signed neighbors sharing its taxonomy sign. Under random assignment with these fixed counts, expected same-sign purity is **121/253 = 0.4783**. This is a descriptive benchmark; no independence assumption or binomial p-value is attached to the shared neighbor edges.
""")
    taxonomy = (
        read("taxonomy_structure")
        .groupby("distance", as_index=False)[
            ["positive_negative_5nn_purity", "between_within_ratio"]
        ]
        .mean()
    )
    sections.append(table("taxonomy", taxonomy))
    sections.append(
        fig(
            "28_taxonomy",
            "Signed-label purity is calculated in the direct distance matrices, before any embedding or NRC alignment.",
        )
    )
    sections.append("""Cosine shows the clearest categorical organization, mean purity **0.8602** and between/within-sign distance ratio **1.4096**. Other metrics have purity about **0.773–0.820**, although crossnobis's between/within ratio is only **1.1105**. Taxonomy clustering and exact continuous reference recovery answer related but different questions. Neutral and the four ambiguous labels are excluded only from this signed-neighbor diagnostic.

### Neutral's local position

Neutral at zero is a display translation. It does not establish that neutral is the centroid, a reference midpoint, a manifold vertex or a routing hub. At block 17, its nearest neighbors in the direct RDMs are:
""")
    near = (
        read("neutral_neighbors")
        .query("layer == 17 & rank <= 3")
        .pivot(index="distance", columns="rank", values="emotion")
        .reset_index()
    )
    near.columns = ["distance", "Nearest", "Second", "Third"]
    sections.append(table("neutral_neighbors", near))
    sections.append(
        """Realization is closest for five distances; crossnobis puts approval first. The full layerwise neighbor ranks are in """
        + link("neutral_neighbors")
        + """. Neutral's NRC valence/arousal values are still evaluated as supplied; the target is not redefined to zero by the display anchor.
"""
    )
    sections.append("""## 8. Three-dimensional VAD recovery

The 3D embedding is aligned to three native-scale reference axes with one shared scale. Its target variance shares are **62.76% valence, 15.71% arousal and 21.53% dominance**. A 3D VAD R² is not directly comparable to a 2D V–A R² as an “improvement”: both the input dimension and target loss change.
""")
    sections.append(
        table(
            "vad_ranking",
            vad,
            [
                "distance",
                "method",
                "mean_r2",
                "peak_r2",
                "peak_layer",
                "final_r2",
                "mean_loo_skill",
                "peak_loo_skill",
            ],
        )
    )
    sections.append(
        fig(
            "04_vad_recovery",
            "VAD has its own 336-fit permutation family; fitted R² and LOO skill evaluate different quantities.",
        )
    )
    sections.append("""Cosine–Isomap leads mean VAD fit (**0.4092**) and has the best VAD maximum (**0.5019 at block 12**). Cosine–MDS leads the direct MDS methods (**0.3845**, peak **0.4370 at block 17**). In the full VAD fit, adding a third coordinate does not make every affective axis recover equally well.
""")
    axis_means = (
        axes[axes.dimensions == 3]
        .groupby(["distance", "method", "axis"], as_index=False)[
            ["fitted_r2", "fitted_pearson", "loo_skill"]
        ]
        .mean()
    )
    sections.append(table("vad_axis_means", axis_means))
    sections.append(
        fig(
            "06_vad_axis_skill",
            "Dominance LOO skill is negative on average for all 12 distance/method choices; arousal recovery is more variable.",
        )
    )
    selected_axes = axes.query(
        'dimensions == 3 & distance == "cosine" & method == "isomap" & layer == 12'
    )
    sections.append(
        table("peak_vad_axes", selected_axes, ["axis", "fitted_r2", "fitted_pearson", "loo_skill"])
    )
    sections.append("""At the best VAD geometry, valence is strong, arousal is moderately recovered, and dominance remains worse than its mean baseline despite a positive fitted correlation. On average, the strongest 3D arousal LOO skills are still small: **0.026** for MMD–Isomap and **0.005** for cosine–Isomap. Dominance has negative mean LOO skill for every method. Significant *joint* VAD correspondence therefore cannot be described as independent validation of all three axes. The run does not test unique dominance information after conditioning on V–A.

## 9. How faithfully do the plots represent their input distances?

Stress is `sqrt(Σ(direct − embedded)² / Σdirect²)` over 378 unordered label pairs, without an extra rescaling. Lower is better. Shepard Spearman measures rank agreement between direct and embedded distances. Trustworthiness penalizes false embedded neighbors; continuity penalizes missing original neighbors. Both use **k=5**, independently of the graph parameter. For Isomap, these metrics below compare to the **direct RDM**, while residual variance compares to its geodesic target.
""")
    qmean = (
        quality.query("dimensions == 2")
        .groupby(["distance", "method"], as_index=False)[
            ["stress", "shepard_spearman", "trustworthiness", "continuity"]
        ]
        .mean()
    )
    sections.append(table("quality_2d", qmean))
    sections.append(
        fig(
            "11_stress",
            "Stress is dimensionless and normalized within each RDM; raw dissimilarity magnitudes differ across metrics.",
        )
    )
    sections.append(
        fig(
            "12_trustworthiness",
            "All methods are assessed at five original/embedded neighbors. More coordinates usually help, but graph distortions can persist.",
        )
    )
    sections.append(
        fig(
            "13_shepard_spearman",
            "A high rank correlation can coexist with poor absolute-distance fit.",
        )
    )
    sections.append(
        fig(
            "14_continuity",
            "Continuity asks whether original local relations survive the embedding.",
        )
    )
    sections.append("""Cosine–MDS has mean stress **0.3373** and trustworthiness **0.8472**. Cosine–Isomap raises trustworthiness to **0.8837** while raising stress to **0.4932**. Crossnobis–Isomap lowers stress from **0.5727 to 0.4038** but reduces trustworthiness from **0.8090 to 0.7543**. Energy/MMD show small 2D neighborhood gains, while Euclidean and sliced Wasserstein lose a little trustworthiness. A single “best embedding” score obscures these tradeoffs.

Classical MDS minimizes strain in the Gram representation; an Isomap embedding can have smaller direct-distance stress without contradicting MDS's optimization criterion. The fitted plot is also not a calibrated estimate of individual comment separability.
""")
    sections.append(
        fig(
            "30_shepard",
            "Fixed block 14; points are all 378 label pairs, normalized by that RDM’s maximum distance. Pair observations share labels and are not independent samples.",
        )
    )
    sections.append("""## 10. Intrinsic dimensions, spectra and eigengaps

All dimensions here describe the **28-node emotion geometry**, with at most 27 centered Euclidean dimensions. They do not estimate the intrinsic dimension of 45,446 sample activations or the full 1,024-feature hidden state. Positive-mass dimensions count how many positive MDS eigenvalues reach a fraction of the total positive spectrum; the participation ratio is `(Σλ+)²/Σλ+²`. Negative spectral mass is tracked separately, rather than silently renormalized away.
""")
    id_summary = (
        intrinsic.query('method == "mds"')
        .groupby("distance", as_index=False)
        .agg(
            mean_participation=("participation_ratio", "mean"),
            min_participation=("participation_ratio", "min"),
            max_participation=("participation_ratio", "max"),
            min_d90=("dimensions_90", "min"),
            max_d90=("dimensions_90", "max"),
            mean_negative_fraction=("negative_eigenvalue_fraction", "mean"),
        )
    )
    sections.append(table("intrinsic", id_summary))
    sections.append(
        fig(
            "15_dimensions",
            "Direct MDS and Isomap spectra describe different distance matrices. A lower Isomap participation ratio alone does not establish a truer low-dimensional manifold.",
        )
    )
    sections.append(
        fig(
            "16_spectrum",
            "The first ten normalized modes and cumulative mass, averaged over all blocks. Full spectra remain available for every layer and method.",
        )
    )
    sections.append("""The mean first-two-mode positive mass is **51.71% cosine**, **28.83% crossnobis**, **50.30% energy**, **43.75% Euclidean**, **51.19% MMD**, and **37.28% sliced Wasserstein**. Three modes raise these to approximately **63.79%, 38.51%, 64.46%, 56.28%, 65.42%, and 47.66%**, respectively. Thus even favorable visual maps leave considerable geometric structure unrepresented.

Crossnobis is the most diffuse direct spectrum (mean participation **12.46**), and sliced Wasserstein requires the most dimensions for 90% mass (**15–19**). Cosine is more concentrated but has appreciable negative spectral mass. These metric-dependent spectra support a useful low-dimensional *summary*, not a universal intrinsic rank of two or three.

### Eigengap randomization

For each direct RDM, the audit tests ratios `λ_k/λ_(k+1)` at **k=1…5**. It independently shuffles the 378 off-diagonal values, restores symmetry/zero diagonal, and recomputes MDS for **999** draws. This is an edge-value shuffle null, distinct from emotion-label permutations (which would preserve the spectrum). BH correction spans **168 × 5 = 840** gap tests. The minimum p is **0.001**. A shuffle destroys shared-label structure and can violate metric constraints; this is a benchmark for unusual spectral structure relative to that null, not a calibrated generative model of activation noise.
""")
    gaps = read("eigengaps")
    gapcounts = (
        gaps.assign(significant=gaps.bh_q < 0.05)
        .groupby(["distance", "gap_rank"])
        .significant.sum()
        .unstack()
        .reset_index()
    )
    gapcounts.columns = ["distance"] + [f"Gap {i}: significant blocks /28" for i in range(1, 6)]
    sections.append(table("eigengap_counts", gapcounts))
    sections.append(
        fig(
            "17_eigengaps",
            "Small q values indicate gaps large relative to the off-diagonal shuffle null.",
        )
    )
    sections.append("""Cosine has corrected first and second gaps in **27/28** blocks each, but many third, fourth and fifth gaps are also significant. Other metrics show recurring second or third gaps despite diffuse overall spectra. Multiple significant gaps across ranks, together with substantial residual mass, do not identify a unique fixed intrinsic dimension. In particular, they do not justify treating all remaining eigenmodes as noise.

## 11. Isomap, geodesics and the limits of a curvature claim

The main graph is the union of directed 5NN choices. All main graphs connect at the requested **k=5**; no connectivity expansion is needed in this run. Off-diagonal zero distances are replaced with machine epsilon as graph weights, and self nodes are explicitly excluded. The reconstructed graphs exactly match the saved geodesics. The audit sweep extends the original grid with **k=20 and 27**, while retaining 5NN quality evaluation.
""")
    comp = quality.pivot(
        index=["layer", "distance", "dimensions"],
        columns="method",
        values=["trustworthiness", "stress"],
    )
    deltas = pd.DataFrame(
        {m: comp[m]["isomap"] - comp[m]["mds"] for m in ["trustworthiness", "stress"]}
    ).reset_index()
    delta_mean = deltas.groupby(["distance", "dimensions"], as_index=False)[
        ["trustworthiness", "stress"]
    ].mean()
    trust = delta_mean.pivot(
        index="distance", columns="dimensions", values="trustworthiness"
    ).reset_index()
    trust.columns = ["distance", "Δ trust 1D", "Δ trust 2D", "Δ trust 3D"]
    sections.append(table("isomap_trust_deltas", trust))
    sections.append(
        fig(
            "18_isomap_changes",
            "Differences use the same direct-RDM evaluation target. Positive trustworthiness or negative stress differences favor Isomap.",
        )
    )
    sections.append("""The rank-1 “unrolling” gain is strongest for cosine (**+0.0550** mean trustworthiness). Energy and MMD gain about **+0.013** and **+0.011**; crossnobis loses **0.0145**. There is no uniform large rank-1 advantage across distances. In 2D, cosine gains **+0.0365**, crossnobis loses **−0.0546**, and the other changes are small. A legible rank-1 ordering is insufficient evidence for a curved one-dimensional emotion manifold.
""")
    geo = read("geodesic_pairs")
    geo_summary = (
        geo.groupby("distance").ratio.quantile([0.1, 0.5, 0.9, 0.95]).unstack().reset_index()
    )
    geo_summary.columns = [
        "distance",
        "10th percentile",
        "Median",
        "90th percentile",
        "95th percentile",
    ]
    sections.append(table("geodesic_quantiles", geo_summary))
    sections.append(
        fig(
            "19_geodesic_ratios",
            "Layerwise quantiles; the table pools equal-size label-pair sets across all blocks, omitting zero direct distances.",
        )
    )
    sections.append("""Median graph/direct ratios are approximately **1.28–1.52**, with pooled 90th percentiles **1.58–1.88**. These indicate substantial path inflation at this sparse graph scale. Sparse paths can inflate distances even for Euclidean data, so ratios alone do not identify manifold curvature. Cosine, energy and MMD also have paths shorter than some direct dissimilarities because their inputs can violate triangle inequality; a ratio below one is possible here. It occurs in roughly **3.90%, 2.94% and 6.59%**, respectively, of all unordered pairs before omitting zero denominators.
""")
    sections.append(
        fig(
            "22_graph_sensitivity",
            "Sweeps are diagnostic; the report does not retrospectively replace the main graph with the best-scoring setting.",
        )
    )
    sections.append("""Cosine's mean VA R² varies from **0.385 at k=3** to **0.443 at k=8**; trustworthiness declines as the graph becomes denser after k≈4. Crossnobis's mean VA fit peaks around **0.476 at k=6**, while its trustworthiness generally improves at larger k. Those are different selection objectives. At k=27, geodesics equal direct distances for triangle-respecting RDMs, but can still shorten nonmetric cosine/energy/MMD dissimilarities, so full-graph Isomap need not exactly equal their direct MDS.
""")
    sections.append(
        fig(
            "29_isomap_residuals",
            "Residual variance is 1 − Pearson² between geodesic and embedded distances; it is not a reference-alignment score.",
        )
    )
    sections.append("""The saved Isomap elbow uses the largest perpendicular departure from the endpoint line over **dimensions 1–10**, rather than a fixed residual-drop rule. Median elbows are **3** for cosine, energy, MMD and sliced Wasserstein, and **4** for crossnobis/Euclidean; individual elbows range **2–6**. A bounded-search elbow is a visualization heuristic, not a confidence interval or proof of intrinsic rank.

### An explicit check of parabolic appearance

As a limited diagnostic, the audit predicts the second unaligned coordinate from the first using a linear or quadratic polynomial, and compares leave-one-label-out squared prediction errors. First and second MDS coordinates are orthogonal by construction, so the in-sample linear fit has R²≈0; that is an algebraic fact. Quadratic predictive improvement over the linear baseline depends on orientation, outliers and the chosen first-two-mode plane.
""")
    curv = (
        read("curvature")
        .groupby(["distance", "method"], as_index=False)[["quadratic_r2", "quadratic_loo_skill"]]
        .mean()
    )
    sections.append(table("curvature", curv))
    sections.append("""The quadratic fit is not broadly predictive: mean LOO improvement is negative in **9 of 12** method/distance combinations. Cosine–MDS is **−0.021**, cosine–Isomap **+0.009**, and crossnobis–MDS **−1.780**. The strongest positive average is MMD–MDS (**+0.119**). This supports describing curved-looking configurations where visible, while withholding a general parabolic-manifold claim. This fixed-axis diagnostic is not rotation invariant and has no multiplicity-adjusted causal interpretation.

## 12. Metric assumptions, clipping and cosine normalization

Raw unbiased estimates and embedding distances must be distinguished. Across all layers, energy has **6** negative raw pairs and MMD **24**; their clipped embedding distances become zero for those pairs. At worst a single MMD layer has **14/378** zero pairs, versus **4/378** for energy. Crossnobis has no negative pairs in this run. Clipping preserves a usable nonnegative input but changes geometry and creates distance ties between distinct labels. Stable alphabetical tie ordering is used for neighbor diagnostics.
""")
    audit = (
        read("metric_audit")
        .groupby("distance", as_index=False)
        .agg(
            total_raw_negative_pairs=("raw_negative_pairs", "sum"),
            max_zero_pairs_per_layer=("zero_distance_pairs", "max"),
            mean_triangle_violation=("triangle_violation_fraction", "mean"),
            max_triangle_violation=("triangle_violation_fraction", "max"),
        )
    )
    sections.append(table("metric_audit", audit))
    sections.append(
        fig(
            "25_metric_assumptions",
            "Triangle violations are the fraction of 3,276 unordered label triples violating the triangle inequality; spectral fraction uses absolute eigenvalue mass.",
        )
    )
    sections.append("""Direct Euclidean is positive semidefinite up to rounding, as expected. Cosine has mean negative eigenvalue fraction **12.75%**; Isomap cosine raises it to **21.29%**. Main Isomap spectra have roughly **14–21%** mean negative mass across distances. The positive-mode coordinates therefore approximate matrices that are not globally Euclidean. Sliced Wasserstein has essentially no negative spectral mass in this finite dataset, without implying that every possible W1 geometry is Euclidean.

For a controlled cosine check, replace dissimilarity `d=1−cos` with unit-sphere chord distance `sqrt(2d)` on the **same normalized projected centroids**. This is Euclidean in the original normalized-vector space. It preserves pair rankings but changes metric geometry, the MDS spectrum and path sums.
""")
    chord = (
        read("cosine_metric_sensitivity")
        .groupby(["transform", "method"], as_index=False)[
            ["va_r2", "negative_eigenvalue_fraction", "trustworthiness", "stress"]
        ]
        .mean()
    )
    sections.append(table("cosine_chord", chord))
    sections.append(
        fig(
            "26_cosine_chord",
            "Chord and cosine use identical normalized centroids; stress is measured against each transform’s own input matrix.",
        )
    )
    sections.append("""Chord–MDS eliminates negative spectral mass and changes mean VA R² from **0.4447 to 0.4325**, a modest drop of **0.0122**. Thus the broad reference correspondence survives a principled Euclidean distance transformation. This sensitivity check does not imply that the arbitrary cosine origin is irrelevant: normalizing the neutral-centered projected centroids is itself a substantive representation choice. Isomap on chord distances still has substantial negative mass, as graph shortest-path geometry need not be Euclidean.

## 13. Agreement across distances and emotion-specific neighborhoods
""")
    ag = (
        read("distance_agreement")
        .query('method == "mds"')
        .groupby(["distance_i", "distance_j"], as_index=False)[
            ["spearman", "knn_overlap", "procrustes_r2"]
        ]
        .mean()
    )
    sections.append(table("distance_agreement", ag))
    sections.append(
        fig(
            "20_distance_agreement",
            "Means of matched-layer agreement; RDM correlations, neighborhood overlap and MDS shape similarity measure distinct forms of agreement.",
        )
    )
    sections.append("""Energy and MMD are almost duplicates at the RDM level (**ρ=0.9955**), in 5NN overlap (**0.9431**) and in 2D MDS shape (**R²=0.9904**). Euclidean and sliced Wasserstein also agree strongly (**ρ=0.9530**). By contrast, cosine versus crossnobis has **ρ=0.4219**, neighborhood overlap **0.5520**, and MDS shape R² **0.3271**. Crossnobis can agree reasonably with Euclidean distances (**ρ≈0.798**) while producing a very different two-dimensional projection (**R²≈0.367**). Truncation can conceal the shared higher-dimensional structure.
""")
    multidim = (
        pd.read_csv(run / "analyses/embedding_agreement.csv")
        .groupby(["dimensions", "method"], as_index=False)
        .procrustes_r2.mean()
    )
    sections.append(
        "### Agreement beyond two dimensions\n\n"
        "The source also retains 1D and 3D shape agreement for all distance pairs. "
        "The following means span all 15 distance pairs and 28 matched layers. "
        "They are descriptive averages, not cross-validated accuracy. Energy–MMD "
        "MDS agreement remains high from 1D through 3D (0.988, 0.990, 0.996), "
        "whereas cosine–crossnobis MDS agreement is 0.541, 0.327 and 0.430. "
        "A disagreement is therefore not resolved uniformly by one extra dimension.\n"
    )
    sections.append(table("multidimensional_agreement", multidim))
    sections.append(
        fig(
            "27_neighborhood_instability",
            "For each label and block, mean 1−overlap over all 15 distance pairs. Larger values mean local relations depend more on the metric.",
        )
    )
    sections.append("""Amusement (**0.488**), surprise (**0.459**), grief (**0.434**), pride (**0.410**), remorse (**0.407**) and sadness (**0.404**) have the most distance-dependent neighbors on average. Confusion/disapproval (both about **0.141**) and nervousness (**0.151**) are much more stable. Instability is not simply rarity: amusement is well sampled, while nervousness is rare. Every label-specific change is relative to the compared distance estimands, not an observed semantic change in the model.

Selected nearest/farthest direct pairs below use each distance's **best MDS V–A block**, so the depths differ and the table is illustrative rather than a controlled same-depth comparison.
""")
    sections.append(
        table(
            "selected_pair_extremes",
            read("selected_pairs").query("rank == 1"),
            ["distance", "layer", "kind", "emotion_i", "emotion_j", "value"],
        )
    )
    sections.append(
        """Cosine's closest pair is confusion–curiosity and farthest is annoyance–joy at block 17. Four final-block metrics put anger–gratitude farthest; crossnobis instead emphasizes gratitude–grief at block 14. The differences are scientifically meaningful: angular centroid structure, norm/distribution structure and whitened contrasts are not interchangeable measures of emotion separation. Full top-five lists are in """
        + link("selected_pairs")
        + """.

## 14. Cross-layer persistence and transitions
"""
    )
    sections.append(
        fig(
            "21_cross_layer",
            "Matched-label Procrustes R² between 2D embeddings, with each matrix aligned independently. Diagonals are one.",
        )
    )
    cross = read("cross_layer")
    adjacent = (
        cross[cross.layer_j == cross.layer_i + 1]
        .groupby(["distance", "method"], as_index=False)[["procrustes_r2", "rdm_spearman"]]
        .mean()
    )
    sections.append(table("adjacent_layer_similarity", adjacent))
    sections.append("""Adjacent direct RDM ranks are highly persistent: mean Spearman is approximately **0.966–0.988**. Neighboring 2D MDS shapes are also stable (**0.896–0.949**), while Isomap is less stable (**0.825–0.897**). For cosine, adjacent MDS shape R² is **0.949** versus **0.851** for Isomap. Small changes in sparse graph edges can have large effects on shortest paths and low-rank eigenvectors.

Heatmaps show broad middle-depth blocks of shared structure with early/final departures. They also explain why layers should not be treated as 28 independent experiments or bootstrapped as independent observations. The sudden final-block improvement for several direct MDS metrics occurs alongside changing neutral subspace dimension, so an attribution entirely to newly emerging affective computation would be premature.

## 15. Correspondence before embedding and nuisance controls

Representational similarity analysis (RSA) correlates the 378 direct pair distances with reference distances for **valence**, **arousal**, or **joint V–A**. Its one-sided randomization test permutes emotion labels jointly on matrix rows/columns; it does not treat shared-label pairs as independent observations. BH spans **504** raw RSA tests (168 RDMs × 3 targets).

The partial rank correlation residualizes rank-transformed pair values against differences in log class count, summed log class count, and differences in mean word count. Single-axis analyses also control for the other axis's reference distance. These are exploratory nuisance controls; p-values/q-values belong to the **raw RSA**, not to the partial correlations. Controls cannot remove all lexical, corpus or sample-estimation effects.
""")
    rsa = (
        read("reference_rsa")
        .groupby(["distance", "reference_axis"], as_index=False)[["spearman", "partial_spearman"]]
        .mean()
    )
    sections.append(table("rsa_means", rsa))
    sections.append(
        fig(
            "23_reference_rsa",
            "Partial correlations are descriptive; raw label-permutation tests are supplied separately in the source table.",
        )
    )
    sections.append("""Cosine shows the strongest direct joint reference agreement: mean **ρ=0.466**, increasing to **0.482** after the stated nuisance controls. Valence is similarly positive (**0.480 raw, 0.486 partial**). For the other distances, mean joint RSA is about **0.150–0.182**, while arousal RSA is negative on average; cosine arousal is near zero (**0.003 raw**, **0.059 partial**). The weak arousal results are therefore visible before dimensionality reduction and are not solely a failure of the Procrustes fitting procedure.

## 16. Sample balance, exact-text splits and uncertainty

Two checks operate on the **saved projected activation matrices**, at blocks **0, 7, 14, 17, 21, 27**. The first five evenly span depth; block 17 is an additional selected cosine–MDS peak. They do not refit the neutral covariance or grand-centroid origin. The projection is fixed using all splits, so these checks condition on a transductive transformation.

For each block, 200 draws retain exactly **47 comments per emotion** without replacement within each draw. This deliberately discards much of the support for common labels and retains all grief examples. The 2.5–97.5% ranges below summarize these **balanced subsampling draws**. They are not confidence intervals for the full-data score and are not a bootstrap of all labels or transformation uncertainty.
""")
    balanced = (
        read("balanced_draws")
        .groupby(["distance", "layer"])
        .va_r2.quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
    )
    balanced.columns = ["distance", "layer", "2.5% draw", "Median draw", "97.5% draw"]
    full = read("va_recovery").query('method == "mds"')[["distance", "layer", "va_procrustes_r2"]]
    balanced = balanced.merge(full, on=["distance", "layer"])
    sections.append(table("balanced_sampling", balanced))
    sections.append(
        fig(
            "24_sample_sensitivity",
            "Top: balanced-draw intervals. Bottom: exact-text-disjoint split centroids. Both condition on the same frozen all-split neutral projection.",
        )
    )
    sections.append("""Cosine block 17 drops from **0.5475** to balanced median **0.3819**, with a draw range **0.2858–0.5096**. The final block is more resilient: all-data **0.4133**, balanced median **0.4181** and range **0.3564–0.4962**. This change in the relative depth profile suggests that the chosen full-data peak partly depends on the amount of support available to each centroid. It does not imply that sample size caused every difference, nor that balanced 47-example centroids are intrinsically better estimators.

The second check globally deduplicates exact text, retaining the first occurrence, then computes separate train/validation/test centroids from frozen transformed vectors. Counts become **36,151 / 4,510 / 4,552**. The smallest classes have only **39 / 6 / 2** comments respectively, so held-out split centroids can be very noisy. These splits are disjoint in exact text, but not guaranteed independent in topic or near-duplicate content. The alignment to NRC is refitted on all labels within each split.
""")
    split = read("frozen_split_sensitivity")
    sections.append(
        table(
            "split_sensitivity",
            split,
            ["distance", "layer", "split", "va_r2", "rdm_spearman_to_train"],
        )
    )
    sections.append("""At cosine block 17, train/validation/test fits are **0.5456 / 0.3198 / 0.4364**; at block 27, **0.4207 / 0.3335 / 0.3737**. Test cosine RDM agreement with train remains substantial (**ρ≈0.831–0.873** from blocks 7–27). Euclidean final-block fit changes more strongly (**0.4098 train, 0.2224 validation, 0.1709 test**) despite test/train distance rank correlation **0.8019**. A stable distance ordering is compatible with unstable low-rank reference recovery.

The checks support persistence of some structure beyond one pooled set of comments, while exposing sampling sensitivity. They cannot establish fully prospective generalization because centering and neutral PCA were not fitted exclusively on training data. No distribution-metric resampling, model-seed replication, reference-rating uncertainty interval, or end-to-end projection bootstrap is included.
""")
    sections.append("""## 17. Which labels account for reference mismatch?

The peak cosine–MDS block 17 is a useful audit example because its shape is relatively stable across neighboring blocks. The table lists all labels, ordered by fitted squared V–A error. LOO error uses a separately refitted 27-label similarity transformation for that omitted label.
""")
    residuals = read("emotion_residuals").query(
        'dimensions == 2 & distance == "cosine" & method == "mds" & layer == 17'
    )
    sections.append(
        table(
            "cosine17_residuals",
            residuals.sort_values("squared_error", ascending=False),
            ["emotion", "squared_error", "loo_squared_error"],
        )
    )
    sections.append(
        """Sadness, remorse and grief contribute **32.32%** of the fitted residual sum. Sadness has NRC arousal **−0.424** but fitted arousal **+0.5195**; remorse has target valence **−0.794** but fitted **+0.1833**. These are substantive mismatches masked by a single joint score. Conversely, pride, embarrassment and joy have small fitted error in this configuration; that does not by itself imply that their sample centroids are reliably estimated.

Refitting after removing remorse changes joint R² by **+0.0406**, and removing sadness by **+0.0339**. Removing joy decreases it by **0.0216**. Removing neutral changes **0.5475 to 0.5500**, so this particular peak is not driven by neutral alone. These influence changes also alter the target denominator and the fit, and are not additive contributions. They are diagnostics, not a rationale for excluding inconvenient labels. See """
        + link("label_influence")
        + """ for all V–A and VAD label deletions.

## 18. Context from the existing representation comparison

The earlier [six-representation analysis](../representation_comparison/README.md) compares raw, centered and projected centroids with cosine–MDS. It uses the same definition of LT-NeutralProjected, so it provides a useful limited transformation contrast.
"""
    )
    comparison_path = Path("analyses/representation_comparison/tables/layerwise_scores.csv")
    if comparison_path.exists():
        previous = pd.read_csv(comparison_path)
        # Validate the shared geometry before citing the ancillary comparison.
        current = read("va_recovery").query('distance == "cosine" & method == "mds"')
        matched = current.merge(previous[["layer", "LT-NeutralProjected"]], on="layer")
        assert len(matched) == 28
        assert np.allclose(matched.va_procrustes_r2, matched["LT-NeutralProjected"], atol=1e-8)
        prev_summary = pd.read_csv("analyses/representation_comparison/tables/summary.csv")
        columns = [
            c
            for c in [
                "variant",
                "mean_r2",
                "median_r2",
                "max_r2",
                "best_layer",
                "peak_r2",
                "peak_layer",
            ]
            if c in prev_summary
        ]
        if len(columns) > 1:
            sections.append(
                table(
                    "previous_representations",
                    prev_summary[prev_summary.variant.str.startswith("LT-")],
                    columns,
                )
            )
    sections.append("""In that comparison, LT-Raw averages **0.1700**, LT-Centered **0.4358**, and LT-NeutralProjected **0.4447**. The large gain is associated with centering; projection adds only **0.0090** on average and loses to centering at 15 of 28 layers. At block 17, centering reaches **0.5551**, exceeding projected **0.5475**. Projection is therefore not a uniformly beneficial or isolated source of the observed correspondence.

This ancillary contrast applies to exact-centroid cosine–MDS. It should not be generalized to all six distances using a raw-run baseline with a different PCA pipeline. This run contains no matched raw/centered full-width distribution-distance ablation. Its results also do not identify a causal benefit from neutral removal.

## 19. What the evidence supports, and what remains open

The converging observations support **a stable valence-related organization of emotion-label representations**, visible in direct neighborhoods, direct reference RSA, low-rank maps, and corrected alignment tests. The evidence for faithful full V–A/VAD geometry is weaker: arousal and dominance often fail held-out axis baselines, several reference labels are poorly placed, and substantial spectral mass lies beyond the plotted dimensions.

The manifold evidence is mixed and depends on dissimilarity definition and graph choice. Sparse geodesic inflation, non-Euclidean input matrices, Isomap's negative spectra, and inconsistent rank-1 gains prevent a confident claim of a universal low-dimensional curved affective manifold. These outputs support investigating such structure rather than resolving its topology.

Material limits are:

1. **One model and extraction configuration.** No independent source-model commit, model-size comparison, model-weight replication, prompt ablation or pooling ablation is established by this run.
2. **Transductive and selected analyses.** All splits enter centering/projection and original RDM construction. LOO labels test target fitting conditional on geometry. Peak layers and ancillary graph/preprocessing checks remain exploratory.
3. **Uneven and dependent text support.** Rare centroids, repeated or related comments, capped distributions, and finite random Wasserstein directions add uncertainty that label permutation does not capture.
4. **Reference scope.** Lexical NRC label ratings are not per-comment human emotion intensities. Their calibration and upstream sampling error are not independently re-estimated. The axes have unequal variance and may share information.
5. **Finite geometry.** Twenty-eight label nodes cannot determine hidden-state topology or sample-level intrinsic dimension. MDS/Isomap are descriptive approximations with explicit distortions.
6. **Missing behavioral evidence.** No classification outcomes, calibrated confidence, misclassification boundaries, steering interventions or neural recordings are present to assess those downstream claims.

The most informative next experiments are a training-only fit of both centering and neutral PCA with evaluation on fresh comments; matched full-space raw/centered/projected ablations for every distance; resampling comments **and refitting the full transformation**; external label-level affect references and per-comment ratings; independent model/pooling replications; and preregistered layer/graph selection before behavioral testing. These are follow-up experiments, not conclusions already supplied by this analysis.

## 20. Reproduction and artifact navigation

The complete report can be rebuilt from the existing run without model inference:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
python experiments/04_analyze_lt_neutral_projected.py
```

To redraw figures and rebuild this Markdown from the saved analysis tables:

```bash
PYTHONPATH=src python experiments/04_analyze_lt_neutral_projected.py --render-only
```

The command exposes `--run` for relocating the frozen input, `--report` for the analysis destination, `--render-only` to redraw saved tables, and `--bundle-only` to refresh this local package without rerunning computations or plotting. It reproduces the stated 4,999/999/200 audit with seed 1701. The reusable scientific function accepts other settings, but this curated narrative explicitly requires the documented run and audit settings. Report figures are rendered **only after writing and rereading their own plot-ready CSVs**. Intermediate/summary tables and plot-ready tables are separate. The report is packaged in `analyses/lt_neutral_projected/` with local `figures/`, `tables/`, `plot_data/`, `metadata/`, `source/` and `plots/` subdirectories. The original gallery data are in `plot_data/original/`. Canonical generated artifacts remain under the source run's `report/` in `outputs/`, so rebuilding the computations does not depend on relocating the original run. Structured operational progress is centralized at `logs/qwen--qwen3-0.6b__lt-neutral-projected-analysis-20261021/run.log`.

The audit independently verifies the frozen scientific scores and geodesics. Focused tests also check exact three-axis similarity recovery including held-out labels, and demonstrate why perfect axis correlation can coexist with poor shared-scale axis R². Exact analysis-library versions are saved with the report artifacts; numerical tolerances are stated in Section 1.
""")
    sections.append(f"""- [All report figures and CSV mappings]({metadata}/figure_index.csv)
- [Analysis environment]({metadata}/analysis_environment.json)
- [Analysis source hashes]({metadata}/analysis_source_sha256.json) and [source snapshot]({base}/source/)
- [Audit provenance, reference hash and input RDM hashes]({metadata}/provenance.json)
- [Validation results]({metadata}/validation.json)
- [Original self-contained exploration gallery](exploration.html)
- [Local gallery plot manifest]({metadata}/plot_manifest.json)
- [Original run validation]({runlink}/analyses/run_validation.json)
- {link("source_inventory", "Complete source-artifact inventory")}

## Appendix A. Full layerwise V–A scores

Columns identify distance and embedding method. Scores are the independently recomputed fits; layers are zero based. These tables preserve the numerical context of the layerwise figures rather than displaying only favorable examples.
""")
    for title, name, score in [
        ("V–A", "va_recovery", "va_procrustes_r2"),
        ("VAD", "vad_audit", "vad_procrustes_r2"),
    ]:
        scores = read(name)
        scores["configuration"] = scores.distance + "/" + scores.method
        pivot = scores.pivot(index="layer", columns="configuration", values=score).reset_index()
        if title == "VAD":
            sections.append("## Appendix B. Full layerwise VAD scores\n")
        sections.append(table("full_" + name, pivot))
    sections.append("## Appendix C. Sample counts and reference values\n")
    splits = (
        read("split_balance").pivot(index="emotion", columns="split", values="size").reset_index()
    )
    counts = source[["emotion", "sample_count", "distribution_sample_count", "mean_words"]].merge(
        splits, on="emotion"
    )
    sections.append(table("full_counts", counts))
    ref = pd.read_csv(run / "references/goemotions_nrc_vad.csv")
    sections.append(table("reference_values", ref))
    sections.append("""The reference scores remain on their supplied native scale; display translation of neutral does not alter them. All scores treat labels equally regardless of sample count.

## Appendix D. Figure and source-table coverage

The report embeds all 30 new figures, with an independently reusable CSV and PDF for each. The source run contains 1,008 exploration views spanning every layer/distance/method/dimension; selected maps are embedded above and the original gallery exposes all views. The inventory includes the remaining original diagnostic plots, source matrices, embeddings, sampling records and projection checkpoints.
""")
    figure_table = pd.DataFrame(
        [
            dict(
                Figure=row["name"],
                Purpose=row["title"],
                Rows=row["rows"],
                Data=f"[{row['csv']}]({base}/{row['csv']})",
            )
            for row in registry.values()
        ]
    )
    sections.append(table("figure_coverage", figure_table))
    coverage = [
        (
            "Dataset samples/class counts/labels",
            "Sample support, duplicate audit, exact-text split and balanced checks",
        ),
        (
            "Neutral projection checkpoints and neutral_projection.csv",
            "Removed subspace dimensions/variance; exact-centroid reconstruction",
        ),
        (
            "All 168 raw and embedding RDMs",
            "Finite/symmetric labels, clipping, triangles, RSA, eigengaps, pairs and agreement",
        ),
        (
            "All 1,008 coordinate configurations",
            "1D–3D quality; 2D V–A/3D VAD; maps and label influence",
        ),
        (
            "Full MDS/Isomap spectra and intrinsic_dimension.csv",
            "Participation, retained mass, negative mass, threshold dimensions",
        ),
        (
            "Geodesics and Isomap neighbor sweep",
            "Exact rebuild, path ratios, k=3…27 sensitivity, residuals",
        ),
        (
            "va_recovery.csv / va_alignment.csv",
            "Exact score reproduction, adjusted label permutations, LOO axes/residuals",
        ),
        (
            "vad_recovery.csv / vad_alignment.csv",
            "Exact score reproduction and fresh axis-level/LOO audit",
        ),
        (
            "rdm_agreement.csv / embedding_agreement.csv",
            "Full matched-layer distance/rank/neighborhood and 2D shape agreement",
        ),
        ("neighborhood_change.csv", "All labels and depth, plus instability ranking"),
        ("cross_layer_similarity.csv", "Full 2D layer-pair maps plus direct RDM correlations"),
        (
            "neutral_quadrant_counts.csv",
            "Counts reproducible from exploration coordinates; arbitrary signs prevent affective quadrant interpretation",
        ),
        (
            "Original plots / exploration.html / plot_manifest.json",
            "All-gallery navigation; main quantitative patterns redrawn from CSVs",
        ),
        (
            "manifest / references / orchestration / upload verification",
            "Configuration and provenance inspected; transfer status is operational context, not scientific evidence",
        ),
    ]
    sections.append(
        table("artifact_coverage", pd.DataFrame(coverage, columns=["Source family", "Analysis"]))
    )
    (destination / "README.md").write_text("\n\n".join(sections).strip() + "\n")
    return destination / "README.md"
