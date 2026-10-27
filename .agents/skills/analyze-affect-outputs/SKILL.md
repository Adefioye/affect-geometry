---
name: analyze-affect-outputs
description: Analyze completed affect-geometry runs when asked to interpret saved local CSV metrics and plots, compare layers, methods, representations, or models, or produce a Markdown results report. Do not use for activation extraction or changing the experimental pipeline unless explicitly requested.
---

# Analyze Affect Outputs

Turn an existing run's canonical outputs into a concise, evidence-based report. The deliverable is
one Markdown analysis with compact metric tables and embedded figures; plots live in a sibling
folder. Avoid redundant output artifacts.

## Inputs and scope

- Resolve the exact run root before analysis. Keep models, run IDs, pooling strategies, and raw
  versus neutral-projected representations distinct unless the request explicitly compares them.
- Prefer saved manifests, provenance, validation results, analysis tables, and plot-ready CSVs.
  Treat filenames and schemas as discoverable rather than assuming that every run contains the
  same files.
- Analyze saved outputs only. Do not rerun extraction, RDM construction, or embedding stages to
  fill a missing result unless the user explicitly asks.
- Never delete, overwrite, or reorganize canonical run artifacts while reporting.

## Workflow

1. Inventory the selected run's CSVs, metadata, and figures. Record the model, run ID, dataset,
   representation, pooling, layers, distance notions, embedding methods, and random seeds when
   available.
2. Validate that tables being joined have compatible labels and experimental identifiers. Flag
   missing files, duplicate rows, inconsistent emotion orderings, invalid values, failed
   validations, or incomplete layers instead of silently excluding them.
3. Select metrics that answer the research question. Summarize important values in memory and
   report them as Markdown tables; do not dump every column or create a second CSV containing the
   same information.
4. Reuse an existing figure when it is valid and readable. Generate a new figure only when needed
   to answer the question or when the existing figure cannot be traced to canonical data. Put all
   report-specific figures in `analysis-report/plots/`.
5. Write `analysis-report/analysis.md` and embed report figures with relative paths such as
   `plots/va-alignment.png` so the report remains portable after upload or cloning.
6. Interpret every reported table and every embedded plot. State the main pattern, supporting
   numerical evidence, uncertainty or stability, anomalies, and the limit of the inference.
7. Finish with a direct answer to the research question, robustness assessment, limitations, and
   a short provenance list identifying the canonical source files used.

Read [references/report-format.md](references/report-format.md) before producing or substantially
revising a report.

## Metric selection

Prioritize metrics that materially support a conclusion. Depending on the run, these can include:

- VA alignment: out-of-fold Procrustes R-squared, permutation significance, aligned Pearson and
  Spearman correlations, and per-emotion error;
- embedding fidelity: Stress-1, Shepard correlation, trustworthiness, continuity, graph
  connectivity, and geodesic-to-direct-distance summaries;
- dimensionality: positive spectral mass thresholds, participation ratio, significant eigengaps,
  negative eigenvalue mass, and residual-variance elbows;
- robustness and agreement: prompt-bootstrap intervals, fold/seed stability, RDM correlations,
  embedding similarity, and neighborhood overlap;
- classification context, when present: overall and per-emotion accuracy, valid-label rate,
  confusion patterns, and correct-sample counts.

Do not choose a layer or method from a metric evaluated on the same held-out targets used to claim
generalization. Preserve the run's preregistered selection rule, cross-validation boundary, and
multiplicity correction. Distinguish descriptive full-data alignment from held-out evidence.

## Artifact policy

- The canonical input CSVs remain the reproducibility source. Do not duplicate them inside the
  report directory.
- A new plot must be reproducible from an existing canonical tidy CSV. If no suitable source table
  exists, state that limitation rather than relying on hidden in-memory data. Create a new tidy CSV
  only when the user requests a new derived artifact or the repository's essential-plot rule
  requires data not already stored canonically.
- Do not save standalone metric JSON, CSV, or text summaries when the same values fit naturally in
  the Markdown report.
- Keep full-detail tables in their canonical source files. Markdown tables should be small enough
  to compare without scrolling through raw observations.
- Report relative paths and sufficient identifiers for every source and figure.

## Interpretation boundaries

- Describe association, alignment, preservation, or prediction; do not imply causality from
  geometry alone.
- A visually persuasive embedding is not evidence without quantitative fidelity and stability.
- Procrustes alignment permits rotation, reflection, translation, and, when configured, scale; it
  does not show that native axes already equal valence and arousal.
- Compare good and poor classifiers without treating classification accuracy as an eligibility
  gate unless the study design explicitly requires one.
- Clearly separate confirmatory results from exploratory observations and distinguish absent
  evidence from evidence of absence.
