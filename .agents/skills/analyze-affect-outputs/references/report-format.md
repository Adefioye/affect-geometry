# Analysis report format

Use this format as a decision guide rather than padding the report with empty sections. Omit a
section when its analysis is not present, and explain material missing evidence where it affects a
conclusion.

## Output layout

```text
<run-root>/analysis-report/
|-- analysis.md
`-- plots/
    |-- <descriptive-figure-name>.png
    `-- ...
```

Use lowercase, descriptive plot filenames. Embed them from `analysis.md` with relative Markdown
paths. Do not copy canonical CSVs into `analysis-report/`.

## Recommended report sections

### 1. Run and question

State the research question and identify the run. A compact table should include only identifiers
needed to interpret the results, such as model, run ID, dataset, pooling, representation,
layers, distances, embedding methods, target norms, and seed.

### 2. Data and validation

Summarize sample and emotion coverage, missing artifacts, validation failures, exclusions, and
comparability constraints. Say whether the analysis is complete enough to answer the question.

### 3. Main results

Organize by scientific claim rather than by filename. For each claim:

1. provide a compact Markdown table of the decisive metrics;
2. embed at most the figures that materially clarify the result;
3. explain what the numbers and visual pattern jointly show;
4. discuss uncertainty, stability, and plausible alternative interpretations.

Prefer informative column names with units or definitions. Include sample size, layer, method,
distance, representation, and uncertainty interval when they are necessary to avoid ambiguity.
Round only for presentation and retain enough precision to distinguish relevant values.

### 4. Cross-condition comparison

When comparing models, layers, methods, poolings, or preprocessing conditions, use the same metric
definitions and matched data support. Report absolute values alongside differences. Avoid ranking
conditions whose confidence intervals or stability results do not support a meaningful ordering.

### 5. Robustness and diagnostics

Report the checks that could reverse the interpretation: held-out performance, permutation or
bootstrap results, multiplicity correction, alternative distances, graph connectivity, negative
eigenvalue mass, seed/fold variation, and neighborhood stability as applicable.

### 6. Conclusions and limitations

Answer the research question directly. Separate:

- supported conclusions;
- suggestive but exploratory observations;
- conclusions not supported by the available run;
- concrete limitations or missing analyses.

### 7. Provenance

List the relative paths of the manifest, validation files, canonical CSVs, and existing figures
used. This is a source index, not a copy of their contents.

## Plot commentary

Every embedded plot needs nearby prose covering:

- what is encoded on each axis, color, panel, or line;
- the most important visible pattern;
- the numerical table entry that corroborates or qualifies it;
- instability, outliers, clipping, missing conditions, or selection effects;
- what cannot be concluded from the figure.

Avoid decorative plots, repeated versions of the same comparison, and figures whose information is
already clearer in a small table.

## Failure handling

If canonical data are missing or inconsistent, still create the report only when it can clearly
document the incomplete result. Do not fabricate values, infer identifiers from folder names when
metadata disagree, or repair source outputs without explicit authorization.
