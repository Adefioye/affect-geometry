# NRC affect references

`goemotions_nrc_va.csv` is the existing 28-label V–A reference, including the exact lexical
`source_term` for each label. Its scores use NRC VAD v2.1's `[-1, 1]` scale.

`goemotions_nrc_vad.csv` is the equivalent local reference with a `dominance` column added.
It contains the same 28 labels in the same order, and its valence and arousal values match
`goemotions_nrc_va.csv` exactly. All three axes retain the native `[-1, 1]` scale, with exact
source-term matches and no imputation or scale conversion.

The LT-NeutralProjected runner uses this local CSV by default through
`analysis.vad_reference`. Its `reference` stage validates the CSV and snapshots it, together
with provenance, into `outputs/.../references/`. **No internet connection, ZIP, or full lexicon
is required.** Select a different prepared CSV with `--vad-reference /path/to/reference.csv`.

`goemotions_nrc_vad.provenance.json` records the NRC v2.1 release, source URL, source and
reference SHA-256 hashes, and the exact agreement check against the existing V–A reference.
The [official download](https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip)
returned HTTP 406 during preparation, so the scores were extracted from a
[university-hosted copy of the v2.1 TSV](https://gitlab.uni-hannover.de/nagengast.m/nagengast25-ma-thesis/-/blob/main/data/NRC-Lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt).
The full source TSV stays under `outputs/reference_sources/`; it is not needed for subsequent runs.

To rebuild a run's reference from an already downloaded official ZIP or English TSV instead:

```bash
python experiments/03_run_lt_neutral_projected.py \
  --nrc-vad-lexicon /path/to/NRC-VAD-Lexicon-v2.1.zip
```

Use and cite the lexicon according to the creator's terms on the
[official page](https://saifmohammad.com/WebPages/nrc-vad.html).
