"""Load a local NRC VAD reference, or prepare it from the v2.1 lexicon."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.logging import get_logger, progress

NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_VAD_HOMEPAGE = "https://saifmohammad.com/WebPages/nrc-vad.html"
AXES = ["valence", "arousal", "dominance"]


def read_vad_reference(payload: bytes) -> pd.DataFrame:
    """Validate a prepared emotion-level CSV without needing the original lexicon."""
    table = pd.read_csv(io.BytesIO(payload), keep_default_na=False)
    required = ["emotion", "source_term", *AXES]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"VAD reference CSV is missing required columns: {missing}")
    if table["emotion"].duplicated().any():
        raise ValueError("VAD reference CSV contains duplicate emotion labels")
    if table[["emotion", "source_term"]].eq("").any().any():
        raise ValueError("VAD reference CSV contains empty labels or source terms")
    scores = table[AXES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(scores.to_numpy()).all() or (scores.abs() > 1).any().any():
        raise ValueError("VAD reference scores must be finite and on the [-1, 1] scale")
    table[AXES] = scores
    return table


def read_nrc_vad(payload: bytes) -> tuple[pd.DataFrame, str]:
    """Read a v2.1 English TSV, or locate the English VAD table in its ZIP."""
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            candidates = []
            for name in sorted(archive.namelist()):
                if name.startswith("__MACOSX/") or not name.lower().endswith((".txt", ".tsv")):
                    continue
                try:
                    table, _ = read_nrc_vad(archive.read(name))
                except (ValueError, UnicodeError, pd.errors.ParserError):
                    continue
                candidates.append((table, name))
        if len(candidates) != 1:
            raise ValueError(f"Expected one English NRC VAD table; found {len(candidates)}")
        return candidates[0]
    table = pd.read_csv(io.BytesIO(payload), sep="\t", encoding="utf-8-sig", keep_default_na=False)
    table.columns = [str(column).strip().lower() for column in table.columns]
    # v2 releases use either Word or Term for the lexical entry.
    table = table.rename(columns={"term": "word"})
    if not {"word", *AXES}.issubset(table.columns):
        raise ValueError("Expected NRC v2.1 columns: Word/Term, Valence, Arousal, Dominance")
    table = table[["word", *AXES]].copy()
    if table.isna().any().any():
        raise ValueError("NRC VAD reference contains missing terms or values")
    table["word"] = table["word"].astype(str).str.strip().str.lower()
    if table["word"].duplicated().any():
        raise ValueError("NRC VAD reference contains duplicate terms")
    if (table["word"] == "").any():
        raise ValueError("NRC VAD reference contains empty terms")
    scores = table[AXES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(scores.to_numpy()).all() or (scores.abs() > 1).any().any():
        raise ValueError("NRC VAD v2.1 scores must be finite and on the [-1, 1] scale")
    table[AXES] = scores
    return table, "local-tsv"


def prepare_nrc_vad(
    labels: list[str],
    destination: Path,
    *,
    local_lexicon: Path | None = None,
    local_reference: Path | None = None,
) -> Path:
    """Snapshot a prepared CSV, or derive one from a local/downloaded lexicon."""
    if local_reference is not None and local_lexicon is not None:
        raise ValueError("Provide either a VAD reference CSV or a source lexicon, not both")
    destination.mkdir(parents=True, exist_ok=True)
    if local_reference is not None:
        payload = local_reference.read_bytes()
        table = read_vad_reference(payload).set_index("emotion")
        missing = sorted(set(labels) - set(table.index))
        if missing:
            raise ValueError(f"VAD reference CSV is missing emotions: {missing}")
        path = destination / "goemotions_nrc_vad.csv"
        table.loc[labels].reset_index().to_csv(path, index=False)
        provenance = {
            "source": str(local_reference.resolve()),
            "kind": "prepared_reference_csv",
            "url": NRC_VAD_URL,
            "homepage": NRC_VAD_HOMEPAGE,
            "version": "2.1",
            "scale": "[-1, 1]",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "labels": labels,
        }
        source_provenance = local_reference.with_suffix(".provenance.json")
        if source_provenance.is_file():
            provenance["upstream_provenance"] = json.loads(source_provenance.read_text())
        (destination / "nrc_vad_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        get_logger().info("reference.loaded source=%s labels=%d", local_reference, len(labels))
        return path
    if local_lexicon is not None:
        payload = local_lexicon.read_bytes()
        source = str(local_lexicon.resolve())
    else:
        get_logger().info("reference.download url=%s", NRC_VAD_URL)
        request = urllib.request.Request(NRC_VAD_URL, headers={"User-Agent": "affect-geometry"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Could not download {NRC_VAD_URL}. Download v2.1 from {NRC_VAD_HOMEPAGE} "
                "and pass --nrc-vad-lexicon /path/to/official.zip (or its English TSV)."
            ) from error
        source = NRC_VAD_URL
    table, member = read_nrc_vad(payload)
    lookup = table.set_index("word")
    missing = sorted(set(labels) - set(lookup.index))
    if missing:
        raise ValueError(f"NRC VAD v2.1 is missing exact label matches: {missing}")
    records = []
    for label in progress(labels, description="Preparing NRC VAD label reference", unit="emotion"):
        records.append(
            {
                "emotion": label,
                "source_term": label,
                **lookup.loc[label].to_dict(),
                "reference_version": "2.1",
                "reference_scale": "[-1, 1]",
                "reference_url": NRC_VAD_URL,
            }
        )
    suffix = ".zip" if zipfile.is_zipfile(io.BytesIO(payload)) else ".tsv"
    (destination / f"nrc_vad_v2.1{suffix}").write_bytes(payload)
    path = destination / "goemotions_nrc_vad.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    (destination / "nrc_vad_provenance.json").write_text(
        json.dumps(
            {
                "source": source,
                "url": NRC_VAD_URL,
                "homepage": NRC_VAD_HOMEPAGE,
                "version": "2.1",
                "scale": "[-1, 1]",
                "archive_member": member,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "matching": "exact label",
                "labels": labels,
            },
            indent=2,
        )
        + "\n"
    )
    return path
