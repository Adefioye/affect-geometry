"""Training-only EXP5 cohorts and deterministic text-only donor retrieval."""

import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from affect_geometry.classification.protocol import EMOTIONS, digest, parse_strict


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def child_seed(seed, *parts):
    return int(digest(":".join(map(str, (seed, *parts))))[:16], 16)


def capped_by_label(frame, cap, seed, purpose):
    selected = []
    for label in EMOTIONS:
        group = frame[frame.emotion.eq(label)].sort_values("sample_id")
        rng = np.random.default_rng(child_seed(seed, purpose, label))
        selected.extend(group.iloc[rng.permutation(len(group))[:cap]].to_dict("records"))
    return (
        pd.DataFrame(selected, columns=frame.columns)
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def build_cohorts(rows, config):
    rows = rows.copy()
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any():
        raise ValueError("Missing/duplicate sample IDs")
    if set(rows.emotion) - set(EMOTIONS):
        raise ValueError("Unknown gold labels")
    if not rows.status.eq("completed").all() or rows.correct.isna().any():
        raise ValueError("Resolve source inference failures before EXP5")
    for row in rows.to_dict("records"):
        if (parse_strict(row["raw_response"]) == row["emotion"]) != bool(row["correct"]):
            raise ValueError("Saved strict correctness disagrees with EXP2 parsing")
    rows["duplicate_group"] = rows.text.map(normalize_text).map(digest)
    train = rows[rows.source_split.eq("train")].copy()
    counts = train.groupby("duplicate_group").emotion.nunique()
    conflict = set(counts[counts.gt(1)].index)
    bank = train[train.correct & ~train.duplicate_group.isin(conflict)]
    bank = bank.sort_values("sample_id").drop_duplicates("duplicate_group")
    discovery = capped_by_label(bank, config.discovery_per_emotion, config.seed, "discovery")
    selection = capped_by_label(discovery, config.selection_per_emotion, config.seed, "selection")
    test = rows[rows.source_split.eq("test")].sort_values("sample_id").reset_index(drop=True)
    audit = rows[["sample_id", "source_split", "emotion", "duplicate_group"]].copy()
    all_counts = rows.groupby("duplicate_group").emotion.nunique()
    audit["conflicting_labels"] = audit.duplicate_group.isin(
        set(all_counts[all_counts.gt(1)].index)
    )
    audit["cross_split"] = audit.duplicate_group.isin(
        set(rows.groupby("duplicate_group").source_split.nunique().loc[lambda s: s.gt(1)].index)
    )
    mapping = {}
    errors = train[~train.correct & train.predicted_emotion.isin(EMOTIONS)]
    for index, label in enumerate(EMOTIONS):
        frequencies = errors.loc[errors.emotion.eq(label), "predicted_emotion"].value_counts()
        choices = [v for v in EMOTIONS if v != label and frequencies.get(v, 0) > 0]
        mapping[label] = (
            min(choices, key=lambda v: (-frequencies[v], EMOTIONS.index(v)))
            if choices
            else EMOTIONS[(index + 1) % len(EMOTIONS)]
        )
    return discovery, selection, test, audit, mapping


def grams(text):
    text = normalize_text(text)
    return {text} if len(text) < 5 else {text[i : i + 5] for i in range(len(text) - 4)}


class DonorRetriever:
    def __init__(self, bank, duplicate_threshold=0.90):
        self.bank = bank.sort_values("sample_id").reset_index(drop=True)
        self.threshold = duplicate_threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=None,
            min_df=1,
            max_df=1.0,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            norm="l2",
        )
        self.matrix = None
        if len(bank):
            try:
                self.matrix = self.vectorizer.fit_transform(self.bank.text)
            except ValueError as error:
                if "empty vocabulary" not in str(error):
                    raise
        self.gram_sets = [grams(v) for v in self.bank.text]
        self._cached_recipient = None
        self._rankings = {}

    def ranked(self, recipient, label):
        identity = (recipient["sample_id"], digest(recipient["text"]))
        if identity != self._cached_recipient:
            self._cached_recipient, self._rankings = identity, {}
        if label in self._rankings:
            return self._rankings[label]
        similarities = np.zeros(len(self.bank))
        if self.matrix is not None:
            similarities = self.matrix @ self.vectorizer.transform([recipient["text"]]).T
            similarities = similarities.toarray().ravel()
        recipient_grams = grams(recipient["text"])
        eligible, excluded = [], []
        for i, donor in enumerate(self.bank.to_dict("records")):
            if donor["emotion"] != label:
                continue
            union = recipient_grams | self.gram_sets[i]
            similarity = len(recipient_grams & self.gram_sets[i]) / max(1, len(union))
            if donor["sample_id"] == recipient["sample_id"] or similarity >= self.threshold:
                excluded.append(dict(donor_id=donor["sample_id"], reason="self_or_near_duplicate"))
                continue
            eligible.append(dict(donor=donor, similarity=float(similarities[i])))
        eligible.sort(key=lambda r: (-r["similarity"], r["donor"]["sample_id"]))
        self._rankings[label] = (eligible, excluded)
        return eligible, excluded
