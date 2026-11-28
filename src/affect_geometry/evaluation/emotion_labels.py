"""Native complete-label likelihoods without constrained or substituted generation."""

from dataclasses import asdict, dataclass
from math import isfinite

import torch

from affect_geometry.classification.protocol import EMOTIONS, parse_strict


@dataclass(frozen=True)
class LabelPath:
    label: str
    label_ids: tuple
    stop_id: int

    @property
    def ids(self):
        return (*self.label_ids, self.stop_id)


class EmotionLabelScorer:
    def __init__(self, model, tokenizer, *, stop_id=None, prefix=" ", budget=10, batch_size=1):
        self.model, self.tokenizer = model, tokenizer
        stops = model.generation_config.eos_token_id
        if stops is None:
            stops = tokenizer.eos_token_id
        stops = [stops] if isinstance(stops, int) else list(stops or [])
        if not stops or any(v is None for v in stops):
            raise ValueError("A declared native stop token is required")
        self.stop_id = stops[0] if stop_id is None else stop_id
        if self.stop_id not in stops:
            raise ValueError("Canonical stop is not in the native generation stop set")
        self.budget, self.batch_size = budget, batch_size
        self.paths = {}
        self._checked_boundaries = set()
        for label in EMOTIONS:
            ids = tuple(tokenizer.encode(prefix + label, add_special_tokens=False))
            if not ids or parse_strict(tokenizer.decode(ids, skip_special_tokens=True)) != label:
                raise ValueError(f"Canonical continuation does not decode to {label}")
            if any(v in tokenizer.all_special_ids for v in ids) or len(ids) + 1 > budget:
                raise ValueError(f"Canonical path cannot terminate within budget: {label}")
            self.paths[label] = LabelPath(label, ids, self.stop_id)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def metadata(self):
        return {label: asdict(path) for label, path in self.paths.items()}

    def forward_paths(self, prompt_ids, labels):
        if not prompt_ids:
            raise ValueError("A nonempty pre-answer prompt is required")
        identity = tuple(prompt_ids)
        if identity not in self._checked_boundaries:
            prefix = self.tokenizer.decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for label, path in self.paths.items():
                combined = self.tokenizer.decode(
                    list(prompt_ids) + list(path.label_ids),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if (
                    not combined.startswith(prefix)
                    or parse_strict(combined[len(prefix) :]) != label
                ):
                    raise ValueError(
                        f"Canonical label fails at the actual prompt boundary: {label}"
                    )
            self._checked_boundaries.add(identity)
        sequences = [list(prompt_ids) + list(self.paths[label].ids[:-1]) for label in labels]
        width = max(map(len, sequences))
        limit = getattr(self.model.config, "max_position_embeddings", float("inf"))
        if width + 1 > limit:
            raise ValueError("Context overflow; EXP2 prohibits truncation")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.stop_id
        ids = torch.full((len(labels), width), pad, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[i, : len(sequence)] = 1
        output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        # Convert only continuation-prediction positions, not full prompt-vocabulary tensors.
        logits = output.logits[:, len(prompt_ids) - 1 :].float()
        logp = logits.log_softmax(-1)
        result = {}
        for i, label in enumerate(labels):
            target = torch.tensor(self.paths[label].ids, device=self.device)
            values = logp[i, torch.arange(len(target), device=self.device), target]
            result[label] = (values[:-1].sum(), values[-1])
        return result

    def score(self, prompt_ids, labels=None, patch_factory=None):
        labels = list(EMOTIONS if labels is None else labels)
        result, audit = {}, []
        for start in range(0, len(labels), self.batch_size):
            batch = labels[start : start + self.batch_size]
            with torch.no_grad():
                if patch_factory is None:
                    values = self.forward_paths(prompt_ids, batch)
                else:
                    with patch_factory([len(prompt_ids) - 1] * len(batch)) as patch:
                        values = self.forward_paths(prompt_ids, batch)
                    audit.extend(patch.audit)
            for label, (label_score, stop_score) in values.items():
                a, b = float(label_score), float(stop_score)
                result[label] = dict(
                    label_score=a,
                    stop_score=b,
                    total=a + b,
                    label_tokens=len(self.paths[label].label_ids),
                    token_ids=list(self.paths[label].ids),
                )
        return result, audit


def fixed_rival(scores, gold):
    return max((c for c in EMOTIONS if c != gold), key=lambda c: scores[c]["total"])


def membership_top1(scores):
    """Rank all complete label-plus-stop paths, breaking ties in canonical label order."""
    if set(scores) != set(EMOTIONS) or any(
        not isfinite(scores[label]["total"]) for label in EMOTIONS
    ):
        raise ValueError("Membership top-1 requires finite scores for all canonical emotion paths")
    return max(EMOTIONS, key=lambda label: scores[label]["total"])


def margin(scores, gold, rival):
    return scores[gold]["total"] - scores[rival]["total"]
