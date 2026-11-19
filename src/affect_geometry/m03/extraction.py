"""Batched candidate-prefix replay shared by every layer and lens.

Only the small prediction-position residuals are retained by block hooks. Prefixes
are deduplicated within a comment, and each prefix is forwarded once for all lenses.
This version replays complete prefixes; it does not claim KV-cache/early-exit savings.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from affect_geometry.m03.configuration import Config, fingerprint
from affect_geometry.m03.paths import CandidatePaths
from affect_geometry.m03.readouts import Readouts
from affect_geometry.m03.storage import score_keys


def identity(paths: CandidatePaths, prompt_hash: str, variant: str) -> dict:
    return dict(
        prompt_hash=prompt_hash,
        variant=variant,
        token_paths_hash=fingerprint(dict(prompt=paths.prompt_ids, paths=paths.paths)),
    )


class Scorer:
    def __init__(self, model, tokenizer, readouts: Readouts, config: Config):
        self.model, self.tokenizer, self.readouts, self.config = (
            model,
            tokenizer,
            readouts,
            config,
        )
        self.keys = score_keys(readouts.plan)
        self.calls = 0
        self.native_max_error = 0.0

    @torch.inference_mode()
    def capture(self, sequences: list[tuple[int, ...]], *, native_check: bool):
        width = max(map(len, sequences))
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and width > limit:
            raise ValueError("Candidate path exceeds context; truncation is prohibited")
        device = self.model.model.embed_tokens.weight.device
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, device=device)
            mask[i, : len(seq)] = 1
        positions = mask.sum(-1) - 1
        captured = {}
        handles = []

        def hook(layer):
            def save(_module, _args, output):
                state = output[0] if isinstance(output, tuple) else output
                p = positions.to(state.device)
                captured[layer] = state[torch.arange(len(sequences), device=state.device), p]

            return save

        try:
            for layer, block in enumerate(self.readouts.blocks):
                handles.append(block.register_forward_hook(hook(layer)))
            kwargs = dict(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                use_cache=False,
            )
            if native_check:
                # Project only the columns containing a prediction position.
                columns = positions.unique(sorted=True)
                output = self.model(**kwargs, logits_to_keep=columns)
                offsets = torch.searchsorted(columns, positions).to(output.logits.device)
                native = output.logits[
                    torch.arange(len(sequences), device=output.logits.device), offsets
                ].float()
            else:
                self.model.model(**kwargs)
                native = None
            self.calls += 1
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != self.config.n_blocks:
            raise ValueError("Missing post-block captures")
        return captured, native

    @torch.inference_mode()
    def score(
        self,
        requests: list[CandidatePaths],
        *,
        diagnostic: list[bool] | None = None,
        native_check: bool = False,
        shared: bool = True,
        batch_size: int | None = None,
    ) -> list[dict]:
        started = time.perf_counter()
        batch_size = batch_size or self.config.prefix_batch_size
        diagnostic = diagnostic or [False] * len(requests)
        values = [
            dict(
                label_scores=np.zeros((len(self.keys), 28)),
                stop_scores=np.zeros((len(self.keys), 28)),
                token_counts=np.tile([len(p) - 1 for p in req.paths], (len(self.keys), 1)),
            )
            for req in requests
        ]
        jobs = []
        for i, req in enumerate(requests):
            if shared:
                node_edges = {node: [] for node in req.nodes}
                for node, token, c, stop in req.edges():
                    node_edges[node].append((token, c, stop))
                jobs.extend((i, node, edges) for node, edges in node_edges.items())
            else:
                jobs.extend((i, node, [(token, c, stop)]) for node, token, c, stop in req.edges())
        for start in range(0, len(jobs), batch_size):
            part = jobs[start : start + batch_size]
            sequences = [requests[i].prompt_ids + node for i, node, _edges in part]
            states, actual_native = self.capture(sequences, native_check=native_check)
            f_logp = None
            for k, (lens, layer) in enumerate(self.keys):
                for chunk in range(0, len(part), self.config.projection_batch_size):
                    stop_chunk = min(len(part), chunk + self.config.projection_batch_size)
                    logits = self.readouts.project(states[layer][chunk:stop_chunk], lens, layer)
                    if lens == "native" and actual_native is not None:
                        actual = actual_native[chunk:stop_chunk].to(logits.device)
                        error = (logits - actual).abs().max().item()
                        self.native_max_error = max(self.native_max_error, error)
                        torch.testing.assert_close(
                            logits,
                            actual,
                            atol=self.config.native_atol,
                            rtol=self.config.native_rtol,
                        )
                    logp = torch.log_softmax(logits, dim=-1)
                    if not torch.isfinite(logp).all():
                        raise ValueError("Nonfinite full-vocabulary distribution")
                    if lens == "native":
                        if f_logp is None:
                            f_logp = torch.empty(
                                (len(part), logp.shape[-1]), dtype=torch.float32, device=logp.device
                            )
                        f_logp[chunk:stop_chunk] = logp
                    for j in range(chunk, stop_chunk):
                        i, node, edges = part[j]
                        row_logp = logp[j - chunk]
                        tokens = [edge[0] for edge in edges]
                        selected = row_logp[tokens].double().cpu().numpy()
                        for (_, label, is_stop), value in zip(edges, selected, strict=True):
                            field = "stop_scores" if is_stop else "label_scores"
                            values[i][field][k, label] += value
                        if diagnostic[i] and not node:
                            f = f_logp[j].to(row_logp.device)
                            values[i].setdefault("kl_native_to_lens", np.zeros(len(self.keys)))[
                                k
                            ] = (f.exp() * (f - row_logp)).sum().item()
                            if self.config.retain_diagnostic_distributions:
                                values[i].setdefault(
                                    "diagnostic_logp",
                                    np.zeros((len(self.keys), len(row_logp)), dtype=np.float16),
                                )[k] = row_logp.cpu().numpy().astype(np.float16)
            del states, f_logp
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for i, result in enumerate(values):
            result["prefix_count"] = np.asarray(sum(job[0] == i for job in jobs))
            result["extraction_seconds"] = np.asarray(elapsed / len(requests))
        return values

    def technical_checks(self, requests: list[CandidatePaths]) -> dict:
        """Compare shared-prefix, independent-path, and single-example execution."""
        subset = requests[:2]
        batched = self.score(subset, native_check=True)
        independent = self.score([subset[0]], native_check=True, shared=False, batch_size=1)[0]
        separate = [self.score([req], native_check=True, batch_size=1)[0] for req in subset]
        for field in ("label_scores", "stop_scores"):
            np.testing.assert_allclose(
                batched[0][field], independent[field], atol=self.config.score_atol, rtol=0
            )
            for one, many in zip(separate, batched, strict=True):
                np.testing.assert_allclose(
                    one[field], many[field], atol=self.config.score_atol, rtol=0
                )
        return dict(
            native_max_abs_logit_error=self.native_max_error,
            native_atol=self.config.native_atol,
            native_rtol=self.config.native_rtol,
            candidate_score_atol=self.config.score_atol,
            shared_prefix_equivalence=True,
            individual_batch_equivalence=True,
            raw_final_logit_lens_native_equivalence=True,
            checked_sample_count=len(subset),
        )
