"""Native EXP5 execution, separate from cohort policy and experiment orchestration."""

from contextlib import nullcontext
from math import isfinite

import torch

from affect_geometry.classification.inference import Generator
from affect_geometry.classification.protocol import digest, prompt, score
from affect_geometry.evaluation.emotion_labels import EmotionLabelScorer
from affect_geometry.interventions.neuron_patching import NeuronPatch
from affect_geometry.models.mlp_neurons import NativeMLPNeurons
from affect_geometry.neurons.attribution import attribute


class NativeNeuronRunner:
    def __init__(self, config, manifest, model=None, tokenizer=None):
        if model is None:
            generator = Generator(
                manifest["model_id"],
                manifest["model_revision"],
                manifest["chat"],
                dtype=config.dtype,
                device=config.device,
                max_new_tokens=manifest["max_new_tokens"],
            )
            model, tokenizer = generator.model, generator.tokenizer
        self.model, self.tokenizer = model, tokenizer
        self.config, self.manifest = config, manifest
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.adapter = NativeMLPNeurons(model)
        self.widths = self.adapter.widths
        if manifest["chat"] and tokenizer.chat_template != manifest["chat_template"]:
            raise ValueError("Native chat template differs from EXP2")
        self.scorer = EmotionLabelScorer(
            model,
            tokenizer,
            stop_id=config.stop_token_id,
            prefix=config.label_prefix,
            budget=manifest["max_new_tokens"],
            batch_size=config.score_batch_size,
        )

    def encode(self, row, menu=None):
        payload, order = prompt(row, self.manifest["seed"], label_order=menu)
        rendered = (
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True
            )
            if self.manifest["chat"]
            else payload
        )
        ids = self.tokenizer.encode(rendered, add_special_tokens=not self.manifest["chat"])
        if menu is None:
            if digest(rendered) != row["prompt_hash"] or rendered != row["rendered_prompt"]:
                raise ValueError("EXP2 prompt reconstruction mismatch")
            if len(ids) != row["input_token_count"]:
                raise ValueError("EXP2 input token count mismatch")
        if len(ids) + self.manifest["max_new_tokens"] > getattr(
            self.model.config, "max_position_embeddings", float("inf")
        ):
            raise ValueError("Context overflow; truncation is prohibited")
        return dict(
            ids=ids,
            rendered_prompt=rendered,
            prompt_hash=digest(rendered),
            label_order=order,
            prompt_position=len(ids) - 1,
        )

    def patch(self, members, donor, alpha=0.25, offsets=None):
        return lambda positions: NeuronPatch(
            self.adapter, members, donor, positions, alpha=alpha, offsets=offsets
        )

    def scores(self, info, labels=None, patch=None):
        return self.scorer.score(info["ids"], labels, patch)

    def prompt_logits(self, info, patch=None):
        ids = torch.tensor([info["ids"]], device=self.scorer.device)
        context = patch([info["prompt_position"]]) if patch else nullcontext()
        with torch.no_grad(), context:
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        return output.logits[0, -1].detach().float().cpu().numpy()

    def generate(self, info, gold, patch=None):
        ids = torch.tensor([info["ids"]], device=self.scorer.device)
        context = patch([info["prompt_position"]]) if patch else nullcontext()
        with torch.inference_mode(), context as edit:
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.manifest["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
            )
        tokens = output[0, len(info["ids"]) :].tolist()
        stops = self.model.generation_config.eos_token_id
        if stops is None:
            stops = self.tokenizer.eos_token_id
        stops = [stops] if isinstance(stops, int) else list(stops)
        end = next((i for i, token in enumerate(tokens) if token in stops), None)
        if end is not None:
            tokens = tokens[: end + 1]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return dict(
            **score(raw, gold),
            raw_response=raw,
            generated_token_ids=tokens,
            generation_finish_reason="eos" if end is not None else "max_new_tokens",
            edit_audit=edit.audit if patch else [],
        )

    def attribution(self, info, gold, totals=None):
        return attribute(self.adapter, self.scorer, info["ids"], gold, self.config.top_k, totals)

    def coordinates(self, info, members):
        members = self.adapter.validate_members(members)
        if not members:
            return {}
        with torch.no_grad(), self.adapter.capture([info["prompt_position"]], members) as captured:
            ids = torch.tensor([info["ids"]], device=self.scorer.device)
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        result = {}
        for layer in sorted(captured):
            channels = [j for member_layer, j in members if member_layer == layer]
            result.update(
                {(layer, j): float(captured[layer][0, i]) for i, j in enumerate(channels)}
            )
        return result

    def replay(self, row, result):
        if result["raw_response"] != row["raw_response"]:
            raise ValueError(f"EXP2 generation replay mismatch: {row['sample_id']}")
        if list(result["generated_token_ids"]) != list(row["generated_token_ids"]):
            raise ValueError(f"EXP2 token replay mismatch: {row['sample_id']}")
        if result["correct"] != bool(row["correct"]):
            raise ValueError("EXP2 strict outcome replay mismatch")

    def finite_difference(self, info, gold, record):
        key = (record["layer"], record["channel"])
        dtype = next(self.model.parameters()).dtype
        # A fixed absolute step overwhelms small OLMo2 activations. Conversely, tiny
        # steps disappear in native low precision. Check a prespecified step grid,
        # selecting its most stable adjacent pair without consulting the gradient.
        low_precision = dtype != torch.float32
        scale = abs(record["activation"])
        base = max(scale, 1e-3) if low_precision else max(1e-4, 0.1 * scale)
        activation = torch.tensor(record["activation"], dtype=dtype)
        trials = []
        for factor in (0.25, 0.5, 1.0, 2.0):
            epsilon = base * factor
            endpoints = [float((activation.float() + sign * epsilon).to(dtype)) for sign in (-1, 1)]
            if endpoints[0] == endpoints[1]:
                continue
            scores = []
            for sign in (-1, 1):
                patch = self.patch([key], {}, offsets={key: sign * epsilon})
                values, _ = self.scores(info, [gold], patch)
                scores.append(values[gold]["total"])
            measured = (scores[1] - scores[0]) / (endpoints[1] - endpoints[0])
            if not isfinite(measured):
                raise ValueError("Nonfinite attribution finite difference")
            trials.append(
                dict(epsilon=epsilon, endpoints=endpoints, scores=scores, estimate=measured)
            )
        pairs = []
        for left, right in zip(trials, trials[1:]):
            gap = abs(left["estimate"] - right["estimate"]) / max(
                abs(left["estimate"]), abs(right["estimate"]), 1e-9
            )
            pairs.append((gap, left, right))
        if not pairs:
            raise ValueError("Finite-difference steps are unresolved in the native dtype")
        gap, left, right = min(pairs, key=lambda pair: pair[0])
        measured = (left["estimate"] + right["estimate"]) / 2
        expected = record["gradient"]
        floor = 0.03 if low_precision else 1e-3
        tolerance = floor + 0.25 * abs(expected)
        if (
            gap > (0.25 if low_precision else 0.1)
            or abs(measured - expected) > tolerance
            or (abs(expected) > floor and measured * expected <= 0)
        ):
            raise ValueError(f"Attribution finite difference mismatch: {expected} vs {measured}")
        return dict(
            layer=key[0],
            channel=key[1],
            dtype=str(dtype),
            trials=trials,
            selected_epsilons=[left["epsilon"], right["epsilon"]],
            relative_step_disagreement=gap,
            gradient=expected,
            finite_difference=measured,
            tolerance=tolerance,
        )
