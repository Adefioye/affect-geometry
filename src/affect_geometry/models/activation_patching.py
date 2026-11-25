"""HF post-block, final-token interventions for Gemma2, Apertus and OLMo2."""

from inspect import signature

import torch

from affect_geometry.interventions.directional import directional_replace
from affect_geometry.models.huggingface import decoder_layers


class LastTokenPatcher:
    """Use actual decoder outputs, including the final block before final norm."""

    def __init__(self, model, tokenizer, device, max_tokens=128):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.layers = decoder_layers(model)
        self.max_tokens = max_tokens
        parameters = signature(model.forward).parameters
        self.logit_options = {}
        for keyword in ("logits_to_keep", "num_logits_to_keep"):
            if keyword in parameters:
                self.logit_options = {keyword: 1}
                break
        self.answer_ids = []
        for answer in (" Negative", " Positive"):
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Answer {answer!r} must be one token; got {ids}")
            self.answer_ids.append(ids[0])
        if len(set(self.answer_ids)) != 2:
            raise ValueError("Positive and negative answers have identical token IDs")

    def encode_batch(self, prompts):
        rows = [self.tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
        if not rows or any(not row for row in rows):
            raise ValueError("Prompt batches must contain nonempty token sequences")
        width = max(map(len, rows))
        if width > self.max_tokens:
            raise ValueError("Prompt exceeds token budget; truncation is prohibited")
        padding = [width - len(row) for row in rows]
        pad_id = self.tokenizer.pad_token_id if any(padding) else 0
        if pad_id is None:
            raise ValueError("Mixed-length prompt batches require a tokenizer padding token")
        # Left padding aligns the final prompt tokens for both patching and the LM head.
        input_ids = [[pad_id] * pad + row for row, pad in zip(rows, padding)]
        mask = [[0] * pad + [1] * len(row) for row, pad in zip(rows, padding)]
        attention_mask = torch.tensor(mask, device=self.device)
        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(dim=-1) - 1).clamp_min_(0),
        }

    def run(self, prompt, layer, donor=None, direction=None, full=False):
        scores, captured = self.run_batch(
            self.encode_batch([prompt]), layer, donor, direction, full
        )
        return scores.item(), captured

    @torch.inference_mode()
    def run_batch(self, tokens, layer, donor=None, direction=None, full=False):
        """Return one logit difference and one post-block activation per prompt."""
        if layer < 0 or layer >= len(self.layers):
            raise ValueError("Layer is not a zero-based decoder block index")
        captured = []

        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :].detach().clone())
            if donor is None:
                return output
            replacement = donor.to(hidden.device, hidden.dtype)
            if replacement.shape != captured[-1].shape:
                raise ValueError("Expected one donor activation per recipient prompt")
            edited = hidden.clone()
            edited[:, -1, :] = (
                replacement
                if full
                else directional_replace(hidden[:, -1, :], replacement, direction)
            )
            return (edited, *output[1:]) if isinstance(output, tuple) else edited

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            output = self.model(**tokens, use_cache=False, **self.logit_options)
            logits = output.logits[:, -1, self.answer_ids].float()
            delta = logits[:, 1] - logits[:, 0]
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Selected block was not executed exactly once")
        return delta, captured[0]
