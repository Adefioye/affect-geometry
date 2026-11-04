"""HF post-block, final-token interventions for Gemma2, Apertus and OLMo2."""

import torch

from affect_geometry.interventions.directional import directional_replace
from affect_geometry.models.huggingface import decoder_layers


class LastTokenPatcher:
    """Use actual decoder outputs, including the final block before final norm."""

    def __init__(self, model, tokenizer, device, max_tokens=128):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.layers = decoder_layers(model)
        self.max_tokens = max_tokens
        self.answer_ids = []
        for answer in (" Negative", " Positive"):
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Answer {answer!r} must be one token; got {ids}")
            self.answer_ids.append(ids[0])
        if len(set(self.answer_ids)) != 2:
            raise ValueError("Positive and negative answers have identical token IDs")

    def encode(self, prompt):
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        if encoded["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Prompt exceeds token budget; truncation is prohibited")
        return {
            k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")
        }

    def run(self, prompt, layer, donor=None, direction=None, full=False):
        if layer < 0 or layer >= len(self.layers):
            raise ValueError("Layer is not a zero-based decoder block index")
        tokens = self.encode(prompt)
        captured = []

        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :].detach().clone())
            if donor is None:
                return output
            replacement = donor.to(hidden.device, hidden.dtype)
            edited = hidden.clone()
            edited[:, -1, :] = (
                replacement
                if full
                else directional_replace(hidden[:, -1, :], replacement, direction)
            )
            return (edited, *output[1:]) if isinstance(output, tuple) else edited

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output = self.model(**tokens, use_cache=False)
            logits = output.logits[0, -1].float()
            delta = float(logits[self.answer_ids[1]] - logits[self.answer_ids[0]])
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Selected block was not executed exactly once")
        return delta, captured[0]
