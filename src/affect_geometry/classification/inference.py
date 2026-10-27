"""Native Hugging Face inference; no activation hooks or upstream script imports."""

from .protocol import digest, prompt


class Generator:
    def __init__(
        self, model_id, revision, chat, dtype="bfloat16", device="auto", max_new_tokens=10
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype), device_map=device
        )
        self.model.eval()
        self.chat, self.limit = chat, max_new_tokens
        if chat and not self.tokenizer.chat_template:
            raise ValueError("Native chat template is missing")
        self.template = self.tokenizer.chat_template if chat else None

    def __call__(self, row, seed):
        import torch

        payload, order = prompt(row, seed)
        rendered = (
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True
            )
            if self.chat
            else payload
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=not self.chat)
        size = inputs["input_ids"].shape[1]
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and size + self.limit > context:
            raise ValueError("Input exceeds context; truncation is prohibited")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.limit,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        ids = output[0, size:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = (
            "eos"
            if ids and ids[-1] in eos
            else "max_new_tokens"
            if len(ids) >= self.limit
            else "other_stop"
        )
        return dict(
            raw_response=self.tokenizer.decode(ids, skip_special_tokens=True),
            generated_token_ids=ids,
            rendered_prompt=rendered,
            prompt_hash=digest(rendered),
            label_order=order,
            input_token_count=size,
            was_truncated=False,
            generation_finish_reason=reason,
            generated_token_count=len(ids),
            max_new_tokens=self.limit,
        )
