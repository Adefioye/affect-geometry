"""Native Hugging Face inference; no activation hooks or upstream script imports."""

from .protocol import digest, prompt


class Generator:
    def __init__(
        self,
        model_id,
        revision,
        chat,
        dtype="bfloat16",
        device="auto",
        max_new_tokens=10,
        token=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .environment import load_hf_token

        token = token or load_hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            token=token,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Batching requires a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.chat, self.limit = chat, max_new_tokens
        if chat and not self.tokenizer.chat_template:
            raise ValueError("Native chat template is missing")
        self.template = self.tokenizer.chat_template if chat else None

    def __call__(self, row, seed):
        return self.batch([row], seed)[0]

    def batch(self, rows, seed):
        import torch

        prompts = [prompt(row, seed) for row in rows]
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True
            )
            if self.chat
            else payload
            for payload, _ in prompts
        ]
        inputs = self.tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=not self.chat
        )
        width = inputs["input_ids"].shape[1]
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and width + self.limit > context:
            raise ValueError("Input exceeds context; truncation is prohibited")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.limit,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        results = []
        for index, text in enumerate(rendered):
            ids = output[index, width:].tolist()
            # Finished sequences are padded while other batch members continue.
            end = next((i for i, token in enumerate(ids) if token in eos), None)
            if end is not None:
                ids = ids[: end + 1]
            reason = (
                "eos"
                if end is not None
                else "max_new_tokens"
                if len(ids) >= self.limit
                else "other_stop"
            )
            results.append(
                dict(
                    raw_response=self.tokenizer.decode(ids, skip_special_tokens=True),
                    generated_token_ids=ids,
                    rendered_prompt=text,
                    prompt_hash=digest(text),
                    label_order=prompts[index][1],
                    input_token_count=lengths[index],
                    was_truncated=False,
                    generation_finish_reason=reason,
                    generated_token_count=len(ids),
                    max_new_tokens=self.limit,
                    inference_batch_size=len(rows),
                )
            )
        return results
