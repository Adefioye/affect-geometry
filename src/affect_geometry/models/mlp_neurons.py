"""Architecture-checked access to native post-gating, pre-down-projection channels."""

from contextlib import contextmanager

import torch

from affect_geometry.models.huggingface import decoder_layers


class NativeMLPNeurons:
    def __init__(self, model):
        if model.config.model_type not in {"olmo2", "llama"}:
            raise ValueError(f"No validated EXP5 adapter for {model.config.model_type}")
        self.model = model
        self.modules = {}
        self.widths = {}
        for index, block in enumerate(decoder_layers(model)):
            module = getattr(getattr(block, "mlp", None), "down_proj", None)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"Block {index} lacks an unfused native MLP down projection")
            self.modules[index] = module
            self.widths[index] = module.in_features

    def validate_members(self, members):
        normalized = sorted({(int(layer), int(channel)) for layer, channel in members})
        for layer, channel in normalized:
            if layer not in self.widths or not 0 <= channel < self.widths[layer]:
                raise ValueError(f"Invalid native neuron {(layer, channel)}")
        return normalized

    @contextmanager
    def capture(self, positions=None, members=None):
        """Capture graph tensors, or copy just specified prompt coordinates for clean caches."""
        captured, handles = {}, []
        grouped = None
        if members is not None:
            grouped = {}
            for layer, channel in self.validate_members(members):
                grouped.setdefault(layer, []).append(channel)
        try:
            for layer, module in self.modules.items():
                if grouped is not None and layer not in grouped:
                    continue

                def hook(_module, args, layer=layer):
                    value = args[0]
                    if value.ndim != 3 or value.shape[-1] != self.widths[layer]:
                        raise ValueError("Unsupported down-projection tensor layout")
                    if grouped is None:
                        captured[layer] = value
                    else:
                        rows = torch.arange(value.shape[0], device=value.device)
                        pos = torch.as_tensor(positions, device=value.device)
                        state = value[rows, pos]
                        captured[layer] = state[:, grouped[layer]].detach().float().cpu().clone()

                handles.append(module.register_forward_pre_hook(hook))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def gradient_anchor(self):
        def anchor(_module, _args, output):
            return output if output.requires_grad else output.detach().requires_grad_(True)

        handle = self.model.get_input_embeddings().register_forward_hook(anchor)
        try:
            yield
        finally:
            handle.remove()
