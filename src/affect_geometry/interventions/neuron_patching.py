"""Sparse, request-scoped MLP edits; explicit first-forward state protects decode steps."""

import torch


class NeuronPatch:
    def __init__(self, adapter, members, donor, positions, alpha=0.25, offsets=None):
        self.adapter = adapter
        self.members = adapter.validate_members(members)
        self.donor = donor
        self.positions = list(positions)
        self.alpha = alpha
        self.offsets = offsets
        self.handles = []
        self.forward_count = 0
        self.edit_this_forward = False
        self.audit = []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("A patch request cannot be entered twice")
        self.forward_count, self.edit_this_forward, self.audit = 0, False, []

        def start_forward(_module, _args):
            self.edit_this_forward = self.forward_count == 0
            self.forward_count += 1

        self.handles.append(self.adapter.model.register_forward_pre_hook(start_forward))
        grouped = {}
        for layer, channel in self.members:
            grouped.setdefault(layer, []).append(channel)
        try:
            for layer, channels in grouped.items():

                def edit(_module, args, layer=layer, channels=channels):
                    if not self.edit_this_forward:
                        return None
                    original = args[0]
                    if original.shape[0] != len(self.positions):
                        raise ValueError("Patch request batch/position mismatch")
                    if min(self.positions) < 0 or max(self.positions) >= original.shape[1]:
                        raise ValueError("Saved prompt position is outside the forward tensor")
                    rows = torch.arange(original.shape[0], device=original.device)[:, None]
                    pos = torch.as_tensor(self.positions, device=original.device)[:, None]
                    cols = torch.as_tensor(channels, device=original.device)[None, :]
                    current = original[rows, pos, cols].float()
                    if self.offsets is not None:
                        delta = torch.tensor(
                            [self.offsets.get((layer, j), 0.0) for j in channels],
                            device=original.device,
                        )
                        replacement = current + delta
                    else:
                        target = torch.tensor(
                            [self.donor[(layer, j)] for j in channels], device=original.device
                        )
                        replacement = (1 - self.alpha) * current + self.alpha * target
                    edited = original.clone()
                    native = replacement.to(original.dtype)
                    edited[rows, pos, cols] = native
                    norms = (native.float() - current).norm(dim=-1).detach().cpu().tolist()
                    self.audit.extend(
                        dict(
                            layer=layer,
                            batch_row=i,
                            position=self.positions[i],
                            count=len(channels),
                            norm=norm,
                            forward=self.forward_count,
                        )
                        for i, norm in enumerate(norms)
                    )
                    return (edited, *args[1:])

                self.handles.append(self.adapter.modules[layer].register_forward_pre_hook(edit))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.edit_this_forward = False


def zero_values(members):
    return {tuple(member): 0.0 for member in members}
