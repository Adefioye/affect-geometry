"""True-gradient native activation attribution, with signed sparse top-k records."""

import torch


def attribute(adapter, scorer, prompt_ids, gold, top_k=5, totals=None):
    with torch.enable_grad(), adapter.gradient_anchor(), adapter.capture() as captured:
        parts = scorer.forward_paths(prompt_ids, [gold])[gold]
        loss = parts[0] + parts[1]
        tensors = tuple(captured[layer] for layer in sorted(captured))
        if not tensors or not loss.requires_grad:
            raise RuntimeError("Attribution is disconnected from the native loss path")
        derivatives = torch.autograd.grad(loss, tensors)
        records = []
        position = len(prompt_ids) - 1
        for layer, tensor, derivative in zip(sorted(captured), tensors, derivatives, strict=True):
            a = tensor[0, position].detach().float()
            g = derivative[0, position].detach().float()
            signed = a * g
            if not torch.isfinite(signed).all() or not torch.isfinite(g).all():
                raise ValueError("Nonfinite attribution")
            if totals is not None:
                totals[layer] = signed.abs().cpu().numpy()
            order = torch.argsort(signed.abs(), descending=True, stable=True)[:top_k]
            for rank, channel in enumerate(order.tolist()):
                records.append(
                    dict(
                        layer=layer,
                        channel=channel,
                        activation=float(a[channel]),
                        gradient=float(g[channel]),
                        signed_contribution=float(signed[channel]),
                        importance=float(signed[channel].abs()),
                        rank=rank,
                    )
                )
    captured.clear()
    return records
