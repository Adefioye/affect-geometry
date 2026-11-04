"""Native-space directional replacement; adapted from sentiment-manifold semantics."""

import torch


def directional_replace(recipient, donor, direction):
    """Replace one coordinate; preserve the recipient's orthogonal component."""
    if recipient.shape != donor.shape:
        raise ValueError("Donor and recipient shapes differ")
    d = torch.as_tensor(direction, device=recipient.device, dtype=torch.float32)
    if d.ndim != 1 or d.numel() != recipient.shape[-1]:
        raise ValueError("Direction must have shape [hidden_size]")
    norm = d.norm()
    if not torch.isfinite(d).all() or norm <= 1e-12:
        raise ValueError("Direction must be finite and nonzero")
    d = d / norm
    edited = recipient.float() + ((donor.float() - recipient.float()) @ d)[..., None] * d
    return edited.to(recipient.dtype)
