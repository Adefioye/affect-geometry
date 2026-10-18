import torch

from affect_geometry.activations.extract import pool_hidden


def test_last_token_and_mean_pool_exclude_special_tokens() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [4.0], [100.0]]])
    attention = torch.tensor([[1, 1, 1, 1]])
    special = torch.tensor([[1, 0, 0, 1]])
    assert pool_hidden(hidden, attention, special, "last_token").item() == 4.0
    assert pool_hidden(hidden, attention, special, "mean").item() == 3.0


def test_pooling_uses_each_rows_last_content_token() -> None:
    hidden = torch.arange(12.0).reshape(2, 3, 2)
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    special = torch.zeros_like(attention)
    pooled = pool_hidden(hidden, attention, special, "last_token")
    assert torch.equal(pooled, torch.stack([hidden[0, 1], hidden[1, 2]]))
