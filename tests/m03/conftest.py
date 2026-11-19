from dataclasses import replace

import pytest
import torch

from affect_geometry.m03.configuration import Config
from affect_geometry.m03.smoke import tiny_bundle


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config():
    return replace(
        Config(),
        model_id="synthetic/tiny-gemma2",
        device="cpu",
        dtype="float32",
        n_blocks=3,
        jacobian_target_layer=2,
        bootstrap_replicates=40,
        native_atol=1e-5,
        native_rtol=1e-5,
        score_atol=1e-5,
    )


@pytest.fixture
def bundle(tiny_config):
    return tiny_bundle(tiny_config)
