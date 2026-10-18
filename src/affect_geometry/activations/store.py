from __future__ import annotations

from pathlib import Path

import numpy as np

from affect_geometry.logging import progress


class ActivationStore:
    """One memory-mapped matrix per pooling and layer."""

    def __init__(
        self,
        root: Path,
        n_samples: int,
        n_layers: int,
        hidden_size: int,
        poolings: list[str],
        dtype: str = "float16",
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.arrays: dict[tuple[str, int], np.memmap] = {}
        for pooling in progress(
            poolings,
            description="Creating activation stores",
            unit="pooling",
            leave=False,
        ):
            directory = root / pooling
            directory.mkdir(parents=True, exist_ok=True)
            for layer in progress(
                range(n_layers),
                description=f"Allocating {pooling} layers",
                unit="layer",
                leave=False,
            ):
                path = directory / f"layer_{layer:03d}.npy"
                self.arrays[(pooling, layer)] = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=dtype,
                    shape=(n_samples, hidden_size),
                )

    def write(self, pooling: str, layer: int, start: int, values: np.ndarray) -> None:
        self.arrays[(pooling, layer)][start : start + len(values)] = values

    def flush(self) -> None:
        for array in progress(
            self.arrays.values(),
            description="Flushing activation stores",
            total=len(self.arrays),
            unit="layer",
            leave=False,
        ):
            array.flush()


def load_layer(root: Path, pooling: str, layer: int, mmap_mode: str = "r") -> np.ndarray:
    return np.load(root / pooling / f"layer_{layer:03d}.npy", mmap_mode=mmap_mode)
