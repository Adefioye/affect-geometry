from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def create(cls, root: str | Path) -> RunLayout:
        layout = cls(Path(root))
        for path in (
            layout.dataset,
            layout.activations,
            layout.checkpoints,
            layout.rdms,
            layout.embeddings,
            layout.analyses,
            layout.plot_data,
            layout.plots,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def activations(self) -> Path:
        return self.root / "activations"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def rdms(self) -> Path:
        return self.root / "rdms"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def analyses(self) -> Path:
        return self.root / "analyses"

    @property
    def plot_data(self) -> Path:
        return self.root / "plot_data"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    def write_json(self, relative: str | Path, value: dict[str, Any]) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path


def safe_slug(value: str) -> str:
    return value.lower().replace("/", "--").replace(" ", "_")
