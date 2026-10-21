from __future__ import annotations

import numpy as np
import pandas as pd

from affect_geometry.analysis.agreement import embedding_procrustes_r2


def cross_layer_similarity(
    coordinates_by_layer: dict[int, np.ndarray],
) -> pd.DataFrame:
    layers = sorted(coordinates_by_layer)
    rows = []
    for left_index, left in enumerate(layers):
        for right in layers[left_index + 1 :]:
            rows.append(
                {
                    "layer_i": left,
                    "layer_j": right,
                    "procrustes_r2": embedding_procrustes_r2(
                        coordinates_by_layer[left], coordinates_by_layer[right]
                    ),
                }
            )
    return pd.DataFrame(rows)
