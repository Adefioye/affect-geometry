"""Small centroid Rips clique filtration, exact Z2 persistence through H1.

Triangles are included to resolve H1 deaths. No metric-space claim is made for cosine input.
"""

from itertools import combinations

import numpy as np
import pandas as pd


def persistence(distances, bound=2.0, grid_size=101):
    if not np.isfinite(bound) or bound <= 0 or grid_size < 2:
        raise ValueError("Invalid filtration settings")
    d = np.asarray(distances)
    simplices = [(0.0, (i,)) for i in range(len(d))]
    for size in (2, 3):
        for vertices in combinations(range(len(d)), size):
            birth = max(float(d[i, j]) for i, j in combinations(vertices, 2))
            if birth <= bound:
                simplices.append((birth, vertices))
    simplices.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    positions = {s: i for i, (_, s) in enumerate(simplices)}
    reduced, pivot_owner, births, paired = {}, {}, set(), set()
    intervals = []
    for j, (time, vertices) in enumerate(simplices):
        column = (
            {positions[face] for face in combinations(vertices, len(vertices) - 1)}
            if len(vertices) > 1
            else set()
        )
        while column and max(column) in pivot_owner:
            column ^= reduced[pivot_owner[max(column)]]
        reduced[j] = column
        if not column:
            births.add(j)
        else:
            pivot = max(column)
            pivot_owner[pivot] = j
            paired.add(pivot)
            born, simplex = simplices[pivot]
            if len(simplex) <= 2:
                intervals.append(
                    dict(
                        dimension=len(simplex) - 1,
                        birth=born,
                        death=time,
                        birth_simplex=",".join(map(str, simplex)),
                        death_simplex=",".join(map(str, vertices)),
                    )
                )
    for j in sorted(births - paired):
        born, simplex = simplices[j]
        if len(simplex) <= 2:
            intervals.append(
                dict(
                    dimension=len(simplex) - 1,
                    birth=born,
                    death=np.inf,
                    birth_simplex=",".join(map(str, simplex)),
                    death_simplex="",
                )
            )
    diagrams = pd.DataFrame(intervals)
    betti = []
    for time in np.linspace(0, bound, grid_size):
        for dim in (0, 1):
            count = (
                (diagrams.dimension == dim) & (diagrams.birth <= time) & (diagrams.death > time)
            ).sum()
            betti.append(dict(filtration=time, dimension=dim, betti=int(count)))
    return diagrams, pd.DataFrame(betti)
