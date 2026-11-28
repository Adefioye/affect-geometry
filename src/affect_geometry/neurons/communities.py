"""Leiden candidates, matched control groups, and deterministic training-only selection."""

import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.data.neuron_transfer import child_seed
from affect_geometry.persistence.correctness_geometry import fingerprint


def community_id(model_identity, members):
    return fingerprint(
        dict(
            model=model_identity,
            site="native_mlp_pre_down",
            members=sorted(tuple(v) for v in members),
        )
    )[:24]


def discover_communities(edges, attributions, model_identity, config):
    if edges.empty:
        return []
    try:
        import igraph as ig
        import leidenalg
    except ImportError as error:
        raise ImportError("Install the EXP5 graph extra: pip install -e '.[neurons]'") from error
    nodes = sorted(
        set(zip(edges.layer_i, edges.channel_i, strict=True))
        | set(zip(edges.layer_j, edges.channel_j, strict=True))
    )
    lookup = {v: i for i, v in enumerate(nodes)}
    pairs = [
        (lookup[(int(r.layer_i), int(r.channel_i))], lookup[(int(r.layer_j), int(r.channel_j))])
        for r in edges.itertuples()
    ]
    graph = ig.Graph(n=len(nodes), edges=pairs, directed=False)
    weights = edges.excess.astype(float).tolist()
    selected_sets = {
        sample_id: set(zip(part.layer, part.channel, strict=True))
        for sample_id, part in attributions.groupby("sample_id", sort=True)
    }
    groups = {}
    for resolution in config.resolutions:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.CPMVertexPartition,
            weights=weights,
            resolution_parameter=float(resolution),
            seed=config.seed,
            n_iterations=-1,
        )
        for cluster in partition:
            members = sorted(nodes[i] for i in cluster)
            if not config.minimum_size <= len(members) <= config.maximum_size:
                continue
            identifier = community_id(model_identity, members)
            if identifier not in groups:
                member_set = set(members)
                support = sum(len(member_set & s) >= 2 for s in selected_sets.values())
                groups[identifier] = dict(
                    community_id=identifier,
                    members=[list(v) for v in members],
                    resolutions=[],
                    discovery_support=support,
                )
            groups[identifier]["resolutions"].append(float(resolution))
    return sorted(groups.values(), key=lambda c: (-c["discovery_support"], c["community_id"]))


def control_members(community, widths, attributions, seed):
    members = {tuple(v) for v in community["members"]}
    random_members, influential = [], []
    for layer in sorted({member_layer for member_layer, _ in members}):
        excluded = {j for member_layer, j in members if member_layer == layer}
        count = len(excluded)
        choices = np.array([j for j in range(widths[layer]) if j not in excluded])
        if len(choices) < count:
            raise ValueError("Insufficient nonmember channels for matched controls")
        rng = np.random.default_rng(child_seed(seed, "control", community["community_id"], layer))
        random_members.extend((layer, int(j)) for j in rng.choice(choices, count, replace=False))
        part = attributions[attributions.layer.eq(layer)]
        # Caller supplies the full training mean table, including neurons outside sparse top-k.
        scores = part.groupby("channel").importance.sum().to_dict()
        ranked = sorted(choices.tolist(), key=lambda j: (-scores.get(j, 0), j))
        influential.extend((layer, int(j)) for j in ranked[:count])
    return dict(
        random=[list(v) for v in sorted(random_members)],
        unclustered=[list(v) for v in sorted(influential)],
    )


def select_communities(candidates, effects, tolerance, config):
    supported, summaries = {}, []
    for candidate in candidates:
        identifier = candidate["community_id"]
        rows = effects[effects.community_id.eq(identifier)] if len(effects) else effects
        labels = {}
        for label in EMOTIONS:
            part = rows[rows.emotion.eq(label)] if len(rows) else rows
            values = part.net_effect.to_numpy() if len(part) else np.array([])
            resolved = int((np.abs(values) > tolerance).sum())
            mean = float(values.mean()) if len(values) else None
            qualifies = (
                mean is not None and mean > tolerance and resolved >= config.minimum_effect_examples
            )
            summaries.append(
                dict(
                    community_id=identifier,
                    emotion=label,
                    n=len(values),
                    resolved=resolved,
                    mean_net_effect=mean,
                    qualifies=qualifies,
                )
            )
            if qualifies:
                labels[label] = mean
        if labels:
            supported[identifier] = labels
    remaining = [c for c in candidates if c["community_id"] in supported]
    shortlist, covered = [], set()
    while remaining and len(shortlist) < config.community_budget:
        remaining.sort(
            key=lambda c: (
                -len(set(supported[c["community_id"]]) - covered),
                -max(supported[c["community_id"]].values()),
                c["community_id"],
            )
        )
        candidate = remaining.pop(0)
        members = {tuple(v) for v in candidate["members"]}
        if any(
            len(members & {tuple(v) for v in other["members"]})
            / len(members | {tuple(v) for v in other["members"]})
            >= config.maximum_overlap
            for other in shortlist
        ):
            continue
        shortlist.append(dict(candidate, supported_labels=supported[candidate["community_id"]]))
        covered.update(supported[candidate["community_id"]])
    return shortlist, pd.DataFrame(
        summaries,
        columns=["community_id", "emotion", "n", "resolved", "mean_net_effect", "qualifies"],
    )
