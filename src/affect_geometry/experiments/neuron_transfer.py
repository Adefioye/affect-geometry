"""Resumable EXP5 stages. Test outcomes never enter discovery or training selection."""

import importlib.metadata
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.data.classification_source import read_classification
from affect_geometry.data.neuron_transfer import DonorRetriever, build_cohorts
from affect_geometry.evaluation.emotion_labels import fixed_rival, margin
from affect_geometry.evaluation.neuron_transfer import NativeNeuronRunner
from affect_geometry.interventions.neuron_patching import zero_values
from affect_geometry.logging import get_logger, progress
from affect_geometry.neurons.communities import (
    control_members,
    discover_communities,
    select_communities,
)
from affect_geometry.neurons.graph import census, incidence
from affect_geometry.persistence.correctness_geometry import checksum, fingerprint
from affect_geometry.persistence.neuron_transfer import NeuronStore, RecordLedger

STAGES = ("prepare", "smoke", "attribute", "graph", "select", "freeze", "test", "report")
DEPENDENCIES = dict(
    prepare=(),
    smoke=("prepare",),
    attribute=("smoke",),
    graph=("attribute",),
    select=("graph", "smoke"),
    freeze=("select",),
    test=("freeze",),
    report=("freeze",),
)


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    # Include shared source readers, inference, and persistence as well as EXP5 modules.
    return {str(p.relative_to(root)): checksum(p) for p in sorted(root.rglob("*.py"))}


def _records(frame):
    return frame.to_dict("records")


def _score_delta(before, after, gold, rival):
    return dict(
        margin_effect=margin(before, gold, rival) - margin(after, gold, rival),
        gold_label_effect=before[gold]["label_score"] - after[gold]["label_score"],
        gold_stop_effect=before[gold]["stop_score"] - after[gold]["stop_score"],
    )


class NeuronTransferExperiment:
    def __init__(self, config, runner_factory=NativeNeuronRunner):
        self.config = config
        self.source, self.source_manifest, audit = read_classification(
            config.classification_dir, config.model_id, allow_partial=config.allow_partial_source
        )
        for key in ("generated_token_ids", "label_order", "raw_response"):
            if key not in self.source:
                raise ValueError(f"EXP5 source missing {key}")
        if not config.allow_partial_source and not audit["full_population"]:
            raise ValueError("The scientific run requires the full EXP2 source")
        if self.source_manifest.get("max_new_tokens") != 10:
            raise ValueError("EXP5 requires EXP2's 10-token generation budget")
        for key in ("model_revision", "dataset_revision"):
            if not re.fullmatch(r"[a-fA-F0-9]{40}", self.source_manifest[key]):
                raise ValueError(f"EXP5 requires an immutable commit SHA for {key}")
        if self.source_manifest.get("dtype", config.dtype) != config.dtype:
            raise ValueError("Model dtype differs from the saved EXP2 run")
        software = {}
        for name in (
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "pyarrow",
            "igraph",
            "leidenalg",
        ):
            try:
                software[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                software[name] = None
        metadata = dict(
            experiment="exp5_neuron_transfer_v1",
            config=config.to_dict(),
            source_manifest=self.source_manifest,
            source_audit=audit,
            code=code_fingerprint(),
            software=software,
            inference_scope="exploratory"
            if config.allow_partial_source or config.test_previously_inspected
            else "prespecified_oracle_diagnostic",
        )
        self.store = NeuronStore(config.root, metadata)
        self.factory, self._runner = runner_factory, None

    @property
    def runner(self):
        if self._runner is None:
            self._runner = self.factory(self.config, self.source_manifest)
        return self._runner

    def provenance(self, stage):
        previous = {}
        for dependency in DEPENDENCIES[stage]:
            payload = self.provenance(dependency)
            self.store.require(dependency, payload)
            previous[dependency] = checksum(self.store.root / f"checkpoints/{dependency}.json")
        if stage == "report" and self.store.read_json("shortlist.json")["communities"]:
            self.store.require("test", self.provenance("test"))
            previous["test"] = checksum(self.store.root / "checkpoints/test.json")
        return dict(manifest=fingerprint(self.store.manifest), stage=stage, predecessors=previous)

    def run(self, stage="all"):
        for name in STAGES if stage == "all" else (stage,):
            provenance = self.provenance(name)
            if self.store.valid(name, provenance):
                get_logger().info("exp5.reuse stage=%s", name)
                continue
            if name == "test" and not self.store.read_json("shortlist.json")["communities"]:
                get_logger().info("exp5.test.skipped reason=no_supported_communities")
                continue
            started = time.perf_counter()
            paths = getattr(self, name)()
            self.store.complete(name, provenance, paths)
            get_logger().info(
                "exp5.complete stage=%s seconds=%.3f", name, time.perf_counter() - started
            )
        if stage in {"all", "report"}:
            self.store.finish()
        return self.store.root

    def frame(self, name):
        return pd.read_parquet(self.store.root / name)

    def ledger(self, name):
        return RecordLedger(self.store.root / name, self.store.manifest)

    def prepare(self):
        discovery, selection, test, audit, alternative = build_cohorts(self.source, self.config)
        first = discovery.sort_values("input_token_count").groupby("emotion", sort=True).head(1)
        rest = discovery[~discovery.sample_id.isin(first.sample_id)].sort_values(
            "input_token_count"
        )
        smoke = pd.concat([first, rest]).head(self.config.smoke_size)
        paths = [
            self.store.parquet(name, frame)
            for name, frame in (
                ("discovery.parquet", discovery),
                ("selection.parquet", selection),
                ("test.parquet", test),
                ("smoke.parquet", smoke),
                ("duplicate_audit.parquet", audit),
            )
        ]
        roles = pd.concat(
            [
                frame[["sample_id", "emotion", "duplicate_group"]].assign(role=role)
                for frame, role in (
                    (discovery, "discovery_donor"),
                    (selection, "training_selection"),
                    (test, "test"),
                    (smoke, "smoke"),
                )
            ],
            ignore_index=True,
        )
        paths.extend(
            [
                self.store.csv("cohorts.csv", roles),
                self.store.json("alternative_labels.json", alternative),
            ]
        )
        support = []
        for label in EMOTIONS:
            support.append(
                dict(
                    emotion=label,
                    discovery=int(discovery.emotion.eq(label).sum()),
                    selection=int(selection.emotion.eq(label).sum()),
                    test=int(test.emotion.eq(label).sum()),
                )
            )
        paths.append(self.store.csv("cohort_support.csv", pd.DataFrame(support)))
        return paths

    def smoke(self):
        import torch

        rows = self.frame("smoke.parquet")
        if rows.empty:
            return [self.store.json("smoke_profile.json", dict(status="no_discovery_support"))]
        started = time.perf_counter()
        runner = self.runner
        load_seconds = time.perf_counter() - started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        drift, logit_drift, checks, timings = 0.0, 0.0, [], []
        components = {}

        def measured(name, action):
            before = time.perf_counter()
            value = action()
            components.setdefault(name, []).append(time.perf_counter() - before)
            return value

        retriever = DonorRetriever(self.frame("discovery.parquet"), self.config.duplicate_threshold)
        donor_profile = dict(status="no_eligible_training_donor")
        members = [(layer, 0) for layer in runner.widths]
        for index, row in enumerate(
            progress(_records(rows), description="Checking EXP5 native access")
        ):
            info = runner.encode(row)
            t0 = time.perf_counter()
            generated = measured("generation", lambda: runner.generate(info, row["emotion"]))
            runner.replay(row, generated)
            scores, _ = measured("baseline_28_path_scoring", lambda: runner.scores(info))
            logits = runner.prompt_logits(info)
            rival = fixed_rival(scores, row["emotion"])
            coordinates = measured(
                "donor_coordinate_capture", lambda: runner.coordinates(info, members)
            )
            for alpha in (0.0, 1.0):
                patch = runner.patch(members, coordinates, alpha)
                edited = runner.generate(info, row["emotion"], patch)
                if edited["generated_token_ids"] != generated["generated_token_ids"]:
                    raise ValueError("No-op/self-donor changed generation tokens")
                edited_logits = runner.prompt_logits(info, patch)
                logit_drift = max(logit_drift, float(np.abs(edited_logits - logits).max()))
                values, _ = measured(
                    "edited_two_path_scoring",
                    lambda: runner.scores(info, [row["emotion"], rival], patch),
                )
                drift = max(
                    drift,
                    *(
                        abs(values[c]["total"] - scores[c]["total"])
                        for c in (row["emotion"], rival)
                    ),
                )
            if index == 0:
                attribution = measured(
                    "attribution", lambda: runner.attribution(info, row["emotion"])
                )
                checks.append(
                    measured(
                        "finite_difference",
                        lambda: runner.finite_difference(
                            info, row["emotion"], max(attribution, key=lambda r: r["importance"])
                        ),
                    )
                )
            if donor_profile["status"] == "no_eligible_training_donor":
                ranked, _ = retriever.ranked(row, row["emotion"])
                if ranked:
                    donor = ranked[0]["donor"]
                    donor_info = runner.encode(donor, info["label_order"])
                    verified = measured(
                        "matched_rendering_donor_verification",
                        lambda: runner.generate(donor_info, donor["emotion"]),
                    )
                    donor_profile = dict(
                        status="profiled", donor_id=donor["sample_id"], correct=verified["correct"]
                    )
            timings.append(time.perf_counter() - t0)
        if logit_drift > (1e-4 if self.config.dtype == "float32" else 1e-2):
            raise ValueError("No-op/self-donor changed prompt logits beyond numerical tolerance")
        profile = dict(
            status="passed",
            sample_ids=rows.sample_id.tolist(),
            max_noop_score_drift=drift,
            max_noop_logit_drift=logit_drift,
            score_tolerance=max(1e-5, 5 * drift),
            finite_differences=checks,
            seconds=time.perf_counter() - started,
            model_load_seconds=load_seconds,
            seconds_per_checked_comment=timings,
            component_seconds=components,
            training_donor_profile=donor_profile,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        )
        return [
            self.store.json("smoke_profile.json", profile),
            self.store.json("label_paths.json", runner.scorer.metadata()),
            self.store.json(
                "architecture.json", dict(widths=runner.widths, site="native_mlp_pre_down")
            ),
        ]

    def attribute(self):
        frame = self.frame("discovery.parquet")
        path = self.store.root / "attribution/records.sqlite"
        records = []
        with self.ledger("attribution/records.sqlite") as ledger:
            for row in progress(
                _records(frame), description="Attributing successful training comments"
            ):
                if ledger.get(row["sample_id"]) is None:
                    totals = {}
                    info = self.runner.encode(row)
                    values = self.runner.attribution(info, row["emotion"], totals)
                    ledger.put_attribution(
                        row["sample_id"],
                        dict(
                            sample_id=row["sample_id"], prompt_hash=info["prompt_hash"], rows=values
                        ),
                        totals,
                    )
            for record in ledger.rows():
                records.extend(dict(sample_id=record["sample_id"], **row) for row in record["rows"])
            means = ledger.importance_means()
        table = pd.DataFrame(
            records,
            columns=[
                "sample_id",
                "layer",
                "channel",
                "activation",
                "gradient",
                "signed_contribution",
                "importance",
                "rank",
            ],
        )
        return [
            path,
            self.store.parquet("attribution.parquet", table),
            self.store.parquet("importance_means.parquet", means),
        ]

    def graph(self):
        attrs = self.frame("attribution.parquet")
        if attrs.empty:
            edges, metadata, candidates = pd.DataFrame(), dict(status="empty_discovery"), []
        else:
            widths = {
                int(k): v for k, v in self.store.read_json("architecture.json")["widths"].items()
            }
            matrices = incidence(attrs, self.frame("discovery.parquet").sample_id.tolist(), widths)
            tile_provenance = self.provenance("graph")

            def cached_tile(a, b):
                stage = f"graph_tile_{a}_{b}"
                return (
                    self.frame(f"graph_tiles/{a}_{b}.parquet")
                    if self.store.valid(stage, tile_provenance)
                    else None
                )

            def save_tile(a, b, frame):
                path = self.store.parquet(f"graph_tiles/{a}_{b}.parquet", frame)
                self.store.complete(f"graph_tile_{a}_{b}", tile_provenance, [path])

            edges, metadata = census(matrices, self.config, save_tile, cached_tile)
            identity = {key: self.source_manifest[key] for key in ("model_id", "model_revision")}
            candidates = discover_communities(edges, attrs, identity, self.config)
        return [
            self.store.parquet("graph_edges.parquet", edges),
            self.store.json("graph_null_manifest.json", metadata),
            self.store.json("communities.json", candidates),
        ]

    def select(self):
        candidates = self.store.read_json("communities.json")[: self.config.candidate_budget]
        rows = self.frame("selection.parquet")
        effects, controls = [], {}
        widths = (
            {int(k): v for k, v in self.store.read_json("architecture.json")["widths"].items()}
            if candidates
            else {}
        )
        means = self.frame("importance_means.parquet")
        with self.ledger("selection/records.sqlite") as ledger:
            for candidate in progress(
                candidates, description="Testing training community ablations"
            ):
                cid = candidate["community_id"]
                controls[cid] = control_members(candidate, widths, means, self.config.seed)
                for row in _records(rows):
                    baseline_key = "baseline:" + row["sample_id"]
                    baseline = ledger.get(baseline_key)
                    if baseline is None:
                        info = self.runner.encode(row)
                        scores, _ = self.runner.scores(info)
                        baseline = dict(scores=scores, rival=fixed_rival(scores, row["emotion"]))
                        ledger.put(baseline_key, baseline)
                    key = cid + ":" + row["sample_id"]
                    record = ledger.get(key)
                    if record is None:
                        info = self.runner.encode(row)
                        gold, rival = row["emotion"], baseline["rival"]
                        deltas = {}
                        for name, members in (
                            ("community", candidate["members"]),
                            ("random", controls[cid]["random"]),
                        ):
                            patch = self.runner.patch(members, zero_values(members), 1.0)
                            scores, _ = self.runner.scores(info, [gold, rival], patch)
                            deltas[name] = _score_delta(baseline["scores"], scores, gold, rival)
                        record = dict(
                            community_id=cid,
                            sample_id=row["sample_id"],
                            emotion=gold,
                            net_effect=deltas["community"]["margin_effect"]
                            - deltas["random"]["margin_effect"],
                            effects=deltas,
                        )
                        ledger.put(key, record)
                    effects.append(record)
        effect_frame = pd.DataFrame(
            effects, columns=["community_id", "sample_id", "emotion", "net_effect", "effects"]
        )
        tolerance = self.store.read_json("smoke_profile.json").get("score_tolerance", 1e-5)
        shortlist, support = select_communities(candidates, effect_frame, tolerance, self.config)
        return [
            self.store.root / "selection/records.sqlite",
            self.store.parquet("community_effects.parquet", effect_frame),
            self.store.csv("community_support.csv", support),
            self.store.json("controls.json", controls),
            self.store.json(
                "shortlist.json",
                dict(
                    communities=shortlist,
                    primary=shortlist[0]["community_id"] if shortlist else None,
                    tolerance=tolerance,
                    status="selected" if shortlist else "no_supported_communities",
                ),
            ),
        ]

    def freeze(self):
        shortlist = self.store.read_json("shortlist.json")
        bank = self.frame("discovery.parquet")
        retriever = DonorRetriever(bank, self.config.duplicate_threshold)
        vocabulary = getattr(retriever.vectorizer, "vocabulary_", {})
        payload = dict(
            protocol="exp5_frozen_v1",
            config=self.config.to_dict(),
            shortlist=shortlist,
            controls=self.store.read_json("controls.json"),
            donor_ids=bank.sample_id.tolist(),
            test_ids=self.frame("test.parquet").sample_id.tolist(),
            alternative_labels=self.store.read_json("alternative_labels.json"),
            retriever=dict(settings=repr(retriever.vectorizer), vocabulary=vocabulary),
        )
        profile = self.store.read_json("smoke_profile.json")
        components = profile.get("component_seconds", {})
        test = self.frame("test.parquet")
        size, groups = len(test), len(shortlist["communities"])
        correct = int(test.correct.sum())
        workload = dict(
            test_rows=size,
            frozen_communities=groups,
            baseline_generation_conditions=size if groups else 0,
            maximum_edited_generation_conditions=4 * size * groups,
            heldout_ablation_conditions=2 * correct * groups,
            maximum_donor_verification_attempts=2 * self.config.donor_attempts * size * groups,
            baseline_path_scores=28 * size if groups else 0,
            maximum_edited_path_scores=8 * size * groups,
            heldout_ablation_path_scores=4 * correct * groups,
            caveat=(
                "Bounds precede donor-cache/abstention reuse. "
                "Smoke timing is not a GPU-hour guarantee."
            ),
        )
        if groups and components:
            generation = float(np.mean(components["generation"]))
            scoring = float(np.mean(components["baseline_28_path_scoring"]))
            edited = float(np.mean(components["edited_two_path_scoring"]))
            workload["smoke_linear_test_hours_excluding_donor_work"] = (
                size * (generation + scoring)
                + 4 * size * groups * (generation + edited)
                + 2 * correct * groups * edited
            ) / 3600
        return [
            self.store.json("frozen.json", dict(payload=payload, fingerprint=fingerprint(payload))),
            self.store.json("compute_budget.json", workload),
        ]

    def _donor(self, recipient, community, label, relevance, retriever, ledger, required_members):
        ranked, excluded = retriever.ranked(recipient, label)
        attempts = []
        menu = self.runner.encode(recipient)["label_order"]
        for candidate in ranked[: self.config.donor_attempts]:
            row = candidate["donor"]
            info = self.runner.encode(row, menu)
            key = fingerprint(
                dict(donor=row["sample_id"], prompt=info["prompt_hash"], members=required_members)
            )
            cached = ledger.get(key)
            if cached is None:
                generated = self.runner.generate(info, row["emotion"])
                cached = dict(
                    donor_id=row["sample_id"],
                    prompt_hash=info["prompt_hash"],
                    generation=generated,
                    attribution=[],
                )
                if generated["correct"]:
                    cached["attribution"] = self.runner.attribution(info, row["emotion"])
                    coordinates = self.runner.coordinates(info, required_members)
                    ordered = sorted(coordinates)
                    path = self.store.arrays(
                        f"donor_activations/{key}.npz",
                        members=np.asarray(ordered, dtype=np.int64),
                        values=np.array([coordinates[v] for v in ordered], dtype=np.float32),
                    )
                    cached["coordinates_path"] = str(path.relative_to(self.store.root))
                    cached["coordinates_sha256"] = checksum(path)
                ledger.put(key, cached)
            active = {(r["layer"], r["channel"]) for r in cached["attribution"]}
            relevant = bool(active & {tuple(v) for v in community["members"]})
            attempts.append(
                dict(
                    donor_id=row["sample_id"],
                    similarity=candidate["similarity"],
                    strict_correct=cached["generation"]["correct"],
                    relevant=relevant,
                )
            )
            if cached["generation"]["correct"] and (relevant or not relevance):
                path = self.store.root / cached["coordinates_path"]
                if checksum(path) != cached["coordinates_sha256"]:
                    raise ValueError("Corrupt donor coordinate cache")
                with np.load(path, allow_pickle=False) as arrays:
                    coords = {
                        tuple(map(int, member)): float(value)
                        for member, value in zip(arrays["members"], arrays["values"], strict=True)
                    }
                return dict(
                    donor_id=row["sample_id"],
                    donor_prompt_hash=info["prompt_hash"],
                    similarity=candidate["similarity"],
                    attempts=attempts,
                    exclusions=excluded,
                    abstention=None,
                ), coords
        reason = (
            "no_eligible_donor"
            if not ranked
            else "no_verified_relevant_donor"
            if relevance
            else "no_verified_donor"
        )
        return dict(
            donor_id=None,
            donor_prompt_hash=None,
            similarity=None,
            attempts=attempts,
            exclusions=excluded,
            abstention=reason,
        ), None

    def test(self):
        frozen = self.store.read_json("frozen.json")
        if fingerprint(frozen["payload"]) != frozen["fingerprint"]:
            raise ValueError("Frozen specification checksum mismatch")
        communities = frozen["payload"]["shortlist"]["communities"]
        controls = frozen["payload"]["controls"]
        rows = self.frame("test.parquet")
        retriever = DonorRetriever(self.frame("discovery.parquet"), self.config.duplicate_threshold)
        union = sorted(
            {
                tuple(v)
                for c in communities
                for group in (
                    c["members"],
                    controls[c["community_id"]]["random"],
                    controls[c["community_id"]]["unclustered"],
                )
                for v in group
            }
        )
        with (
            self.ledger("test/predictions.sqlite") as predictions,
            self.ledger("test/donors.sqlite") as donors,
            self.ledger("test/pairs.sqlite") as pairs,
            self.ledger("test/ablations.sqlite") as ablations,
        ):
            for row in progress(
                _records(rows), description="Evaluating frozen EXP5 test conditions"
            ):
                info, gold = self.runner.encode(row), row["emotion"]
                sample_id = row["sample_id"]
                baseline_key = "baseline:" + sample_id
                baseline = predictions.get(baseline_key)
                if baseline is None:
                    generation = self.runner.generate(info, gold)
                    self.runner.replay(row, generation)
                    scores, _ = self.runner.scores(info)
                    baseline = dict(
                        sample_id=sample_id,
                        emotion=gold,
                        duplicate_group=row["duplicate_group"],
                        community_id="baseline",
                        arm="baseline",
                        alpha=0.0,
                        generation=generation,
                        scores=scores,
                        rival=fixed_rival(scores, gold),
                        donor_id=None,
                        donor_prompt_hash=None,
                        abstention=None,
                        prompt_hash=info["prompt_hash"],
                    )
                    predictions.put(baseline_key, baseline)
                rival = baseline["rival"]
                for community in communities:
                    cid = community["community_id"]
                    prefix = cid + ":" + sample_id
                    if all(
                        predictions.get(prefix + ":" + arm) is not None
                        for arm in ("community", "random", "unclustered", "wrong_label")
                    ) and (
                        not baseline["generation"]["correct"]
                        or all(
                            ablations.get(prefix + ":" + arm) is not None
                            for arm in ("community", "random")
                        )
                    ):
                        continue
                    same, coords = self._donor(row, community, gold, True, retriever, donors, union)
                    alternative = frozen["payload"]["alternative_labels"][gold]
                    wrong, wrong_coords = self._donor(
                        row, community, alternative, False, retriever, donors, union
                    )
                    pairs.put(
                        prefix,
                        dict(
                            sample_id=sample_id,
                            community_id=cid,
                            same_label=same,
                            wrong_label=wrong,
                        ),
                    )
                    arms = (
                        ("community", community["members"], same, coords),
                        ("random", controls[cid]["random"], same, coords),
                        ("unclustered", controls[cid]["unclustered"], same, coords),
                        ("wrong_label", community["members"], wrong, wrong_coords),
                    )
                    for arm, members, donor, values in arms:
                        key = prefix + ":" + arm
                        if predictions.get(key) is not None:
                            continue
                        if values is None:
                            generation, scores, edit_audit = (
                                baseline["generation"],
                                {c: baseline["scores"][c] for c in (gold, rival)},
                                [],
                            )
                        else:
                            patch = self.runner.patch(members, values, self.config.alpha)
                            generation = self.runner.generate(info, gold, patch)
                            scores, edit_audit = self.runner.scores(info, [gold, rival], patch)
                        predictions.put(
                            key,
                            dict(
                                sample_id=sample_id,
                                emotion=gold,
                                duplicate_group=row["duplicate_group"],
                                community_id=cid,
                                arm=arm,
                                alpha=self.config.alpha,
                                generation=generation,
                                scores=scores,
                                score_edit_audit=edit_audit,
                                rival=rival,
                                prompt_hash=info["prompt_hash"],
                                **{
                                    k: donor[k]
                                    for k in ("donor_id", "donor_prompt_hash", "abstention")
                                },
                            ),
                        )
                    if baseline["generation"]["correct"]:
                        for arm, members in (
                            ("community", community["members"]),
                            ("random", controls[cid]["random"]),
                        ):
                            key = prefix + ":" + arm
                            if ablations.get(key) is not None:
                                continue
                            patch = self.runner.patch(members, zero_values(members), 1.0)
                            scores, audit = self.runner.scores(info, [gold, rival], patch)
                            ablations.put(
                                key,
                                dict(
                                    sample_id=sample_id,
                                    emotion=gold,
                                    community_id=cid,
                                    arm=arm,
                                    **_score_delta(baseline["scores"], scores, gold, rival),
                                    scores=scores,
                                    edit_audit=audit,
                                ),
                            )
            from affect_geometry.reporting.neuron_transfer import export_predictions

            paths = export_predictions(self.store, predictions)
            paths.extend(
                [
                    self.store.parquet("donor_pairs.parquet", pairs.frame()),
                    self.store.parquet("test_ablation_effects.parquet", ablations.frame()),
                ]
            )
        paths.extend(
            self.store.root / f"test/{name}.sqlite"
            for name in ("predictions", "donors", "pairs", "ablations")
        )
        paths.extend(sorted((self.store.root / "donor_activations").glob("*.npz")))
        return paths

    def report(self):
        from affect_geometry.reporting.neuron_transfer import write_report

        return write_report(self.store, self.config)
