#!/usr/bin/env python3
"""Select fixed-topology operating conditions for a cross-network comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from baseline import load_splits, plan_partitions, sha256
from download_assets import verify
from research_data import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    GEN_COLUMNS,
    network_identity,
    stable_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="case30_ieee")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "assets/manifests/cross-network-assets.lock.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT
        / "assets/model/mlflow/eval/case30_ieee/base/seed0/773035d6693e4de8b2114a0dd0b9549a/artifacts/stats/case30_ieee_scenario_splits.json",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "assets/data/case30_ieee/raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets/operating-conditions/case30",
    )
    parser.add_argument("--conditions", type=int, default=115)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.conditions < 2:
        parser.error("conditions must be at least two")

    lock = json.loads(args.lock.read_text())
    entries = [item for item in lock["files"] if item["repo_type"] == "dataset"]
    for entry in entries:
        if not verify(ROOT / entry["local"], entry):
            raise ValueError(f"missing or corrupt input: {entry['local']}")
    splits = load_splits(args.split)
    test_ids = sorted(splits["test"])
    plan, _ = plan_partitions(test_ids, entries)

    rows: list[dict] = []
    for number, (partition, ids) in enumerate(plan, 1):
        frames = [
            pd.read_parquet(
                args.raw_dir / f"{table}_data.parquet" / partition,
                columns=["scenario", "load_scenario_idx"] + columns,
                filters=[("scenario", "in", ids)],
            )
            for table, columns in (
                ("bus", BUS_COLUMNS),
                ("gen", GEN_COLUMNS),
                ("branch", BRANCH_COLUMNS),
            )
        ]
        grouped = [frame.groupby("scenario", sort=False) for frame in frames]
        for scenario_id in ids:
            bus, gen, branch = [group.get_group(scenario_id) for group in grouped]
            load_groups = set(bus.load_scenario_idx)
            if len(load_groups) != 1:
                raise ValueError(f"ambiguous load group for scenario {scenario_id}")
            rows.append(
                {
                    "scenario_id": int(scenario_id),
                    "load_scenario_idx": int(next(iter(load_groups))),
                    "partition": partition,
                    "research_split": "robustness",
                    **network_identity(bus, gen, branch),
                }
            )
        if number % 100 == 0 or number == len(plan):
            print(f"indexed {number}/{len(plan)} partitions; {len(rows)} test scenarios", flush=True)

    frame = pd.DataFrame(rows).sort_values("scenario_id")
    coverage = (
        frame.groupby("topology_id")
        .agg(
            groups=("load_scenario_idx", "nunique"),
            scenarios=("scenario_id", "size"),
            buses=("bus_count", "first"),
            generators=("active_gen_count", "first"),
            branches=("active_branch_count", "first"),
        )
        .reset_index()
        .sort_values(["groups", "scenarios", "topology_id"], ascending=[False, False, True])
    )
    topology = str(coverage.iloc[0].topology_id)
    candidates = frame[frame.topology_id == topology].sort_values(
        ["electrical_id", "scenario_id"]
    )
    representatives = candidates.drop_duplicates("load_scenario_idx")
    ordered_groups = stable_order(
        representatives.load_scenario_idx.tolist(), args.seed, f"{args.network}-robustness"
    )
    chosen_groups = set(ordered_groups[: args.conditions])
    selected = representatives[representatives.load_scenario_idx.isin(chosen_groups)].sort_values(
        "scenario_id"
    )
    if len(selected) < args.conditions:
        raise ValueError(
            f"selected topology has only {len(selected)} load groups; requested {args.conditions}"
        )
    if selected.node_map_id.nunique() != 1:
        raise ValueError("selected topology does not have a stable node mapping")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"{args.network}_index.csv"
    frame.to_csv(index_path, index=False, float_format="%.17g")
    coverage.to_csv(args.output_dir / f"{args.network}_topology_coverage.csv", index=False)
    dependencies = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (args.lock, args.split, index_path)
    }
    selection = {
        "schema_version": 1,
        "name": f"{args.network}_robustness",
        "research_split": "robustness",
        "topology_id": topology,
        "index_path": str(index_path.relative_to(ROOT)),
        "dependencies": dependencies,
        "scenario_ids": selected.scenario_id.astype(int).tolist(),
    }
    selection_path = args.output_dir / f"{args.network}_all.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                "network": args.network,
                "official_test_scenarios": len(frame),
                "topologies": len(coverage),
                "selected_topology": topology,
                "selected_conditions": len(selected),
                "buses": int(selected.bus_count.iloc[0]),
                "generators": int(selected.active_gen_count.iloc[0]),
                "branches": int(selected.active_branch_count.iloc[0]),
                "selection": str(selection_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
