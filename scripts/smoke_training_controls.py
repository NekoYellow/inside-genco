#!/usr/bin/env python3
"""Run isolated one-batch training smokes for the GENCO architecture controls."""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/gridfm-graphkit"))

SMOKE_ROOT = ROOT / "outputs/training-smoke"
SOURCE_RAW = ROOT / "assets/data/case118_ieee/raw"
NORMALIZER = ROOT / "assets/model/case118_ieee/base/seed0/normalizer_stats.pt"
ARCHITECTURES = {
    "untied": "GNS_heterogeneous",
    "tied_tail": "GNS_heterogeneous_tied_tail",
    "shared_graph_step_norms": "GNS_heterogeneous_shared_graph_step_norms",
}


def prepare_smoke_data() -> tuple[Path, Path]:
    data_root = SMOKE_ROOT / "data"
    raw_root = data_root / "case118_ieee/raw"
    split_root = SMOKE_ROOT / "splits"
    raw_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    for table_name in ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet"):
        source = SOURCE_RAW / table_name / "scenario_partition=0"
        table = pd.read_parquet(source)
        table = table.loc[table["scenario"].isin((0, 1, 2))].copy()
        if set(table["scenario"].unique().tolist()) != {0, 1, 2}:
            raise ValueError(f"smoke source is missing scenarios 0, 1, or 2: {source}")
        table.to_parquet(raw_root / table_name, index=False)

    torch.save(torch.tensor([0], dtype=torch.long), split_root / "train.pt")
    torch.save(torch.tensor([1], dtype=torch.long), split_root / "val.pt")
    torch.save(torch.tensor([2], dtype=torch.long), split_root / "test.pt")
    return data_root, split_root


def prepare_smoke_config(architecture: str, split_root: Path) -> Path:
    source = ROOT / "outputs/training-controls/configs/untied_seed0.yaml"
    config = yaml.safe_load(source.read_text())
    config["model"]["type"] = ARCHITECTURES[architecture]
    config["data"]["scenarios"] = [3]
    config["data"]["workers"] = 0
    config["data"]["split_from_existing_files"] = str(split_root)
    config["training"]["batch_size"] = 1
    config["training"]["epochs"] = 1
    config["training"]["accelerator"] = "cpu"
    config["training"]["devices"] = 1
    config["training"]["strategy"] = "auto"
    config["callbacks"]["patience"] = 2

    config_root = SMOKE_ROOT / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    path = config_root / f"{architecture}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_smoke(architecture: str, data_root: Path, split_root: Path) -> dict:
    from gridfm_graphkit.cli import main_cli

    config_path = prepare_smoke_config(architecture, split_root)
    log_root = SMOKE_ROOT / "mlruns"
    before = set(log_root.rglob("best_model_state_dict.pt")) if log_root.exists() else set()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    main_cli(
        Namespace(
            command="train",
            config=str(config_path),
            exp_name="case118-training-controls-smoke",
            run_name=f"{architecture}-{run_stamp}",
            log_dir=str(log_root),
            data_path=str(data_root),
            normalizer_stats=str(NORMALIZER),
            compile=None,
            bfloat16=False,
            tf32=False,
            dataset_wrapper=None,
            plugins=["inside_genco.tied_tail_plugin"],
            num_workers=0,
            batch_size=1,
            dataset_wrapper_cache_dir=None,
            profiler=None,
            compute_dc_ac_metrics=False,
            report_performance=True,
        )
    )
    after = set(log_root.rglob("best_model_state_dict.pt"))
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(f"expected one new smoke checkpoint, found {len(created)}")
    return {
        "architecture": architecture,
        "status": "complete",
        "checkpoint": str(created[0].relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=(*ARCHITECTURES, "all"),
        default="all",
    )
    args = parser.parse_args()

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(SMOKE_ROOT / "matplotlib"))
    data_root, split_root = prepare_smoke_data()
    selected = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    results = [run_smoke(name, data_root, split_root) for name in selected]
    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scenarios": {"train": [0], "val": [1], "test": [2]},
            "epochs": 1,
            "batch_size": 1,
            "accelerator": "cpu",
            "corrections": 12,
        },
        "results": results,
    }
    manifest_path = SMOKE_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
