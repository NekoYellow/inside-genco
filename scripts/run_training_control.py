#!/usr/bin/env python3
"""Run one matched GENCO architecture control with a provenance manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/gridfm-graphkit"))

ARCHITECTURES = ("untied", "tied_tail", "shared_graph_step_norms")
SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mp-context", choices=("spawn", "fork", "forkserver"))
    parser.add_argument("--compile", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    protocol_path = ROOT / "configs/training-controls.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    prepared_path = ROOT / protocol["output_root"] / "prepared-manifest.json"
    if not prepared_path.is_file():
        raise FileNotFoundError(
            "training controls are not prepared; run scripts/prepare_training_controls.py"
        )
    prepared = json.loads(prepared_path.read_text())
    run_spec = next(
        run
        for run in prepared["runs"]
        if run["architecture"] == args.architecture and run["seed"] == args.seed
    )
    config_path = ROOT / run_spec["config"]
    if sha256(config_path) != run_spec["config_sha256"]:
        raise ValueError("resolved training config hash differs from prepared manifest")
    normalizer_path = ROOT / prepared["normalizer_stats"]
    if sha256(normalizer_path) != prepared["normalizer_stats_sha256"]:
        raise ValueError("normalizer hash differs from prepared manifest")
    for split in prepared["split_files"].values():
        path = ROOT / split["path"]
        if sha256(path) != split["sha256"]:
            raise ValueError(f"fixed split hash differs: {path}")

    output_root = ROOT / protocol["output_root"]
    log_dir = output_root / "mlruns"
    run_directory = output_root / "runs" / args.architecture / f"seed{args.seed}"
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists() and not args.dry_run:
        raise FileExistsError(f"run manifest already exists: {manifest_path}")
    plan = {
        "architecture": args.architecture,
        "seed": args.seed,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "fixed_split_manifest_sha256": sha256(prepared_path),
        "normalizer_stats": str(normalizer_path.relative_to(ROOT)),
        "normalizer_stats_sha256": sha256(normalizer_path),
        "data_path": protocol["data_path"],
        "epochs": protocol["budget"]["epochs"],
        "checkpoint_selection": protocol["checkpoint_selection"],
        "losses": protocol["invariants"]["losses"],
        "loss_weights": protocol["invariants"]["loss_weights"],
        "optimizer": {
            key: protocol["invariants"][key]
            for key in ("optimizer", "learning_rate", "beta1", "beta2", "lr_decay", "lr_patience")
        },
        "precision": "bfloat16" if args.bfloat16 else "float32",
        "tf32": bool(args.tf32),
        "compile": args.compile,
        "num_workers": args.num_workers,
        "mp_context": args.mp_context,
        "log_dir": str(log_dir.relative_to(ROOT)),
        "run_name": f"{args.architecture}-seed{args.seed}",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "research_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": plan,
    }

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    before = set(log_dir.rglob("best_model_state_dict.pt")) if log_dir.exists() else set()
    try:
        from argparse import Namespace
        from gridfm_graphkit.cli import main_cli

        namespace = Namespace(
            command="train",
            config=str(config_path),
            exp_name=protocol["name"],
            run_name=plan["run_name"],
            log_dir=str(log_dir),
            data_path=str(ROOT / protocol["data_path"]),
            normalizer_stats=str(normalizer_path),
            compile=args.compile,
            bfloat16=args.bfloat16,
            tf32=args.tf32,
            dataset_wrapper=None,
            plugins=["inside_genco.tied_tail_plugin"],
            num_workers=args.num_workers,
            dataset_wrapper_cache_dir=None,
            profiler=None,
            compute_dc_ac_metrics=False,
            report_performance=True,
            mp_context=args.mp_context,
        )
        main_cli(namespace)
        after = set(log_dir.rglob("best_model_state_dict.pt"))
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(f"expected one new best checkpoint, found {len(created)}")
        checkpoint = created[0]
        manifest.update(
            status="complete",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            runtime_seconds=time.perf_counter() - started,
            best_checkpoint=str(checkpoint.relative_to(ROOT)),
            best_checkpoint_sha256=sha256(checkpoint),
        )
        save()
    except BaseException as error:
        manifest.update(
            status="failed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            runtime_seconds=time.perf_counter() - started,
            error={"type": type(error).__name__, "message": str(error)},
        )
        save()
        raise


if __name__ == "__main__":
    main()
