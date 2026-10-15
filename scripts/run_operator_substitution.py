#!/usr/bin/env python3
"""Test whether GENCO graph operators are interchangeable across corrections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor/gridfm-graphkit"))

import numpy as np
import pandas as pd
import torch

from baseline import (
    clean_json,
    graph_batches,
    load_model,
    load_splits,
    model_state_sha256,
    plan_partitions,
    scenario_metrics,
    sha256,
)
from download_assets import verify
from research_data import load_selection
from run_trajectory_analysis import PHYSICAL_METRICS, load_variant, rows_by_scenario
from inside_genco.interventions import InterventionPlan
from inside_genco.trace import TraceForwardAdapter


def parse_steps(raw: str | None, layer_count: int) -> list[int]:
    """Parse one-based compatible correction numbers."""

    if raw is None:
        return list(range(2, layer_count + 1))
    try:
        steps = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not steps or len(steps) != len(set(steps)):
        raise argparse.ArgumentTypeError("steps must be non-empty and unique")
    if any(step < 2 or step > layer_count for step in steps):
        raise argparse.ArgumentTypeError(f"steps must be in 2..{layer_count}")
    return sorted(steps)


def summarize_substitutions(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario_id",
        "target_step",
        "source_step",
        "r_inf_pu_100MVA",
        "delta_r_inf_pu_100MVA",
        "positive_delta_r_inf_pu_100MVA",
        "delta_vm_mae_pu",
        "delta_va_mae_rad",
        "delta_p_mae_all_bus_MW",
        "delta_q_mae_all_bus_Mvar",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"operator rows are missing columns: {missing}")
    return (
        rows.groupby(["target_step", "source_step"])
        .agg(
            conditions=("scenario_id", "size"),
            final_r_inf_mean=("r_inf_pu_100MVA", "mean"),
            final_r_inf_median=("r_inf_pu_100MVA", "median"),
            delta_final_r_inf_mean=("delta_r_inf_pu_100MVA", "mean"),
            delta_final_r_inf_positive_mean=(
                "positive_delta_r_inf_pu_100MVA", "mean"
            ),
            delta_final_r_inf_p95=(
                "delta_r_inf_pu_100MVA", lambda values: float(np.quantile(values, 0.95))
            ),
            delta_vm_mae_pu_mean=("delta_vm_mae_pu", "mean"),
            delta_va_mae_rad_mean=("delta_va_mae_rad", "mean"),
            delta_p_mae_all_bus_MW_mean=("delta_p_mae_all_bus_MW", "mean"),
            delta_q_mae_all_bus_Mvar_mean=("delta_q_mae_all_bus_Mvar", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/checkpoints.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--targets", help="comma-separated one-based target corrections")
    parser.add_argument("--sources", help="comma-separated one-based source corrections")
    parser.add_argument("--component", choices=("joint", "graph", "norm"), default="joint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("threads and batch size must be positive")

    variant = load_variant(args.config, args.variant)
    data_lock = json.loads(variant["data_lock"].read_text())
    model_lock = json.loads(variant["model_lock"].read_text())
    official_test = load_splits(variant["split"])["test"]
    ids, metadata, selection_info = load_selection(
        variant["selection"], official_test, ROOT
    )

    data_prefix = str(variant["raw_dir"].relative_to(ROOT)) + "/"
    data_entries = [
        item
        for item in data_lock["files"]
        if item["repo_type"] == "dataset" and item["local"].startswith(data_prefix)
    ]
    partition_plan, required_data = plan_partitions(ids, data_entries)
    model_prefix = str(variant["model_dir"].relative_to(ROOT)) + "/"
    required_model = [
        item
        for item in model_lock["files"]
        if item["repo_type"] == "model" and item["local"].startswith(model_prefix)
    ]
    for entry in required_data + required_model:
        if not verify(ROOT / entry["local"], entry):
            raise ValueError(f"missing or corrupt input: {entry['local']}")

    output = (
        args.output or ROOT / "outputs/operator-substitution" / (args.variant if args.component == "joint" else f"{args.variant}-{args.component}")
    ).resolve()
    if output.exists():
        manifest_path = output / "manifest.json"
        if not args.overwrite or not manifest_path.is_file():
            raise FileExistsError(f"output exists: {output}")
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("status") != "complete"
            or previous.get("variant", {}).get("name") != args.variant
            or previous.get("component", "joint") != args.component
        ):
            raise ValueError("refusing to overwrite an incomplete or mismatched artifact")
    else:
        output.mkdir(parents=True)
    manifest_path = output / "manifest.json"

    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in variant.items()
        },
        "selection": selection_info,
        "conditions": len(ids),
        "config_sha256": sha256(args.config),
        "data_lock_sha256": sha256(variant["data_lock"]),
        "model_lock_sha256": sha256(variant["model_lock"]),
        "device": "cpu",
        "forward_dtype": "float32",
        "metric_dtype": "float64",
        "threads": args.threads,
        "component": args.component,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "src/inside_genco/trace.py", ROOT / "src/inside_genco/interventions.py")},
    }

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(clean_json(manifest), indent=2) + "\n")

    save_manifest()
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        config, normalizer, _stats, task = load_model(
            "cpu", variant["model_dir"], variant["network"]
        )
        layer_count = len(task.model.layers)
        targets = parse_steps(args.targets, layer_count)
        sources = parse_steps(args.sources, layer_count)
        manifest["targets"] = targets
        manifest["sources"] = sources
        save_manifest()
        batch_size = (
            int(config.training.batch_size) if args.batch_size is None else args.batch_size
        )
        state_before = model_state_sha256(task)
        adapter = TraceForwardAdapter(task)
        rows = []
        completed = 0

        with torch.inference_mode():
            for batch, groups, topologies in graph_batches(
                partition_plan,
                config,
                normalizer,
                batch_size,
                metadata,
                raw_dir=variant["raw_dir"],
            ):
                baseline = adapter(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch.edge_attr_dict,
                    batch.mask_dict,
                )
                baseline_rows, _ = scenario_metrics(
                    baseline.output,
                    batch,
                    normalizer,
                    groups,
                    topologies,
                    "float64",
                )
                baseline_by_id = rows_by_scenario(baseline_rows)
                if completed == 0:
                    original = task.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
                    for node in ("bus", "gen"):
                        torch.testing.assert_close(baseline.output[node], original[node], rtol=1e-5, atol=1e-6)
                    manifest["baseline_forward_verified"] = True

                for target_step in targets:
                    for source_step in sources:
                        if source_step == target_step:
                            changed_rows = baseline_rows
                        else:
                            changed = adapter(
                                batch.x_dict,
                                batch.edge_index_dict,
                                batch.edge_attr_dict,
                                batch.mask_dict,
                                intervention=InterventionPlan(
                                    operator_sources={target_step - 1: source_step - 1},
                                    operator_component=args.component,
                                ),
                            )
                            changed_rows, _ = scenario_metrics(
                                changed.output,
                                batch,
                                normalizer,
                                groups,
                                topologies,
                                "float64",
                            )
                        for changed_row in changed_rows:
                            scenario_id = int(changed_row["scenario_id"])
                            reference = baseline_by_id[scenario_id]
                            delta_r_inf = float(
                                changed_row["r_inf_pu_100MVA"]
                                - reference["r_inf_pu_100MVA"]
                            )
                            rows.append(
                                {
                                    "scenario_id": scenario_id,
                                    "target_step": target_step,
                                    "source_step": source_step,
                                    "substituted": source_step != target_step,
                                    **{
                                        metric: float(changed_row[metric])
                                        for metric in PHYSICAL_METRICS
                                    },
                                    "delta_r_inf_pu_100MVA": delta_r_inf,
                                    "positive_delta_r_inf_pu_100MVA": max(
                                        delta_r_inf, 0.0
                                    ),
                                    "delta_vm_mae_pu": float(
                                        changed_row["vm_mae_pu"]
                                        - reference["vm_mae_pu"]
                                    ),
                                    "delta_va_mae_rad": float(
                                        changed_row["va_mae_rad"]
                                        - reference["va_mae_rad"]
                                    ),
                                    "delta_p_mae_all_bus_MW": float(
                                        changed_row["p_mae_all_bus_MW"]
                                        - reference["p_mae_all_bus_MW"]
                                    ),
                                    "delta_q_mae_all_bus_Mvar": float(
                                        changed_row["q_mae_all_bus_Mvar"]
                                        - reference["q_mae_all_bus_Mvar"]
                                    ),
                                }
                            )

                completed += len(batch.scenario_id)
                print(f"{args.component}: {completed}/{len(ids)} conditions", flush=True)

        frame = pd.DataFrame(rows)
        expected_rows = len(ids) * len(targets) * len(sources)
        if len(frame) != expected_rows:
            raise ValueError(
                f"operator matrix has {len(frame)} rows; expected {expected_rows}"
            )
        if not np.isfinite(frame[list(PHYSICAL_METRICS)].to_numpy(float)).all():
            raise ValueError("operator substitution produced non-finite physical metrics")
        summary = summarize_substitutions(frame)
        frame.to_csv(output / "condition_level.csv", index=False, float_format="%.10g")
        summary.to_csv(output / "matrix_summary.csv", index=False, float_format="%.10g")

        best = summary.loc[
            summary.groupby("target_step")["final_r_inf_mean"].idxmin(),
            ["target_step", "source_step"],
        ]
        result = {
            "variant": args.variant,
            "component": args.component,
            "label": variant["label"],
            "conditions": len(ids),
            "targets": targets,
            "sources": sources,
            "diagonal_best_targets": int(
                (best.target_step == best.source_step).sum()
            ),
            "comparable_targets": int(len(best)),
            "model_state_sha256": state_before,
            "evaluation_seconds": time.perf_counter() - started,
        }
        (output / "summary.json").write_text(
            json.dumps(clean_json(result), indent=2) + "\n"
        )
        state_after = model_state_sha256(task)
        if state_after != state_before:
            raise ValueError("frozen checkpoint state changed during operator substitution")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest.update(
            status="complete",
            platform=platform.platform(),
            research_git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            model_state_sha256_before=state_before,
            model_state_sha256_after=state_after,
            process_peak_RSS_bytes=int(
                peak if sys.platform == "darwin" else peak * 1024
            ),
            summary=result,
        )
        save_manifest()
        print(json.dumps(clean_json(result), indent=2))
    except BaseException as error:
        manifest.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        save_manifest()
        raise


if __name__ == "__main__":
    main()
