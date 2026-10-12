#!/usr/bin/env python3
"""Run trajectory, CKA, and single-correction ablations for one model variant."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import yaml

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
from genco_depth.interventions import InterventionPlan
from genco_depth.trajectory_metrics import local_step_metrics, node_relative_l2_updates
from genco_depth.similarity import linear_cka
from genco_depth.trace import TraceForwardAdapter


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    pd.DataFrame(rows).to_csv(path, index=False)


def stable_rows(batch, node_type: str) -> tuple[list[tuple[int, int]], torch.Tensor]:
    graph = batch[node_type].batch.detach().cpu()
    scenarios = batch.scenario_id.detach().cpu()[graph]
    devices = batch[node_type].device_id.detach().cpu()
    keys = [(int(s), int(d)) for s, d in zip(scenarios.tolist(), devices.tolist())]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate stable {node_type} identity")
    return keys, devices


def center_by_device(value: torch.Tensor, devices: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu().double()
    devices = devices.detach().cpu().long()
    unique, counts = torch.unique(devices, sorted=True, return_counts=True)
    if len(set(counts.tolist())) != 1:
        raise ValueError("fixed-topology centering requires every device in every condition")
    means = {int(device): value[devices == device].mean(dim=0) for device in unique}
    return value - torch.stack([means[int(device)] for device in devices.tolist()])


def rows_by_scenario(rows: list[dict]) -> dict[int, dict]:
    result = {int(row["scenario_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("scenario metrics are not unique")
    return result


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def bootstrap_associations(
    ablations: pd.DataFrame,
    local_updates: pd.DataFrame,
    local_change: pd.Series,
    *,
    samples: int = 2000,
    seed: int = 20261012,
) -> dict:
    """Resample operating conditions and recompute step-level rank associations.

    CKA is kept fixed because it is a pooled representation statistic.  Direct
    update norms and ablation damage are both recomputed from each paired
    operating-condition resample.
    """

    steps = sorted(map(int, local_change.index))
    damage = ablations.pivot(
        index="scenario_id", columns="correction_step", values="positive_delta_final_r_inf"
    )[steps]
    updates = local_updates[local_updates.node_type == "bus"].pivot(
        index="scenario_id", columns="correction_step", values="update_relative_l2"
    )[steps]
    if not damage.index.equals(updates.index) or damage.isna().any().any() or updates.isna().any().any():
        raise ValueError("bootstrap inputs are not complete and paired by operating condition")
    cka_score = local_change.loc[steps].to_numpy(float)
    damage_values = damage.to_numpy(float)
    update_values = updates.to_numpy(float)
    late_positions = np.asarray([index for index, step in enumerate(steps) if 7 <= step <= 11])
    rng = np.random.default_rng(seed)
    values = {
        "cka_all": np.empty(samples),
        "cka_late": np.empty(samples),
        "direct_all": np.empty(samples),
        "direct_late": np.empty(samples),
    }
    for sample in range(samples):
        selected = rng.integers(0, len(damage_values), size=len(damage_values))
        sampled_damage = damage_values[selected].mean(axis=0)
        sampled_update = np.median(update_values[selected], axis=0)
        values["cka_all"][sample] = rank_correlation(cka_score, sampled_damage)
        values["cka_late"][sample] = rank_correlation(
            cka_score[late_positions], sampled_damage[late_positions]
        )
        values["direct_all"][sample] = rank_correlation(sampled_update, sampled_damage)
        values["direct_late"][sample] = rank_correlation(
            sampled_update[late_positions], sampled_damage[late_positions]
        )

    point_damage = damage_values.mean(axis=0)
    point_update = np.median(update_values, axis=0)
    result = {}
    point = {
        "cka_all": rank_correlation(cka_score, point_damage),
        "cka_late": rank_correlation(cka_score[late_positions], point_damage[late_positions]),
        "direct_all": rank_correlation(point_update, point_damage),
        "direct_late": rank_correlation(point_update[late_positions], point_damage[late_positions]),
    }
    for name, estimates in values.items():
        finite = estimates[np.isfinite(estimates)]
        result[name] = {
            "spearman": point[name],
            "bootstrap_ci_low": float(np.quantile(finite, 0.025)),
            "bootstrap_ci_high": float(np.quantile(finite, 0.975)),
            "bootstrap_samples": int(len(finite)),
        }
    result["late_steps"] = [step for step in steps if 7 <= step <= 11]
    result["resampling_unit"] = "operating condition"
    return result


def load_variant(path: Path, name: str) -> dict:
    config = yaml.safe_load(path.read_text())
    variants = config.get("variants", {})
    if name not in variants:
        raise ValueError(f"unknown variant {name}; choose from {sorted(variants)}")
    variant = dict(variants[name])
    variant["name"] = name
    for key in ("model_dir", "raw_dir", "split", "selection", "data_lock", "model_lock"):
        variant[key] = (ROOT / variant[key]).resolve()
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/checkpoints.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="replace a complete artifact for the same variant")
    args = parser.parse_args()
    if args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("threads and batch size must be positive")

    variant = load_variant(args.config, args.variant)
    data_lock = json.loads(variant["data_lock"].read_text())
    model_lock = json.loads(variant["model_lock"].read_text())
    official_test = load_splits(variant["split"])["test"]
    ids, metadata, selection_info = load_selection(variant["selection"], official_test, ROOT)

    data_prefix = str(variant["raw_dir"].relative_to(ROOT)) + "/"
    data_entries = [
        item
        for item in data_lock["files"]
        if item["repo_type"] == "dataset" and item["local"].startswith(data_prefix)
    ]
    plan, required_data = plan_partitions(ids, data_entries)
    model_prefix = str(variant["model_dir"].relative_to(ROOT)) + "/"
    required_model = [
        item
        for item in model_lock["files"]
        if item["repo_type"] == "model" and item["local"].startswith(model_prefix)
    ]
    expected_model_files = {"best_model_state_dict.pt", "normalizer_stats.pt"}
    if not expected_model_files <= {Path(item["local"]).name for item in required_model}:
        raise ValueError("model lock does not cover the selected checkpoint")
    for entry in required_data + required_model:
        if not verify(ROOT / entry["local"], entry):
            raise ValueError(f"missing or corrupt input: {entry['local']}")

    output = (args.output or ROOT / f"outputs/trajectory-analysis/{args.variant}").resolve()
    if output.exists():
        old_manifest = output / "manifest.json"
        if not args.overwrite or not old_manifest.is_file():
            raise FileExistsError(f"output exists: {output}; use --overwrite only for a complete matching run")
        old = json.loads(old_manifest.read_text())
        if old.get("status") != "complete" or old.get("variant", {}).get("name") != args.variant:
            raise ValueError("refusing to overwrite an incomplete or mismatched robustness artifact")
    else:
        output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": {key: str(value) if isinstance(value, Path) else value for key, value in variant.items()},
        "selection": selection_info,
        "conditions": len(ids),
        "config_sha256": sha256(args.config),
        "data_lock_sha256": sha256(variant["data_lock"]),
        "model_lock_sha256": sha256(variant["model_lock"]),
        "device": "cpu",
        "forward_dtype": "float32",
        "metric_dtype": "float64",
        "threads": args.threads,
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
        config, normalizer, stats, task = load_model(
            "cpu", variant["model_dir"], variant["network"]
        )
        layer_count = len(task.model.layers)
        if layer_count != 12:
            raise ValueError(f"expected 12 corrections, found {layer_count}")
        batch_size = int(config.training.batch_size) if args.batch_size is None else args.batch_size
        state_before = model_state_sha256(task)
        adapter = TraceForwardAdapter(task)
        candidate_indices = list(range(1, layer_count - 1))

        identities = {"bus": [], "gen": []}
        devices = {"bus": [], "gen": []}
        states = {
            node: {
                stage: {step: [] for step in range(layer_count)}
                for stage in ("h_in", "h_graph", "h_out")
            }
            for node in ("bus", "gen")
        }
        physical_rows: list[dict] = []
        ablation_rows: list[dict] = []
        local_update_rows: list[dict] = []
        node_update_rows: list[dict] = []
        completed = 0

        with torch.inference_mode():
            for batch, groups, topologies in graph_batches(
                plan,
                config,
                normalizer,
                batch_size,
                metadata,
                raw_dir=variant["raw_dir"],
            ):
                trace = adapter(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
                baseline_rows, _ = scenario_metrics(
                    trace.output, batch, normalizer, groups, topologies, "float64"
                )
                baseline_by_id = rows_by_scenario(baseline_rows)

                for node in ("bus", "gen"):
                    keys, device_ids = stable_rows(batch, node)
                    identities[node].extend(keys)
                    devices[node].append(device_ids)
                for layer in trace.layers:
                    for node in ("bus", "gen"):
                        for stage in ("h_in", "h_graph", "h_out"):
                            states[node][stage][layer.layer_index].append(
                                getattr(layer, f"{stage}_{node}").double()
                            )
                    step_rows, _ = scenario_metrics(
                        {"bus": layer.x_bus, "gen": layer.x_gen},
                        batch,
                        normalizer,
                        groups,
                        topologies,
                        "float64",
                    )
                    for row in step_rows:
                        physical_rows.append(
                            {
                                "scenario_id": int(row["scenario_id"]),
                                "correction_step": layer.display_layer,
                                "r_inf_pu_100MVA": float(row["r_inf_pu_100MVA"]),
                            }
                        )
                    if layer.layer_index > 0:
                        for node_type in ("bus", "gen"):
                            node_batch = batch[node_type].batch.detach().cpu()
                            node_scenarios = batch.scenario_id.detach().cpu()[node_batch]
                            node_devices = batch[node_type].device_id.detach().cpu()
                            relative_updates = node_relative_l2_updates(
                                getattr(layer, f"h_in_{node_type}"),
                                getattr(layer, f"h_out_{node_type}"),
                            ).cpu()
                            node_update_rows.extend(
                                {
                                    "scenario_id": int(scenario_id),
                                    "correction_step": layer.display_layer,
                                    "node_type": node_type,
                                    "device_id": int(device_id),
                                    "update_relative_l2": float(relative_update),
                                }
                                for scenario_id, device_id, relative_update in zip(
                                    node_scenarios.tolist(),
                                    node_devices.tolist(),
                                    relative_updates.tolist(),
                                )
                            )
                    if layer.layer_index in candidate_indices:
                        for graph_index, scenario_id in enumerate(batch.scenario_id.tolist()):
                            for node_type in ("bus", "gen"):
                                graph_mask = batch[node_type].batch.detach().cpu() == graph_index
                                metric = local_step_metrics(
                                    getattr(layer, f"h_in_{node_type}")[graph_mask],
                                    getattr(layer, f"h_out_{node_type}")[graph_mask],
                                )
                                local_update_rows.append(
                                    {
                                        "scenario_id": int(scenario_id),
                                        "correction_step": layer.display_layer,
                                        "node_type": node_type,
                                        "update_relative_l2": metric.update_relative_l2,
                                        "row_cosine_mean": metric.row_cosine_mean,
                                    }
                                )

                for step_index in candidate_indices:
                    changed = adapter(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch.edge_attr_dict,
                        batch.mask_dict,
                        intervention=InterventionPlan(skip_layers=frozenset({step_index})),
                    )
                    changed_rows, _ = scenario_metrics(
                        changed.output, batch, normalizer, groups, topologies, "float64"
                    )
                    for scenario_id, changed_row in rows_by_scenario(changed_rows).items():
                        delta = float(
                            changed_row["r_inf_pu_100MVA"]
                            - baseline_by_id[scenario_id]["r_inf_pu_100MVA"]
                        )
                        ablation_rows.append(
                            {
                                "scenario_id": scenario_id,
                                "correction_step": step_index + 1,
                                "delta_final_r_inf": delta,
                                "positive_delta_final_r_inf": max(delta, 0.0),
                            }
                        )
                completed += len(batch.scenario_id)
                print(f"{args.variant}: {completed}/{len(ids)} conditions", flush=True)

        if sorted({row["scenario_id"] for row in physical_rows}) != ids:
            raise ValueError("evaluated conditions differ from selection")
        state_after = model_state_sha256(task)
        if state_after != state_before:
            raise ValueError("checkpoint state changed during analysis")

        cka_rows: list[dict] = []
        for node in ("bus", "gen"):
            row_ids = identities[node]
            device_ids = torch.cat(devices[node])
            graph_states = {
                step: torch.cat(states[node]["h_graph"][step]) for step in range(layer_count)
            }
            centered_states = {
                step: center_by_device(value, device_ids) for step, value in graph_states.items()
            }
            for centered, values in ((False, graph_states), (True, centered_states)):
                for left in range(layer_count):
                    for right in range(layer_count):
                        result = linear_cka(
                            values[left], values[right], x_ids=row_ids, y_ids=row_ids
                        )
                        cka_rows.append(
                            {
                                "kind": "cross_step_h_graph",
                                "node_type": node,
                                "node_centered": centered,
                                "left_step": left + 1,
                                "right_step": right + 1,
                                **asdict(result),
                            }
                        )
            for step in candidate_indices:
                result = linear_cka(
                    torch.cat(states[node]["h_in"][step]),
                    torch.cat(states[node]["h_out"][step]),
                    x_ids=row_ids,
                    y_ids=row_ids,
                )
                cka_rows.append(
                    {
                        "kind": "whole_step_h_in_h_out",
                        "node_type": node,
                        "node_centered": False,
                        "left_step": step + 1,
                        "right_step": step + 1,
                        **asdict(result),
                    }
                )

        physical = pd.DataFrame(physical_rows)
        step_summary = (
            physical.groupby("correction_step")["r_inf_pu_100MVA"]
            .agg(
                median="median",
                p95=lambda values: float(np.quantile(values, 0.95)),
                mean="mean",
            )
            .reset_index()
        )
        nonmonotone = int(
            sum(
                (group.sort_values("correction_step")["r_inf_pu_100MVA"].diff().dropna() > 0).any()
                for _, group in physical.groupby("scenario_id")
            )
        )
        ablations = pd.DataFrame(ablation_rows)
        ablation_summary = (
            ablations.groupby("correction_step")
            .agg(
                mean_delta_final_r_inf=("delta_final_r_inf", "mean"),
                mean_positive_final_delta_r_inf=("positive_delta_final_r_inf", "mean"),
                p95_delta_final_r_inf=("delta_final_r_inf", lambda values: float(np.quantile(values, 0.95))),
            )
            .reset_index()
        )
        local_updates = pd.DataFrame(local_update_rows)
        node_updates = pd.DataFrame(node_update_rows)
        expected_node_update_steps = set(range(2, layer_count + 1))
        if set(node_updates.correction_step.unique()) != expected_node_update_steps:
            raise ValueError("node-level updates do not cover correction steps 2-12")
        if not np.isfinite(node_updates.update_relative_l2).all():
            raise ValueError("node-level updates contain non-finite values")
        local_update_summary = (
            local_updates.groupby(["correction_step", "node_type"])
            .agg(
                update_relative_l2_mean=("update_relative_l2", "mean"),
                update_relative_l2_median=("update_relative_l2", "median"),
                update_relative_l2_p95=("update_relative_l2", lambda values: float(np.quantile(values, 0.95))),
                row_cosine_mean=("row_cosine_mean", "mean"),
            )
            .reset_index()
        )
        cka = pd.DataFrame(cka_rows)
        local = cka[
            (cka.kind == "whole_step_h_in_h_out")
            & (cka.node_type == "bus")
        ].set_index("left_step")["value"]
        association = ablation_summary.set_index("correction_step").join(local.rename("cka"))
        association["local_change"] = 1.0 - association["cka"]
        bus_update = local_update_summary[local_update_summary.node_type == "bus"].set_index(
            "correction_step"
        )
        association = association.join(bus_update[[
            "update_relative_l2_mean", "update_relative_l2_median", "update_relative_l2_p95"
        ]])
        associations = bootstrap_associations(
            ablations,
            local_updates,
            association["local_change"],
        )
        spearman = associations["cka_all"]["spearman"]
        direct_spearman = associations["direct_all"]["spearman"]

        condition_rows = []
        for step in candidate_indices:
            display_step = step + 1
            paired = local_updates[
                (local_updates.node_type == "bus")
                & (local_updates.correction_step == display_step)
            ][["scenario_id", "update_relative_l2"]].merge(
                ablations[ablations.correction_step == display_step][[
                    "scenario_id", "positive_delta_final_r_inf"
                ]],
                on="scenario_id",
                validate="one_to_one",
            )
            threshold = float(paired.update_relative_l2.quantile(0.25))
            bottom = paired[paired.update_relative_l2 <= threshold]
            condition_rows.append(
                {
                    "correction_step": display_step,
                    "conditions": int(len(paired)),
                    "condition_level_spearman": rank_correlation(
                        paired.update_relative_l2.to_numpy(float),
                        paired.positive_delta_final_r_inf.to_numpy(float),
                    ),
                    "bottom_quartile_update_threshold": threshold,
                    "bottom_quartile_conditions": int(len(bottom)),
                    "bottom_quartile_positive_damage_rate": float(
                        (bottom.positive_delta_final_r_inf > 0).mean()
                    ),
                    "bottom_quartile_damage_median": float(
                        bottom.positive_delta_final_r_inf.median()
                    ),
                }
            )
        late = range(7, 13)
        late_cka = {}
        for node in ("bus", "gen"):
            values = cka[
                (cka.kind == "cross_step_h_graph")
                & (cka.node_type == node)
                & cka.node_centered.astype(bool)
                & cka.left_step.isin(late)
                & cka.right_step.isin(late)
                & (cka.left_step != cka.right_step)
            ].value
            late_cka[node] = float(values.mean())

        write_csv(output / "physical_trajectory.csv", physical_rows)
        step_summary.to_csv(output / "step_summary.csv", index=False)
        write_csv(output / "ablation.csv", ablation_rows)
        ablation_summary.to_csv(output / "ablation_summary.csv", index=False)
        write_csv(output / "local_updates.csv", local_update_rows)
        local_update_summary.to_csv(output / "local_update_summary.csv", index=False)
        node_updates.to_csv(output / "node_updates.csv", index=False, float_format="%.10g")
        pd.DataFrame(condition_rows).to_csv(output / "condition_level_association.csv", index=False)
        write_csv(output / "cka.csv", cka_rows)
        association.reset_index().to_csv(output / "local_change_vs_importance.csv", index=False)

        summary = {
            "variant": args.variant,
            "label": variant["label"],
            "network": variant["network"],
            "size": variant["size"],
            "seed": int(variant["seed"]),
            "conditions": len(ids),
            "buses": int(metadata[ids[0]]["bus_count"]),
            "layers": layer_count,
            "hidden_size": int(config.model.hidden_size),
            "parameters": int(sum(parameter.numel() for parameter in task.parameters())),
            "r_inf_median_by_step": {
                str(int(row.correction_step)): float(row.median)
                for row in step_summary.itertuples()
            },
            "conditions_with_adjacent_worsening": nonmonotone,
            "late_node_centered_cka": late_cka,
            "local_change_importance_spearman": spearman,
            "direct_update_importance_spearman": direct_spearman,
            "association_statistics": associations,
            "smallest_local_change_step": int(association.local_change.idxmin()),
            "smallest_direct_update_step": int(association.update_relative_l2_median.idxmin()),
            "node_update_rows": int(len(node_updates)),
            "least_ablation_damage_step": int(
                association.mean_positive_final_delta_r_inf.idxmin()
            ),
            "model_state_sha256": state_before,
            "evaluation_seconds": time.perf_counter() - started,
        }
        (output / "summary.json").write_text(json.dumps(clean_json(summary), indent=2) + "\n")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest.update(
            status="complete",
            platform=platform.platform(),
            research_git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            model_state_sha256_before=state_before,
            model_state_sha256_after=state_after,
            process_peak_RSS_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
            summary=summary,
        )
        save_manifest()
        print(json.dumps(clean_json(summary), indent=2))
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        save_manifest()
        raise


if __name__ == "__main__":
    main()
