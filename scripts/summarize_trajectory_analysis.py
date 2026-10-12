#!/usr/bin/env python3
"""Compare trajectories, ablations, and node activity across checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm, Normalize


ORDER = (
    "case118_base_seed0",
    "case118_base_seed1",
    "case118_small_seed0",
    "case30_base_seed0",
)
COLORS = ("#264653", "#2a9d8f", "#e76f51", "#7b2cbf")
ACTIVITY_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def save_figure(fig: plt.Figure, category: str, stem: str) -> list[str]:
    output = ROOT / "figures" / category
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(path.relative_to(ROOT))]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/checkpoints.yaml").read_text())["variants"]
    records = []
    tables = {}
    physical_tables = {}
    for name in ORDER:
        directory = ROOT / "outputs/trajectory-analysis" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete robustness run: {name}")
        summary = json.loads((directory / "summary.json").read_text())
        steps = pd.read_csv(directory / "step_summary.csv")
        association = pd.read_csv(directory / "local_change_vs_importance.csv")
        node_updates = pd.read_csv(directory / "node_updates.csv")
        physical_metrics = pd.read_csv(directory / "physical_metric_summary.csv")
        physical_updates = pd.read_csv(directory / "physical_update_summary.csv")
        required_node_columns = {
            "scenario_id", "correction_step", "node_type", "device_id", "update_relative_l2"
        }
        if required_node_columns - set(node_updates.columns):
            raise ValueError(f"incomplete node-update table: {name}")
        if set(node_updates.correction_step.unique()) != set(range(2, 13)):
            raise ValueError(f"node-update steps are incomplete: {name}")
        if set(physical_metrics.correction_step.unique()) != set(range(1, 13)):
            raise ValueError(f"physical-metric steps are incomplete: {name}")
        if set(physical_updates.correction_step.unique()) != set(range(2, 13)):
            raise ValueError(f"physical-update steps are incomplete: {name}")
        tables[name] = (summary, steps, association, node_updates)
        physical_tables[name] = (physical_metrics, physical_updates)
        records.append(
            {
                "variant": name,
                "label": config[name]["label"],
                "network": summary["network"],
                "size": summary["size"],
                "seed": summary["seed"],
                "conditions": summary["conditions"],
                "buses": summary["buses"],
                "hidden_size": summary["hidden_size"],
                "parameters": summary["parameters"],
                "r_inf_median_step_1": float(summary["r_inf_median_by_step"]["1"]),
                "r_inf_median_step_12": float(summary["r_inf_median_by_step"]["12"]),
                "conditions_with_adjacent_worsening": summary["conditions_with_adjacent_worsening"],
                "late_bus_cka": summary["late_node_centered_cka"]["bus"],
                "late_generator_cka": summary["late_node_centered_cka"]["gen"],
                "local_importance_spearman": summary["local_change_importance_spearman"],
                "local_importance_spearman_ci_low": summary["association_statistics"]["cka_all"]["bootstrap_ci_low"],
                "local_importance_spearman_ci_high": summary["association_statistics"]["cka_all"]["bootstrap_ci_high"],
                "late_local_importance_spearman": summary["association_statistics"]["cka_late"]["spearman"],
                "late_local_importance_spearman_ci_low": summary["association_statistics"]["cka_late"]["bootstrap_ci_low"],
                "late_local_importance_spearman_ci_high": summary["association_statistics"]["cka_late"]["bootstrap_ci_high"],
                "direct_update_importance_spearman": summary["direct_update_importance_spearman"],
                "direct_update_importance_spearman_ci_low": summary["association_statistics"]["direct_all"]["bootstrap_ci_low"],
                "direct_update_importance_spearman_ci_high": summary["association_statistics"]["direct_all"]["bootstrap_ci_high"],
                "late_direct_update_importance_spearman": summary["association_statistics"]["direct_late"]["spearman"],
                "late_direct_update_importance_spearman_ci_low": summary["association_statistics"]["direct_late"]["bootstrap_ci_low"],
                "late_direct_update_importance_spearman_ci_high": summary["association_statistics"]["direct_late"]["bootstrap_ci_high"],
                "smallest_local_change_step": summary["smallest_local_change_step"],
                "smallest_direct_update_step": summary["smallest_direct_update_step"],
                "least_ablation_damage_step": summary["least_ablation_damage_step"],
                "model_state_sha256": summary["model_state_sha256"],
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / "results/robustness-summary.csv", index=False)
    physical_result_rows = []
    for row in records:
        metrics, updates = physical_tables[row["variant"]]
        for kind, table in (("solution_quality", metrics), ("adjacent_change", updates)):
            selected = table.copy()
            selected.insert(0, "kind", kind)
            selected.insert(0, "label", row["label"])
            selected.insert(0, "variant", row["variant"])
            physical_result_rows.append(selected)
    pd.concat(physical_result_rows, ignore_index=True).to_csv(
        ROOT / "results/physical-refinement-summary.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for color, row in zip(COLORS, records):
        _, steps, _, _ = tables[row["variant"]]
        axes[0].plot(
            steps.correction_step,
            steps["median"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=row["label"],
        )
    axes[0].set(
        xlabel="Correction step",
        ylabel=r"Median $R_\infty$ (p.u., 100 MVA)",
        yscale="log",
        xticks=range(1, 13),
    )
    axes[0].grid(color="#e5e7eb", linewidth=0.65)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)

    x = np.arange(len(records))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        [row["late_bus_cka"] for row in records],
        width,
        label="Bus",
        color="#2a9d8f",
    )
    axes[1].bar(
        x + width / 2,
        [row["late_generator_cka"] for row in records],
        width,
        label="Generator",
        color="#e76f51",
    )
    axes[1].set(
        ylabel="Mean node-centered CKA, steps 7–12",
        ylim=(0, 1.04),
        xticks=x,
        xticklabels=[row["label"].replace("case118 ", "118\n").replace("case30 ", "30\n") for row in records],
    )
    axes[1].grid(axis="y", color="#e5e7eb", linewidth=0.65)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    trajectory_figures = save_figure(fig, "trajectories", "cross_checkpoint_trajectory_and_cka")

    primary_metrics, primary_physical_updates = physical_tables[ORDER[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    mismatch = primary_metrics[
        primary_metrics.metric.isin(("p_mae_all_bus_MW", "q_mae_all_bus_Mvar"))
    ]
    for metric, label, color in (
        ("p_mae_all_bus_MW", "Active mismatch (MW)", "#264653"),
        ("q_mae_all_bus_Mvar", "Reactive mismatch (Mvar)", "#e76f51"),
    ):
        selected = mismatch[mismatch.metric == metric].sort_values("correction_step")
        axes[0].plot(
            selected.correction_step,
            selected["median"],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    axes[0].set(
        xlabel="Correction step",
        ylabel="Median absolute bus mismatch",
        yscale="log",
        xticks=range(1, 13),
        title="Physical feasibility keeps improving",
    )
    axes[0].legend(frameon=False)

    movement_specs = (
        ("vm_step_mae_pu", r"$V_m$", "#264653"),
        ("va_step_mae_rad", r"$V_a$", "#2a9d8f"),
        ("pg_bus_step_mae_MW", r"bus $P_g$", "#e9c46a"),
        ("qg_bus_step_mae_Mvar", r"bus $Q_g$", "#e76f51"),
        ("pg_gen_step_mae_MW", r"generator $P_g$", "#7b2cbf"),
    )
    for metric, label, color in movement_specs:
        selected = primary_physical_updates[
            primary_physical_updates.metric == metric
        ].sort_values("correction_step")
        values = selected["median"].to_numpy(float)
        axes[1].plot(
            selected.correction_step,
            values / values[0],
            marker="o",
            linewidth=1.6,
            label=label,
            color=color,
        )
    axes[1].set(
        xlabel="Correction step",
        ylabel="Median adjacent change / correction-2 change",
        yscale="log",
        xticks=range(2, 13),
        title="Physical updates shrink without vanishing",
    )
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    for ax in axes:
        ax.grid(color="#e5e7eb", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    physical_figures = save_figure(
        fig, "trajectories", "physical_refinement_trajectory"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    for ax, color, row in zip(axes.ravel(), COLORS, records):
        _, _, association, _ = tables[row["variant"]]
        ax.scatter(
            association.local_change,
            association.mean_positive_final_delta_r_inf,
            s=42,
            color=color,
        )
        for point in association.itertuples():
            ax.annotate(
                str(int(point.correction_step)),
                (point.local_change, point.mean_positive_final_delta_r_inf),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set(xscale="log", yscale="log", title=row["label"])
        ax.grid(color="#e5e7eb", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("CKA geometry change (1 − bus whole-step CKA)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Mean positive final $\Delta R_\infty$")
    importance_figures = save_figure(fig, "interventions", "representation_change_vs_importance")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    for ax, color, row in zip(axes.ravel(), COLORS, records):
        _, _, association, _ = tables[row["variant"]]
        ax.scatter(
            association.update_relative_l2_median,
            association.mean_positive_final_delta_r_inf,
            s=42,
            color=color,
        )
        for point in association.itertuples():
            ax.annotate(
                str(int(point.correction_step)),
                (point.update_relative_l2_median, point.mean_positive_final_delta_r_inf),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set(xscale="log", yscale="log", title=row["label"])
        ax.grid(color="#e5e7eb", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Median direct update $\|H_{out}-H_{in}\|_F/(\|H_{in}\|_F+\epsilon)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Mean positive final $\Delta R_\infty$")
    direct_importance_figures = save_figure(fig, "interventions", "direct_update_vs_importance")

    activity_rows = []
    for row in records:
        node_updates = tables[row["variant"]][3]
        for (node_type, step), selected in node_updates.groupby(
            ["node_type", "correction_step"], sort=True
        ):
            for threshold in ACTIVITY_THRESHOLDS:
                by_condition = (
                    selected.assign(active=selected.update_relative_l2 > threshold)
                    .groupby("scenario_id")["active"]
                    .mean()
                )
                activity_rows.append(
                    {
                        "variant": row["variant"],
                        "label": row["label"],
                        "node_type": node_type,
                        "correction_step": int(step),
                        "threshold": threshold,
                        "active_fraction": float(by_condition.mean()),
                        "condition_active_fraction_median": float(by_condition.median()),
                        "condition_active_fraction_p05": float(by_condition.quantile(0.05)),
                        "condition_active_fraction_p95": float(by_condition.quantile(0.95)),
                        "conditions": int(len(by_condition)),
                        "stable_nodes": int(selected.device_id.nunique()),
                    }
                )
    activity = pd.DataFrame(activity_rows)
    activity.to_csv(ROOT / "results/node-activity-summary.csv", index=False)

    primary_activity = activity[activity.variant == ORDER[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
    threshold_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ACTIVITY_THRESHOLDS)))
    for ax, node_type, title in zip(axes, ("bus", "gen"), ("Bus nodes", "Generator nodes")):
        for threshold, color in zip(ACTIVITY_THRESHOLDS, threshold_colors):
            selected = primary_activity[
                (primary_activity.node_type == node_type)
                & (primary_activity.threshold == threshold)
            ].sort_values("correction_step")
            ax.plot(
                selected.correction_step,
                selected.active_fraction,
                marker="o",
                linewidth=1.8,
                color=color,
                label=fr"$\tau={threshold:g}$",
            )
        ax.set(
            title=title,
            xlabel="Correction step",
            ylabel="Fraction of nodes above threshold",
            ylim=(-0.03, 1.03),
            xticks=range(2, 13),
        )
        ax.grid(color="#e5e7eb", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=1, loc="upper right")
    fig.suptitle(
        "Node-level state activity · case118 Base seed 0",
        fontsize=14,
        fontweight="bold",
    )
    activity_fraction_figures = save_figure(fig, "node-activity", "node_activity_fraction")

    threshold_grid = np.geomspace(0.005, 1.0, 48)
    steps_grid = np.arange(2, 13)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), constrained_layout=True)
    heatmap = None
    for ax, row in zip(axes.ravel(), records):
        node_updates = tables[row["variant"]][3]
        node_updates = node_updates[node_updates.node_type == "bus"]
        matrix = np.asarray(
            [
                [
                    float(
                        (
                            node_updates[node_updates.correction_step == step].update_relative_l2
                            > threshold
                        ).mean()
                    )
                    for step in steps_grid
                ]
                for threshold in threshold_grid
            ]
        )
        heatmap = ax.pcolormesh(
            steps_grid,
            threshold_grid,
            matrix,
            shading="nearest",
            cmap="magma",
            norm=Normalize(vmin=0.0, vmax=1.0),
        )
        ax.set(
            title=row["label"],
            xlabel="Correction step",
            ylabel=r"Relative-update threshold $\tau$",
            yscale="log",
            xticks=range(2, 13),
        )
    assert heatmap is not None
    colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Fraction of active bus nodes")
    fig.suptitle("Bus activity across thresholds and checkpoints", fontsize=14, fontweight="bold")
    activity_threshold_figures = save_figure(fig, "node-activity", "node_activity_threshold_heatmap")

    primary_updates = tables[ORDER[0]][3]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    device_heatmap = None
    positive = primary_updates.loc[primary_updates.update_relative_l2 > 0, "update_relative_l2"]
    norm = LogNorm(vmin=float(positive.quantile(0.01)), vmax=float(positive.quantile(0.99)))
    for ax, node_type, title in zip(axes, ("bus", "gen"), ("Bus nodes", "Generator nodes")):
        matrix = (
            primary_updates[primary_updates.node_type == node_type]
            .groupby(["device_id", "correction_step"])["update_relative_l2"]
            .median()
            .unstack("correction_step")
            .sort_index()
        )
        device_heatmap = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            norm=norm,
        )
        ax.set(
            title=title,
            xlabel="Correction step",
            ylabel="Stable device index",
            xticks=np.arange(len(matrix.columns)),
            xticklabels=matrix.columns.astype(int),
        )
        tick_positions = np.linspace(0, len(matrix.index) - 1, min(7, len(matrix.index)), dtype=int)
        ax.set_yticks(tick_positions, matrix.index.to_numpy()[tick_positions])
    assert device_heatmap is not None
    colorbar = fig.colorbar(device_heatmap, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Median relative node update across 115 conditions")
    fig.suptitle(
        "Where state updates remain active · case118 Base seed 0",
        fontsize=14,
        fontweight="bold",
    )
    activity_device_figures = save_figure(fig, "node-activity", "node_activity_by_device")

    result = {
        "schema_version": 1,
        "design": {
            "seed": "case118 Base seed 0 versus seed 1 on the same 115 conditions",
            "model_size": "case118 Base versus Small at seed 0 on the same 115 conditions",
            "network": "case118 Base seed 0 versus case30 Base seed 0; each uses 115 fixed-topology conditions",
        },
        "variants": records,
        "statistics": {
            "bootstrap_samples": 2000,
            "resampling_unit": "operating condition",
            "late_steps": [7, 8, 9, 10, 11],
        },
        "node_activity": {
            "definition": (
                "per-node ||h_functional-h_in||_2 / (||h_in||_2 + 1e-12); "
                "h_functional is h_out for steps 2-11 and h_graph/readout for step 12"
            ),
            "steps": list(range(2, 13)),
            "thresholds": list(ACTIVITY_THRESHOLDS),
            "summary_csv": "results/node-activity-summary.csv",
        },
        "figures": (
            trajectory_figures
            + physical_figures
            + importance_figures
            + direct_importance_figures
            + activity_fraction_figures
            + activity_threshold_figures
            + activity_device_figures
        ),
    }
    (ROOT / "results/robustness-summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
