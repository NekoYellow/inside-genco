#!/usr/bin/env python3
"""Summarize physical refinement, CKA, node activity, and bypass evidence."""

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


ORDER = (
    "case118_base_seed0",
    "case118_base_seed1",
    "case118_small_seed0",
    "case30_base_seed0",
)
COLORS = ("#264653", "#2a9d8f", "#e76f51", "#7b2cbf")
ACTIVITY_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def summarize_node_activity(updates: pd.DataFrame) -> pd.DataFrame:
    """Average threshold activity over conditions, not over pooled node rows."""
    keys = ["scenario_id", "node_type", "device_id", "correction_step"]
    if updates.duplicated(keys).any():
        raise ValueError("duplicate stable node-update identities")
    rows = []
    for kind in ("graph", "functional"):
        column = f"{kind}_update_relative_l2"
        if not np.isfinite(updates[column]).all() or (updates[column] < 0).any():
            raise ValueError("node updates must be finite and nonnegative")
        for (node, step), selected in updates.groupby(["node_type", "correction_step"]):
            for threshold in ACTIVITY_THRESHOLDS:
                fractions = (
                    selected.assign(active=selected[column] > threshold)
                    .groupby("scenario_id").active.mean()
                )
                rows.append({
                    "kind": kind,
                    "node_type": node,
                    "correction_step": int(step),
                    "threshold": threshold,
                    "active_fraction": float(fractions.mean()),
                    "condition_active_fraction_median": float(fractions.median()),
                    "condition_active_fraction_p05": float(fractions.quantile(0.05)),
                    "condition_active_fraction_p95": float(fractions.quantile(0.95)),
                    "conditions": len(fractions),
                    "stable_nodes": selected.device_id.nunique(),
                })
    return pd.DataFrame(rows)


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
    activity_tables = []
    for name in ORDER:
        directory = ROOT / "outputs/trajectory-analysis" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete robustness run: {name}")
        summary = json.loads((directory / "summary.json").read_text())
        steps = pd.read_csv(directory / "step_summary.csv")
        association = pd.read_csv(directory / "local_change_vs_importance.csv")
        physical_metrics = pd.read_csv(directory / "physical_metric_summary.csv")
        physical_updates = pd.read_csv(directory / "physical_update_summary.csv")
        node_updates = pd.read_csv(directory / "node_updates.csv")
        expected_steps = set(range(2, 13))
        for node in ("bus", "gen"):
            selected = node_updates[node_updates.node_type == node]
            if set(selected.correction_step) != expected_steps:
                raise ValueError(f"incomplete node-update steps: {name}/{node}")
            identities = selected.groupby("correction_step").apply(
                lambda frame: set(zip(frame.scenario_id, frame.device_id)), include_groups=False
            )
            if any(keys != identities.iloc[0] for keys in identities):
                raise ValueError(f"node-update identities change across steps: {name}/{node}")
            if selected.scenario_id.nunique() != summary["conditions"]:
                raise ValueError(f"incomplete node-update conditions: {name}/{node}")
        last = node_updates[node_updates.correction_step == 12]
        if set(last.functional_endpoint) != {"h_graph"} or not np.array_equal(
            last.graph_update_relative_l2.to_numpy(), last.functional_update_relative_l2.to_numpy()
        ):
            raise ValueError(f"final node activity includes unconsumed feedback: {name}")
        if set(node_updates.loc[node_updates.correction_step < 12, "functional_endpoint"]) != {"h_out"}:
            raise ValueError(f"invalid propagated node endpoint: {name}")
        activity = summarize_node_activity(node_updates)
        activity.insert(0, "label", config[name]["label"])
        activity.insert(0, "variant", name)
        activity_tables.append(activity)
        if set(physical_metrics.correction_step.unique()) != set(range(1, 13)):
            raise ValueError(f"physical-metric steps are incomplete: {name}")
        if set(physical_updates.correction_step.unique()) != set(range(2, 13)):
            raise ValueError(f"physical-update steps are incomplete: {name}")
        tables[name] = (summary, steps, association)
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
                "smallest_local_change_step": summary["smallest_local_change_step"],
                "least_ablation_damage_step": summary["least_ablation_damage_step"],
                "model_state_sha256": summary["model_state_sha256"],
            }
        )
    frame = pd.DataFrame(records)
    activity = pd.concat(activity_tables, ignore_index=True)
    activity.to_csv(ROOT / "results/node-activity-summary.csv", index=False)
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
        _, steps, _ = tables[row["variant"]]
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

    primary_cka = pd.read_csv(
        ROOT / "outputs/trajectory-analysis" / ORDER[0] / "cka.csv"
    )
    primary_cka = primary_cka[primary_cka.kind == "cross_step_h_graph"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4), constrained_layout=True)
    heatmap = None
    for row_index, node_type in enumerate(("bus", "gen")):
        for column_index, centered in enumerate((False, True)):
            selected = primary_cka[
                (primary_cka.node_type == node_type)
                & (primary_cka.node_centered.astype(bool) == centered)
            ]
            matrix = selected.pivot(
                index="left_step", columns="right_step", values="value"
            ).sort_index().sort_index(axis=1)
            if matrix.shape != (12, 12) or matrix.isna().any().any():
                raise ValueError("primary cross-correction CKA matrix is incomplete")
            ax = axes[row_index, column_index]
            heatmap = ax.imshow(matrix.to_numpy(float), vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set(
                title=f"{node_type.title()} · {'node-centered' if centered else 'raw'}",
                xlabel="Correction step",
                ylabel="Correction step",
                xticks=np.arange(12),
                xticklabels=range(1, 13),
                yticks=np.arange(12),
                yticklabels=range(1, 13),
            )
    assert heatmap is not None
    colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Linear CKA")
    cka_figures = save_figure(fig, "representations", "cka_multiview_heatmaps")

    primary_activity = activity[(activity.variant == ORDER[0]) & (activity.kind == "graph")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    threshold_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ACTIVITY_THRESHOLDS)))
    for ax, node, title in zip(axes, ("bus", "gen"), ("Bus graph updates", "Generator graph updates")):
        for threshold, color in zip(ACTIVITY_THRESHOLDS, threshold_colors):
            selected = primary_activity[(primary_activity.node_type == node) & (primary_activity.threshold == threshold)].sort_values("correction_step")
            ax.plot(selected.correction_step, selected.active_fraction, marker="o", color=color, label=fr"$\tau={threshold:g}$")
        ax.set(title=title, xlabel="Correction step", ylabel="Mean fraction of nodes above threshold", xticks=range(2, 13), ylim=(-0.03, 1.03))
        ax.grid(color="#e5e7eb", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.suptitle("Node activity within stable geometry · case118 Base seed 0")
    activity_figures = save_figure(fig, "node-activity", "node_activity_fraction")

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
        _, _, association = tables[row["variant"]]
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
            "definition": "per-node ||h_end-h_in||_2 / (||h_in||_2 + 1e-12)",
            "main_figure_endpoint": "h_graph at every step (before physics feedback)",
            "functional_endpoint": "h_out at steps 2-11; h_graph at step 12",
            "steps": list(range(2, 13)), "thresholds": list(ACTIVITY_THRESHOLDS),
            "aggregation": "mean per-condition fraction of nodes strictly above threshold",
            "summary_csv": "results/node-activity-summary.csv",
        },
        "figures": trajectory_figures + cka_figures + activity_figures + physical_figures + importance_figures,
    }
    (ROOT / "results/robustness-summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
