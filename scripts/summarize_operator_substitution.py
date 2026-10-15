#!/usr/bin/env python3
"""Render the primary GENCO cross-correction operator-substitution matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
import pandas as pd


def summarize_components(variant: str, directory: Path) -> None:
    modes = (("joint", "Graph + LayerNorm"), ("graph", "Graph only"), ("norm", "LayerNorm only"))
    matrices, tables, manifests, records = [], [], [], []
    reference = None
    reference_baseline = None
    for mode, label in modes:
        source = directory if mode == "joint" else directory.with_name(f"{variant}-{mode}")
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest.get("component", "joint") != mode:
            raise ValueError(f"incomplete or mismatched component run: {source}")
        identity = (manifest["selection"], manifest["model_state_sha256_before"], manifest["data_lock_sha256"])
        if reference is not None and identity != reference:
            raise ValueError("component experiments use different inputs")
        reference = identity
        conditions = pd.read_csv(source / "condition_level.csv")
        if conditions.duplicated(["scenario_id", "target_step", "source_step"]).any():
            raise ValueError("duplicate condition-level substitution rows")
        diagonal = conditions[conditions.target_step == conditions.source_step]
        baseline = diagonal[diagonal.target_step == 2].set_index("scenario_id").r_inf_pu_100MVA.sort_index()
        if len(baseline) != 115:
            raise ValueError("incomplete baseline conditions")
        if reference_baseline is not None:
            if not baseline.index.equals(reference_baseline.index) or not np.allclose(baseline, reference_baseline, rtol=1e-4, atol=1e-6):
                raise ValueError("unmodified baselines differ across component runs")
        reference_baseline = baseline
        if manifest["model_state_sha256_after"] != manifest["model_state_sha256_before"]:
            raise ValueError("checkpoint changed during component experiment")
        table = pd.read_csv(source / "matrix_summary.csv")
        matrix = table.pivot(index="target_step", columns="source_step", values="delta_final_r_inf_mean").sort_index().sort_index(axis=1)
        if list(matrix.index) != list(range(2, 13)) or list(matrix.columns) != list(range(2, 13)) or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("expected a finite complete corrections 2–12 matrix")
        if not (table.conditions == 115).all() or not np.allclose(np.diag(matrix), 0, atol=1e-12):
            raise ValueError("condition count or diagonal control differs")
        table["component"] = mode
        tables.append(table)
        matrices.append(matrix.to_numpy())
        manifests.append(manifest)
        off = table[table.target_step != table.source_step]
        late = off[(off.target_step >= 7) & (off.source_step >= 7)]
        best = table.loc[table.groupby("target_step").final_r_inf_mean.idxmin()]
        records.append(dict(component=mode, off_diagonal_cells=len(off),
                            positive_cells=int((off.delta_final_r_inf_mean > 0).sum()),
                            diagonal_best_targets=int((best.target_step == best.source_step).sum()),
                            min=float(off.delta_final_r_inf_mean.min()),
                            median=float(off.delta_final_r_inf_mean.median()),
                            mean=float(off.delta_final_r_inf_mean.mean()),
                            max=float(off.delta_final_r_inf_mean.max()),
                            late_median=float(late.delta_final_r_inf_mean.median())))
    mask = np.eye(11, dtype=bool)
    values = np.concatenate([m[~mask] for m in matrices])
    if np.all(values > 0):
        norm, cmap = LogNorm(vmin=values.min(), vmax=values.max()), "Reds"
    else:
        bound = np.abs(values).max()
        norm, cmap = SymLogNorm(linthresh=1e-4, vmin=-bound, vmax=bound), "RdBu_r"
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, (_, label) in zip(axes, matrices, modes):
        image = ax.imshow(np.ma.masked_where(mask, values), cmap=colormap, norm=norm)
        ax.set(title=label, xlabel="Source correction step", ylabel="Target correction step",
               xticks=np.arange(11), xticklabels=range(2, 13), yticks=np.arange(11), yticklabels=range(2, 13))
    fig.colorbar(image, ax=axes, shrink=.85, label=r"Mean final $\Delta R_\infty$ (p.u., 100 MVA)", pad=.02)
    output = ROOT / "figures/operators/operator_substitution_matrix.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(ROOT / "results/operator-substitution-components.csv", index=False)
    paired = dict(graph_greater_than_norm_cells=int((matrices[1][~mask] > matrices[2][~mask]).sum()),
                  graph_greater_than_joint_cells=int((matrices[1][~mask] > matrices[0][~mask]).sum()),
                  norm_greater_than_joint_cells=int((matrices[2][~mask] > matrices[0][~mask]).sum()))
    late_mask = ~mask & (np.arange(2, 13)[:, None] >= 7) & (np.arange(2, 13)[None, :] >= 7)
    paired["late_cells"] = int(late_mask.sum())
    paired["late_joint_less_than_both_components_cells"] = int(
        ((matrices[0][late_mask] < matrices[1][late_mask]) &
         (matrices[0][late_mask] < matrices[2][late_mask])).sum())
    result = dict(variant=variant, conditions=115, components=records, paired_comparison=paired, source_manifests=manifests,
                  figure=str(output.relative_to(ROOT)), color_scale="log" if cmap == "Reds" else "symmetric_log")
    (ROOT / "results/operator-substitution-components.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="case118_base_seed0")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--components", action="store_true", help="compare joint, graph-only, and LN-only interventions")
    args = parser.parse_args()
    directory = (
        args.input or ROOT / f"outputs/operator-substitution/{args.variant}"
    ).resolve()
    if args.components:
        summarize_components(args.variant, directory)
        return
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete operator-substitution artifact: {directory}")
    summary = pd.read_csv(directory / "matrix_summary.csv")
    targets = sorted(summary["target_step"].unique().tolist())
    sources = sorted(summary["source_step"].unique().tolist())
    expected_pairs = len(targets) * len(sources)
    if len(summary) != expected_pairs:
        raise ValueError(
            f"operator matrix has {len(summary)} cells; expected {expected_pairs}"
        )
    matrix = summary.pivot(
        index="target_step",
        columns="source_step",
        values="delta_final_r_inf_mean",
    ).sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("operator-substitution matrix is incomplete")
    summary.to_csv(ROOT / "results/operator-substitution-matrix.csv", index=False)

    values = matrix.to_numpy(float)
    diagonal_mask = np.eye(*values.shape, dtype=bool)
    off_diagonal = values[~diagonal_mask]
    if not len(off_diagonal) or np.any(off_diagonal <= 0):
        raise ValueError("expected strictly positive off-diagonal mean damage")
    display_values = np.ma.masked_where(diagonal_mask, values)
    colormap = plt.get_cmap("Reds").copy()
    colormap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = ax.imshow(
        display_values,
        cmap=colormap,
        norm=LogNorm(vmin=float(off_diagonal.min()), vmax=float(off_diagonal.max())),
        aspect="equal",
    )
    ax.set(
        xlabel="Source graph operator",
        ylabel="Target correction step",
        xticks=np.arange(len(matrix.columns)),
        xticklabels=matrix.columns.astype(int),
        yticks=np.arange(len(matrix.index)),
        yticklabels=matrix.index.astype(int),
        title="Cross-correction operator substitution",
    )
    for position in range(min(len(matrix.index), len(matrix.columns))):
        ax.add_patch(
            plt.Rectangle(
                (position - 0.5, position - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label(r"Mean final $\Delta R_\infty$ (p.u., 100 MVA)")
    output = ROOT / "figures/operators/operator_substitution_matrix.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best = summary.loc[
        summary.groupby("target_step")["final_r_inf_mean"].idxmin(),
        ["target_step", "source_step", "final_r_inf_mean"],
    ].sort_values("target_step")
    best.to_csv(ROOT / "results/operator-substitution-best.csv", index=False)
    result = {
        "schema_version": 1,
        "variant": args.variant,
        "conditions": int(summary.conditions.min()),
        "targets": matrix.index.astype(int).tolist(),
        "sources": matrix.columns.astype(int).tolist(),
        "diagonal_best_targets": int(
            (best.target_step == best.source_step).sum()
        ),
        "off_diagonal_cells": int(len(off_diagonal)),
        "off_diagonal_positive_cells": int((off_diagonal > 0).sum()),
        "off_diagonal_delta_final_r_inf": {
            "min": float(off_diagonal.min()),
            "median": float(np.median(off_diagonal)),
            "mean": float(off_diagonal.mean()),
            "max": float(off_diagonal.max()),
        },
        "figure": str(output.relative_to(ROOT)),
        "source_artifact": str(directory.relative_to(ROOT)),
    }
    (ROOT / "results/operator-substitution-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
