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
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="case118_base_seed0")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    directory = (
        args.input or ROOT / f"outputs/operator-substitution/{args.variant}"
    ).resolve()
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
