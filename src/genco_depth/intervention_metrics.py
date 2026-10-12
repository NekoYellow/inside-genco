"""Intervention metrics and paired, cluster-aware summaries.

The functions in this module deliberately know nothing about GENCO's model
implementation.  They operate on aligned tensors or per-scenario rows so the
scientific definitions can be unit tested without loading the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PropagationMetrics:
    absolute_l2: float | None
    relative_l2: float | None
    amplification: float | None
    finite: bool
    reason: str | None = None


def propagation_metrics(
    baseline: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    initial_relative_l2: float | None,
    eps: float = 1e-12,
) -> PropagationMetrics:
    """Compute d, D and A for one aligned scenario and one trace stage.

    Undefined amplification is represented by ``None`` rather than zero.  A
    non-finite tensor similarly remains visible to downstream failure counts.
    """

    if baseline.shape != perturbed.shape or baseline.ndim != 2:
        raise ValueError("propagation tensors must be aligned two-dimensional arrays")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    left, right = baseline.detach().double(), perturbed.detach().double()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        return PropagationMetrics(None, None, None, False, "non-finite state")
    absolute = torch.linalg.vector_norm(right - left)
    relative = absolute / (torch.linalg.vector_norm(left) + eps)
    absolute_value, relative_value = float(absolute.cpu()), float(relative.cpu())
    if initial_relative_l2 is None or not np.isfinite(initial_relative_l2) or initial_relative_l2 <= eps:
        return PropagationMetrics(absolute_value, relative_value, None, True, "undefined initial offset")
    amplification = relative_value / float(initial_relative_l2)
    return PropagationMetrics(absolute_value, relative_value, amplification, True)


def interaction_effect(off_zero: float, off_delta: float, on_zero: float, on_delta: float) -> float | None:
    """Return I_Q = (off_delta-off_zero) - (on_delta-on_zero)."""

    values = np.asarray([off_zero, off_delta, on_zero, on_delta], dtype=float)
    if not np.isfinite(values).all():
        return None
    return float((off_delta - off_zero) - (on_delta - on_zero))


def cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    samples: int,
    seed: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Bootstrap a paired effect mean by resampling whole load groups.

    Multiple rows from one group are always selected together.  Non-finite
    values are retained in the input count but excluded from the estimand and
    each replicate; their rate is reported separately.
    """

    array = np.asarray(values, dtype=float)
    cluster_array = np.asarray(clusters)
    if array.ndim != 1 or cluster_array.ndim != 1 or len(array) != len(cluster_array) or not len(array):
        raise ValueError("values and clusters must be non-empty aligned vectors")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must lie in (0, 1)")
    finite = np.isfinite(array)
    transformed = transform(array.copy()) if transform else array.copy()
    if transformed.shape != array.shape:
        raise ValueError("transform must preserve shape")
    valid = finite & np.isfinite(transformed)
    unique = np.unique(cluster_array)
    if not valid.any():
        return {
            "n": int(len(array)), "clusters": int(len(unique)), "finite_n": 0,
            "nonfinite_rate": 1.0, "mean": None, "ci_low": None, "ci_high": None,
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    by_cluster = {cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique}
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([by_cluster[item] for item in selected])
        replicate = transformed[positions]
        replicate = replicate[np.isfinite(replicate)]
        estimates[index] = np.mean(replicate) if len(replicate) else np.nan
    estimates = estimates[np.isfinite(estimates)]
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": int(len(array)),
        "clusters": int(len(unique)),
        "finite_n": int(valid.sum()),
        "nonfinite_rate": float(1.0 - finite.mean()),
        "mean": float(np.mean(transformed[valid])),
        "ci_low": float(np.quantile(estimates, alpha)) if len(estimates) else None,
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)) if len(estimates) else None,
    }


def summarize_paired_effect(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Summarize signed and conservative positive damage for one condition."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    signed = cluster_bootstrap_mean(array, clusters, samples=samples, seed=seed)
    positive = cluster_bootstrap_mean(
        array, clusters, samples=samples, seed=seed + 1,
        transform=lambda item: np.maximum(item, 0.0),
    )
    result: dict[str, float | int | None] = {
        **{f"signed_{key}": value for key, value in signed.items()},
        "positive_mean": positive["mean"],
        "positive_ci_low": positive["ci_low"],
        "positive_ci_high": positive["ci_high"],
        "median": float(np.median(finite)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
    }
    return result


def rank_steps(rows: Iterable[dict], *, score_key: str, step_key: str = "correction_step_number") -> list[dict]:
    """Return a deterministic ascending step ranking, keeping non-finite last."""

    copied = [dict(row) for row in rows]
    return sorted(
        copied,
        key=lambda row: (
            not np.isfinite(float(row.get(score_key, np.nan))),
            float(row.get(score_key, np.inf)),
            int(row[step_key]),
        ),
    )
