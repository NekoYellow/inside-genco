"""Similarity measures with explicit row-identity validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class CKAResult:
    value: float
    valid: bool
    reason: str | None
    rows: int
    x_features: int
    y_features: int


def _validate_ids(x_ids: Sequence[object] | None, y_ids: Sequence[object] | None, rows: int) -> None:
    if (x_ids is None) != (y_ids is None):
        raise ValueError("provide row IDs for both representations or neither")
    if x_ids is None:
        return
    x_ids, y_ids = list(x_ids), list(y_ids)
    if len(x_ids) != rows or len(y_ids) != rows:
        raise ValueError("row ID count does not match representation rows")
    if len(set(x_ids)) != rows or len(set(y_ids)) != rows:
        raise ValueError("row IDs must be unique")
    if x_ids != y_ids:
        raise ValueError("representation rows are not aligned by stable ID")


def linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    x_ids: Sequence[object] | None = None,
    y_ids: Sequence[object] | None = None,
    eps: float = 1e-12,
) -> CKAResult:
    """Biased linear CKA in float64, with honest degenerate-input handling."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of rows")
    if x.shape[0] < 2:
        return CKAResult(float("nan"), False, "fewer than two aligned rows", x.shape[0], x.shape[1], y.shape[1])
    _validate_ids(x_ids, y_ids, x.shape[0])
    x = x.detach().double()
    y = y.detach().double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return CKAResult(float("nan"), False, "non-finite representation", x.shape[0], x.shape[1], y.shape[1])
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.T @ y) ** 2
    x_norm = torch.linalg.matrix_norm(x.T @ x)
    y_norm = torch.linalg.matrix_norm(y.T @ y)
    denom = x_norm * y_norm
    if not torch.isfinite(denom) or denom <= eps:
        return CKAResult(float("nan"), False, "zero or numerically degenerate centered variance", x.shape[0], x.shape[1], y.shape[1])
    value = float((cross / denom).clamp(0.0, 1.0).cpu())
    return CKAResult(value, True, None, x.shape[0], x.shape[1], y.shape[1])
