"""Trajectory metrics with explicit identities and finite-value handling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch

from .similarity import CKAResult


@dataclass(frozen=True)
class LocalStepMetrics:
    update_relative_l2: float | None
    row_cosine_mean: float | None
    feedback_relative_l2: float | None
    finite: bool


def _finite_scalar(value: torch.Tensor) -> float | None:
    value = value.detach().double()
    return float(value.cpu()) if value.numel() == 1 and torch.isfinite(value) else None


def local_step_metrics(
    h_in: torch.Tensor,
    h_out: torch.Tensor,
    *,
    h_graph: torch.Tensor | None = None,
    feedback: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> LocalStepMetrics:
    """Return adapter-boundary metrics without silently inventing alignments.

    Relative update and row cosine need the same feature coordinates.  They are
    therefore unavailable when a width-changing correction step is observed.
    Feedback magnitude is measured against the graph state that receives it.
    """

    tensors = [h_in, h_out] + ([] if h_graph is None else [h_graph]) + ([] if feedback is None else [feedback])
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("state metrics require two-dimensional tensors")
    if any(not torch.isfinite(value).all() for value in tensors):
        return LocalStepMetrics(None, None, None, False)

    update = cosine = None
    if h_in.shape == h_out.shape:
        left, right = h_in.detach().double(), h_out.detach().double()
        update = _finite_scalar(torch.linalg.vector_norm(right - left) / (torch.linalg.vector_norm(left) + eps))
        row_denom = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
        valid = row_denom > eps
        if valid.any():
            cosine = _finite_scalar(((left * right).sum(dim=1)[valid] / row_denom[valid]).mean())

    feedback_ratio = None
    if h_graph is not None and feedback is not None:
        if h_graph.shape != feedback.shape:
            raise ValueError("feedback and graph state must have identical shapes")
        feedback_ratio = _finite_scalar(
            torch.linalg.vector_norm(feedback.detach().double())
            / (torch.linalg.vector_norm(h_graph.detach().double()) + eps)
        )
    return LocalStepMetrics(update, cosine, feedback_ratio, True)


def node_relative_l2_updates(
    h_in: torch.Tensor,
    h_out: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return one relative L2 update magnitude per aligned node row."""

    if h_in.ndim != 2 or h_out.ndim != 2:
        raise ValueError("node update metrics require two-dimensional tensors")
    if h_in.shape != h_out.shape:
        raise ValueError("node update metrics require identical input and output shapes")
    if not torch.isfinite(h_in).all() or not torch.isfinite(h_out).all():
        raise ValueError("node update metrics require finite tensors")
    left = h_in.detach().double()
    right = h_out.detach().double()
    return torch.linalg.vector_norm(right - left, dim=1) / (
        torch.linalg.vector_norm(left, dim=1) + eps
    )


def first_threshold_crossing(
    values: Sequence[float], thresholds: Iterable[float], *, one_based_steps: Sequence[int] | None = None
) -> list[dict[str, float | int | bool | None]]:
    """Find the first observed step at or below each fixed threshold."""

    steps = list(range(1, len(values) + 1)) if one_based_steps is None else list(one_based_steps)
    if len(steps) != len(values):
        raise ValueError("step labels and values must have the same length")
    rows = []
    for threshold in thresholds:
        crossing = next((step for step, value in zip(steps, values) if torch.isfinite(torch.tensor(value)) and value <= threshold), None)
        rows.append({"threshold": float(threshold), "reached": crossing is not None, "first_step": crossing})
    return rows


class LinearCKAAccumulator:
    """Feature-covariance linear CKA using mergeable sufficient statistics.

    This never constructs an ``N x N`` Gram matrix.  Stable row identities are
    checked per update and globally, so a reordered or duplicated row cannot be
    accepted accidentally.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.x_features: int | None = None
        self.y_features: int | None = None
        self.sum_x: torch.Tensor | None = None
        self.sum_y: torch.Tensor | None = None
        self.xtx: torch.Tensor | None = None
        self.yty: torch.Tensor | None = None
        self.xty: torch.Tensor | None = None
        self._ids: set[object] = set()

    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        x_ids: Sequence[object],
        y_ids: Sequence[object],
    ) -> None:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("CKA chunks must be aligned two-dimensional tensors")
        left_ids, right_ids = list(x_ids), list(y_ids)
        if len(left_ids) != len(x) or left_ids != right_ids:
            raise ValueError("CKA chunk rows are not aligned by stable ID")
        if len(set(left_ids)) != len(left_ids) or self._ids.intersection(left_ids):
            raise ValueError("CKA stable row IDs must be globally unique")
        if self.x_features is not None and (x.shape[1] != self.x_features or y.shape[1] != self.y_features):
            raise ValueError("CKA feature widths changed between chunks")
        x, y = x.detach().cpu().double(), y.detach().cpu().double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("CKA chunks contain non-finite values")
        if self.x_features is None:
            self.x_features, self.y_features = x.shape[1], y.shape[1]
            self.sum_x = torch.zeros(self.x_features, dtype=torch.float64)
            self.sum_y = torch.zeros(self.y_features, dtype=torch.float64)
            self.xtx = torch.zeros((self.x_features, self.x_features), dtype=torch.float64)
            self.yty = torch.zeros((self.y_features, self.y_features), dtype=torch.float64)
            self.xty = torch.zeros((self.x_features, self.y_features), dtype=torch.float64)
        self.rows += len(x)
        self._ids.update(left_ids)
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.xtx += x.T @ x
        self.yty += y.T @ y
        self.xty += x.T @ y

    def result(self, *, eps: float = 1e-12) -> CKAResult:
        x_features, y_features = self.x_features or 0, self.y_features or 0
        if self.rows < 2:
            return CKAResult(float("nan"), False, "fewer than two aligned rows", self.rows, x_features, y_features)
        centered_x = self.xtx - torch.outer(self.sum_x, self.sum_x) / self.rows
        centered_y = self.yty - torch.outer(self.sum_y, self.sum_y) / self.rows
        centered_xy = self.xty - torch.outer(self.sum_x, self.sum_y) / self.rows
        numerator = torch.linalg.matrix_norm(centered_xy) ** 2
        denominator = torch.linalg.matrix_norm(centered_x) * torch.linalg.matrix_norm(centered_y)
        if not torch.isfinite(denominator) or denominator <= eps:
            return CKAResult(float("nan"), False, "zero or numerically degenerate centered variance", self.rows, x_features, y_features)
        value = float((numerator / denominator).clamp(0.0, 1.0))
        return CKAResult(value, True, None, self.rows, x_features, y_features)


def streaming_linear_cka(
    chunks: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[object], Sequence[object]]]
) -> CKAResult:
    accumulator = LinearCKAAccumulator()
    for x, y, x_ids, y_ids in chunks:
        accumulator.update(x, y, x_ids=x_ids, y_ids=y_ids)
    return accumulator.result()


def cka_record(result: CKAResult) -> dict[str, object]:
    """A small helper used by the runner and unit tests."""

    return asdict(result)
