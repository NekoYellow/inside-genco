"""Explicit, validated interventions for the frozen GENCO forward pass."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping

import torch


VALID_STAGES = frozenset({"h_graph", "h_out"})
VALID_NODE_TYPES = frozenset({"bus", "gen"})


@dataclass(frozen=True)
class TensorEdit:
    """Add or replace one layer state with an exactly aligned tensor.

    Layer indices are zero based.  ``h_graph`` edits happen before the layer's
    readout and physics feedback; ``h_out`` edits happen after feedback and
    therefore affect only subsequent layers.
    """

    layer_index: int
    stage: str
    node_type: str
    tensor: torch.Tensor
    mode: str = "add"
    label: str = ""

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative and zero based")
        if self.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}")
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(f"node_type must be one of {sorted(VALID_NODE_TYPES)}")
        if self.mode not in {"add", "replace"}:
            raise ValueError("mode must be 'add' or 'replace'")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("tensor edit requires a torch.Tensor")


@dataclass(frozen=True)
class InterventionPlan:
    """All modifications to one forward pass.

    An empty plan, feedback scale 1, zero-valued additions, and replacement by
    the original tensor are mathematical no-ops.  Whole-step skipping is only
    valid for intermediate layers because layer 0 changes width and the final
    layer owns the returned prediction.
    """

    skip_layers: frozenset[int] = field(default_factory=frozenset)
    feedback_scales: Mapping[int, float] = field(default_factory=dict)
    operator_sources: Mapping[int, int] = field(default_factory=dict)
    edits: tuple[TensorEdit, ...] = ()

    def validate(self, num_layers: int) -> None:
        if num_layers < 2:
            raise ValueError("trace adapter expects at least two layers")
        for layer in self.skip_layers:
            if layer <= 0:
                raise ValueError("layer 0 cannot be skipped because it changes hidden width")
            if layer >= num_layers - 1:
                raise ValueError("the final layer cannot be skipped because it owns the returned readout")
        for layer, scale in self.feedback_scales.items():
            if not 0 <= layer < num_layers:
                raise ValueError(f"invalid feedback layer {layer}; expected 0..{num_layers - 1}")
            if not isinstance(scale, (int, float)) or not float("-inf") < float(scale) < float("inf"):
                raise ValueError("feedback scales must be finite numbers")
        for target, source in self.operator_sources.items():
            if not isinstance(target, int) or not isinstance(source, int):
                raise TypeError("operator source mappings require integer layer indices")
            if target <= 0 or source <= 0:
                raise ValueError("operator substitution excludes the width-changing layer 0")
            if target >= num_layers or source >= num_layers:
                raise ValueError(
                    f"invalid operator substitution {target} <- {source}; "
                    f"expected indices 1..{num_layers - 1}"
                )
            if target in self.skip_layers:
                raise ValueError("a skipped layer cannot also receive a substituted operator")
        seen: set[tuple[int, str, str]] = set()
        for edit in self.edits:
            if edit.layer_index >= num_layers:
                raise ValueError(f"invalid edit layer {edit.layer_index}; expected 0..{num_layers - 1}")
            key = (edit.layer_index, edit.stage, edit.node_type)
            if key in seen:
                raise ValueError(f"multiple edits target {key}; combine them explicitly")
            seen.add(key)

    def edit_for(self, layer: int, stage: str, node_type: str) -> TensorEdit | None:
        return next(
            (e for e in self.edits if (e.layer_index, e.stage, e.node_type) == (layer, stage, node_type)),
            None,
        )


def apply_tensor_edit(value: torch.Tensor, edit: TensorEdit | None) -> torch.Tensor:
    if edit is None:
        return value
    tensor = edit.tensor.to(device=value.device, dtype=value.dtype)
    if tensor.shape != value.shape:
        raise ValueError(
            f"{edit.label or 'tensor edit'} shape {tuple(tensor.shape)} does not match "
            f"{tuple(value.shape)} at layer {edit.layer_index} {edit.stage}/{edit.node_type}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("intervention tensor contains non-finite values")
    return value + tensor if edit.mode == "add" else tensor.clone()


def _row_seed(
    seed: int, scenario_id: int, device_id: int, node_type: str, width: int,
    layer_index: int | None,
) -> int:
    payload = f"genco-intervention-v2|{seed}|{scenario_id}|{layer_index}|{device_id}|{node_type}|{width}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def stable_relative_perturbation(
    reference: torch.Tensor,
    scenario_ids: torch.Tensor,
    device_ids: torch.Tensor,
    *,
    relative_l2: float,
    seed: int,
    node_type: str,
    layer_index: int | None = None,
) -> torch.Tensor:
    """Create per-scenario perturbations invariant to batch order/grouping.

    Random rows are keyed by ``(seed, scenario_id, stable device_id,
    node_type)``.  Within each scenario the concatenated perturbation is scaled
    to ``relative_l2 * ||reference||_2``.  This makes amplitude checks local to
    each graph instead of allowing large graphs to dominate a batch norm.
    """

    if reference.ndim != 2 or not reference.is_floating_point():
        raise ValueError("reference must be a two-dimensional floating tensor")
    if relative_l2 < 0 or not torch.isfinite(torch.tensor(relative_l2)):
        raise ValueError("relative_l2 must be finite and non-negative")
    if node_type not in VALID_NODE_TYPES:
        raise ValueError(f"node_type must be one of {sorted(VALID_NODE_TYPES)}")
    if layer_index is not None and layer_index < 0:
        raise ValueError("layer_index must be non-negative when provided")
    scenario_ids = torch.as_tensor(scenario_ids).detach().cpu().reshape(-1)
    device_ids = torch.as_tensor(device_ids).detach().cpu().reshape(-1)
    if len(scenario_ids) != len(reference) or len(device_ids) != len(reference):
        raise ValueError("stable IDs must have one row per node")
    pairs = list(zip(scenario_ids.tolist(), device_ids.tolist()))
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate (scenario_id, device_id) rows make alignment ambiguous")

    direction = torch.empty(reference.shape, dtype=torch.float64, device="cpu")
    for row, (scenario_id, device_id) in enumerate(pairs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_row_seed(
            seed, int(scenario_id), int(device_id), node_type, reference.shape[1], layer_index,
        ))
        direction[row] = torch.randn(reference.shape[1], generator=generator, dtype=torch.float64)

    reference_cpu = reference.detach().cpu().double()
    for scenario_id in sorted(set(scenario_ids.tolist())):
        mask = scenario_ids == scenario_id
        target_norm = float(relative_l2) * torch.linalg.vector_norm(reference_cpu[mask])
        direction_norm = torch.linalg.vector_norm(direction[mask])
        if direction_norm == 0:
            raise RuntimeError("deterministic perturbation direction unexpectedly has zero norm")
        direction[mask] *= target_norm / direction_norm
    return direction.to(device=reference.device, dtype=reference.dtype)
