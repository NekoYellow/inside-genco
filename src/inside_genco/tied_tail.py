"""Recurrent-tail parameter sharing for GENCO-compatible models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class SharingReport:
    mode: str
    corrections: int
    shared_corrections: tuple[int, ...]
    unique_parameters_before: int
    unique_parameters_after: int

    def to_dict(self) -> dict:
        return asdict(self)


def count_unique_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters once even when a module is referenced at several steps."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def tie_recurrent_tail(model: nn.Module, *, share_norms: bool) -> SharingReport:
    """Share correction-2's graph operator across corrections 2..N in place.

    Correction 1 stays independent because its input width differs.  When
    ``share_norms`` is false, every correction retains its own bus and generator
    LayerNorm, which isolates graph-operator sharing from step-specific affine
    normalization.
    """

    for name in ("layers", "norms_bus", "norms_gen"):
        if not hasattr(model, name) or not isinstance(getattr(model, name), nn.ModuleList):
            raise TypeError(f"model.{name} must be a torch.nn.ModuleList")
    corrections = len(model.layers)
    if corrections < 2:
        raise ValueError("recurrent-tail sharing requires at least two corrections")
    if len(model.norms_bus) != corrections or len(model.norms_gen) != corrections:
        raise ValueError("graph and normalization module lists must have equal lengths")

    before = count_unique_parameters(model)
    graph_source = model.layers[1]
    bus_norm_source = model.norms_bus[1]
    gen_norm_source = model.norms_gen[1]
    for index in range(2, corrections):
        model.layers[index] = graph_source
        if share_norms:
            model.norms_bus[index] = bus_norm_source
            model.norms_gen[index] = gen_norm_source
    after = count_unique_parameters(model)
    mode = "strict_recurrent_tail" if share_norms else "shared_graph_step_norms"
    return SharingReport(
        mode=mode,
        corrections=corrections,
        shared_corrections=tuple(range(2, corrections + 1)),
        unique_parameters_before=before,
        unique_parameters_after=after,
    )
