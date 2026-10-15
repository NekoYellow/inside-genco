"""Validated correction-level interventions for the frozen GENCO forward pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping



@dataclass(frozen=True)
class InterventionPlan:
    """All modifications to one forward pass.

    Whole-step skipping is only valid for intermediate layers because layer 0
    changes width and the final layer owns the returned prediction.
    """

    skip_layers: frozenset[int] = field(default_factory=frozenset)
    operator_sources: Mapping[int, int] = field(default_factory=dict)
    operator_component: str = "joint"

    def validate(self, num_layers: int) -> None:
        if self.operator_component not in ("joint", "graph", "norm"):
            raise ValueError("operator component must be joint, graph, or norm")
        if num_layers < 2:
            raise ValueError("trace adapter expects at least two layers")
        for layer in self.skip_layers:
            if layer <= 0:
                raise ValueError("layer 0 cannot be skipped because it changes hidden width")
            if layer >= num_layers - 1:
                raise ValueError("the final layer cannot be skipped because it owns the returned readout")
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
