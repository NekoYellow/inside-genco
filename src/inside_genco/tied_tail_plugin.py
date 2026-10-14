"""GridFM plugin registrations for trainable recurrent-tail GENCO controls."""

from __future__ import annotations

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous

from .tied_tail import tie_recurrent_tail


@MODELS_REGISTRY.register("GNS_heterogeneous_tied_tail")
class GNSHeterogeneousTiedTail(GNS_heterogeneous):
    """Independent stem followed by one recurrent graph+normalization operator."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.sharing_report = tie_recurrent_tail(self, share_norms=True)


@MODELS_REGISTRY.register("GNS_heterogeneous_shared_graph_step_norms")
class GNSHeterogeneousSharedGraphStepNorms(GNS_heterogeneous):
    """Independent stem and per-step norms, with one recurrent graph operator."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.sharing_report = tie_recurrent_tail(self, share_norms=False)
