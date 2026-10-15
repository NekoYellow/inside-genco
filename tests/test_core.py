"""Fast unit tests that do not require the multi-gigabyte research assets."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inside_genco.interventions import InterventionPlan
from inside_genco.similarity import linear_cka
from inside_genco.tied_tail import count_unique_parameters, tie_recurrent_tail
from inside_genco.trace import _operator_modules


class SimilarityTests(unittest.TestCase):
    def test_self_and_orthogonal_invariance(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(40, 8, generator=generator)
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        ids = [("scenario", i) for i in range(len(x))]
        self.assertAlmostEqual(linear_cka(x, x, x_ids=ids, y_ids=ids).value, 1.0, places=12)
        self.assertAlmostEqual(linear_cka(x, x @ q, x_ids=ids, y_ids=ids).value, 1.0, places=12)

    def test_alignment_and_degeneracy_are_explicit(self):
        x = torch.arange(24.0).reshape(6, 4)
        ids = list(range(6))
        with self.assertRaisesRegex(ValueError, "not aligned"):
            linear_cka(x, x.flip(0), x_ids=ids, y_ids=ids[::-1])
        result = linear_cka(torch.ones_like(x), x, x_ids=ids, y_ids=ids)
        self.assertFalse(result.valid)
        self.assertTrue(torch.isnan(torch.tensor(result.value)))
        self.assertIn("degenerate", result.reason)


class InterventionTests(unittest.TestCase):
    def test_plan_rejects_invalid_bypasses(self):
        with self.assertRaisesRegex(ValueError, "layer 0"):
            InterventionPlan(skip_layers=frozenset({0})).validate(12)
        with self.assertRaisesRegex(ValueError, "final layer"):
            InterventionPlan(skip_layers=frozenset({11})).validate(12)

    def test_operator_substitution_rejects_incompatible_or_conflicting_layers(self):
        InterventionPlan(operator_sources={1: 2, 11: 10}).validate(12)
        with self.assertRaisesRegex(ValueError, "width-changing layer 0"):
            InterventionPlan(operator_sources={0: 2}).validate(12)
        with self.assertRaisesRegex(ValueError, "width-changing layer 0"):
            InterventionPlan(operator_sources={2: 0}).validate(12)
        with self.assertRaisesRegex(ValueError, "invalid operator substitution"):
            InterventionPlan(operator_sources={2: 12}).validate(12)
        with self.assertRaisesRegex(ValueError, "skipped layer"):
            InterventionPlan(
                skip_layers=frozenset({2}), operator_sources={2: 3}
            ).validate(12)

    def test_operator_substitution_routes_graph_and_both_norms_together(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        model.norms_bus = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(4)]
        )
        model.norms_gen = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(4)]
        )

        graph, bus_norm, gen_norm, source = _operator_modules(
            model, 1, InterventionPlan(operator_sources={1: 3})
        )

        self.assertEqual(source, 3)
        self.assertIs(graph, model.layers[3])
        self.assertIs(bus_norm, model.norms_bus[3])
        self.assertIs(gen_norm, model.norms_gen[3])

        for component, graph_index, norm_index in (("graph", 3, 1), ("norm", 1, 3)):
            plan = InterventionPlan(operator_sources={1: 3}, operator_component=component)
            plan.validate(4)
            graph, bus_norm, gen_norm, _ = _operator_modules(model, 1, plan)
            self.assertIs(graph, model.layers[graph_index])
            self.assertIs(bus_norm, model.norms_bus[norm_index])
            self.assertIs(gen_norm, model.norms_gen[norm_index])
            diagonal = InterventionPlan(operator_sources={1: 1}, operator_component=component)
            graph, bus_norm, gen_norm, _ = _operator_modules(model, 1, diagonal)
            self.assertIs(graph, model.layers[1])
            self.assertIs(bus_norm, model.norms_bus[1])
            self.assertIs(gen_norm, model.norms_gen[1])
        with self.assertRaisesRegex(ValueError, "operator component"):
            InterventionPlan(operator_component="unknown").validate(4)

class RecurrentTailTests(unittest.TestCase):
    @staticmethod
    def model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 4)]
            + [torch.nn.Linear(4, 4) for _ in range(3)]
        )
        model.norms_bus = torch.nn.ModuleList(
            [torch.nn.LayerNorm(4) for _ in range(4)]
        )
        model.norms_gen = torch.nn.ModuleList(
            [torch.nn.LayerNorm(4) for _ in range(4)]
        )
        return model

    def test_strict_tail_shares_graphs_and_norms_but_not_stem(self):
        model = self.model()
        stem = model.layers[0]
        report = tie_recurrent_tail(model, share_norms=True)

        self.assertIs(model.layers[0], stem)
        self.assertIs(model.layers[1], model.layers[2])
        self.assertIs(model.layers[2], model.layers[3])
        self.assertIs(model.norms_bus[1], model.norms_bus[3])
        self.assertIs(model.norms_gen[1], model.norms_gen[3])
        self.assertLess(report.unique_parameters_after, report.unique_parameters_before)
        self.assertEqual(report.unique_parameters_after, count_unique_parameters(model))

    def test_graph_only_tail_preserves_step_specific_norms(self):
        model = self.model()
        bus_norms = list(model.norms_bus)
        report = tie_recurrent_tail(model, share_norms=False)

        self.assertIs(model.layers[1], model.layers[3])
        self.assertEqual(list(model.norms_bus), bus_norms)
        self.assertEqual(report.mode, "shared_graph_step_norms")


if __name__ == "__main__":
    unittest.main()
