"""Fast unit tests that do not require the multi-gigabyte research assets."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inside_genco.interventions import InterventionPlan, TensorEdit, stable_relative_perturbation
from inside_genco.similarity import linear_cka
from inside_genco.trajectory_metrics import (
    LinearCKAAccumulator,
    first_threshold_crossing,
    local_step_metrics,
    node_relative_l2_updates,
)
from inside_genco.intervention_metrics import (
    cluster_bootstrap_mean,
    interaction_effect,
    propagation_metrics,
    rank_steps,
    summarize_paired_effect,
)
from inside_genco.compute_policy import (
    build_policies,
    combination_interaction,
    random_steps,
    uniformly_spaced_steps,
)
from inside_genco.tied_tail import count_unique_parameters, tie_recurrent_tail
from inside_genco.trace import LayerTrace, _operator_modules


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
    def test_stable_perturbation_is_order_invariant_and_per_scenario_scaled(self):
        reference = torch.arange(1.0, 25.0).reshape(6, 4)
        scenarios = torch.tensor([10, 10, 10, 20, 20, 20])
        devices = torch.tensor([2, 0, 1, 2, 0, 1])
        delta = stable_relative_perturbation(
            reference, scenarios, devices, relative_l2=0.03, seed=17, node_type="bus"
        )
        order = torch.tensor([5, 1, 3, 0, 4, 2])
        shuffled = stable_relative_perturbation(
            reference[order], scenarios[order], devices[order],
            relative_l2=0.03, seed=17, node_type="bus",
        )
        inverse = torch.argsort(order)
        torch.testing.assert_close(delta, shuffled[inverse], atol=0, rtol=0)
        for scenario in (10, 20):
            mask = scenarios == scenario
            ratio = torch.linalg.vector_norm(delta[mask]) / torch.linalg.vector_norm(reference[mask])
            self.assertAlmostEqual(float(ratio), 0.03, places=6)

    def test_plan_rejects_ambiguous_or_invalid_edits(self):
        with self.assertRaisesRegex(ValueError, "layer 0"):
            InterventionPlan(skip_layers=frozenset({0})).validate(12)
        with self.assertRaisesRegex(ValueError, "final layer"):
            InterventionPlan(skip_layers=frozenset({11})).validate(12)
        edit = TensorEdit(2, "h_graph", "bus", torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "multiple edits"):
            InterventionPlan(edits=(edit, edit)).validate(12)

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

    def test_perturbation_direction_is_keyed_by_step(self):
        reference = torch.ones(3, 4)
        scenarios = torch.tensor([10, 10, 10])
        devices = torch.tensor([0, 1, 2])
        first = stable_relative_perturbation(
            reference, scenarios, devices, relative_l2=0.1, seed=7,
            node_type="bus", layer_index=1,
        )
        second = stable_relative_perturbation(
            reference, scenarios, devices, relative_l2=0.1, seed=7,
            node_type="bus", layer_index=2,
        )
        self.assertFalse(torch.equal(first, second))

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

    def test_final_layer_functional_state_excludes_unconsumed_feedback(self):
        zero = torch.zeros(1, 2)
        graph = torch.ones(1, 2)
        propagated = torch.full((1, 2), 2.0)
        trace = LayerTrace(
            layer_index=11,
            display_layer=12,
            h_in_bus=zero,
            h_in_gen=zero,
            h_graph_bus=graph,
            h_graph_gen=graph,
            x_bus=None,
            x_gen=None,
            p_flow=None,
            q_flow=None,
            residual_bus=None,
            feedback_bus=None,
            h_out_bus=propagated,
            h_out_gen=propagated,
            feedback_used=False,
            skipped=False,
            feedback_scale=1.0,
        )

        self.assertIs(trace.functional_state("bus"), graph)
        with self.assertRaisesRegex(ValueError, "node_type"):
            trace.functional_state("branch")


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


class TrajectoryMetricTests(unittest.TestCase):
    def test_node_relative_updates_preserve_node_rows(self):
        h_in = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        h_out = torch.tensor([[6.0, 8.0], [0.0, 3.0]])

        actual = node_relative_l2_updates(h_in, h_out)

        torch.testing.assert_close(actual, torch.tensor([1.0, 0.5], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "identical input and output shapes"):
            node_relative_l2_updates(h_in, h_out[:, :1])

    def test_streaming_cka_matches_batch_definition(self):
        generator = torch.Generator().manual_seed(31)
        x = torch.randn(53, 7, generator=generator)
        y = torch.randn(53, 5, generator=generator)
        ids = [("scenario", index) for index in range(len(x))]
        accumulator = LinearCKAAccumulator()
        accumulator.update(x[:19], y[:19], x_ids=ids[:19], y_ids=ids[:19])
        accumulator.update(x[19:], y[19:], x_ids=ids[19:], y_ids=ids[19:])
        expected = linear_cka(x, y, x_ids=ids, y_ids=ids)
        actual = accumulator.result()
        self.assertTrue(actual.valid)
        self.assertAlmostEqual(actual.value, expected.value, places=12)

    def test_streaming_cka_rejects_duplicate_global_ids(self):
        accumulator = LinearCKAAccumulator()
        value = torch.randn(3, 2)
        accumulator.update(value, value, x_ids=[1, 2, 3], y_ids=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, "globally unique"):
            accumulator.update(value, value, x_ids=[3, 4, 5], y_ids=[3, 4, 5])

    def test_local_metrics_and_width_change(self):
        h_in = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
        h_out = h_in * 2
        metrics = local_step_metrics(h_in, h_out, h_graph=h_in, feedback=h_in)
        self.assertAlmostEqual(metrics.update_relative_l2, 1.0)
        self.assertAlmostEqual(metrics.row_cosine_mean, 1.0)
        self.assertAlmostEqual(metrics.feedback_relative_l2, 1.0)
        changed = local_step_metrics(torch.ones(2, 1), torch.ones(2, 3))
        self.assertIsNone(changed.update_relative_l2)
        self.assertIsNone(changed.row_cosine_mean)

    def test_fixed_threshold_crossings_preserve_nonmonotonic_trajectory(self):
        rows = first_threshold_crossing([3.0, 0.8, 1.2, 0.4], [1.0, 0.5])
        self.assertEqual(rows[0]["first_step"], 2)
        self.assertEqual(rows[1]["first_step"], 4)


class InterventionMetricTests(unittest.TestCase):
    def test_propagation_and_undefined_amplification(self):
        baseline = torch.tensor([[3.0, 4.0]])
        changed = torch.tensor([[6.0, 8.0]])
        metric = propagation_metrics(baseline, changed, initial_relative_l2=0.5)
        self.assertAlmostEqual(metric.absolute_l2, 5.0)
        self.assertAlmostEqual(metric.relative_l2, 1.0)
        self.assertAlmostEqual(metric.amplification, 2.0)
        undefined = propagation_metrics(baseline, baseline, initial_relative_l2=0.0)
        self.assertIsNone(undefined.amplification)
        self.assertIn("undefined", undefined.reason)

    def test_feedback_interaction_sign(self):
        # Feedback-on reduces perturbation damage from 3 to 1, so I_Q is +2.
        self.assertEqual(interaction_effect(10.0, 13.0, 5.0, 6.0), 2.0)
        self.assertIsNone(interaction_effect(float("nan"), 1.0, 1.0, 1.0))

    def test_cluster_bootstrap_keeps_cluster_rows_together(self):
        values = [1.0, 3.0, 10.0]
        clusters = [7, 7, 9]
        result = cluster_bootstrap_mean(values, clusters, samples=500, seed=2)
        self.assertAlmostEqual(result["mean"], 14.0 / 3.0)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["finite_n"], 3)

    def test_positive_damage_does_not_cancel_signed_improvements(self):
        result = summarize_paired_effect([-2.0, 1.0], [1, 2], samples=100, seed=3)
        self.assertAlmostEqual(result["signed_mean"], -0.5)
        self.assertAlmostEqual(result["positive_mean"], 0.5)

    def test_step_ranking_is_deterministic_and_nonfinite_last(self):
        rows = [
            {"correction_step_number": 4, "score": 0.2},
            {"correction_step_number": 2, "score": 0.2},
            {"correction_step_number": 3, "score": float("nan")},
        ]
        ranked = rank_steps(rows, score_key="score")
        self.assertEqual([row["correction_step_number"] for row in ranked], [2, 4, 3])


class ComputePolicyTests(unittest.TestCase):
    def test_position_and_random_policies_are_deterministic(self):
        candidates = list(range(2, 12))
        self.assertEqual(uniformly_spaced_steps(candidates, 1), (7,))
        self.assertEqual(uniformly_spaced_steps(candidates, 2), (5, 8))
        self.assertEqual(random_steps(candidates, 2, 1301), (2, 11))
        self.assertEqual(random_steps(candidates, 2, 1301), random_steps(candidates, 2, 1301))

    def test_policy_builder_uses_only_eligible_global_rows_and_highest_cka(self):
        candidates = [2, 3, 4]
        rows = []
        for step, damage, p95 in [(2, 0.3, 0.4), (3, 0.1, 0.2), (4, 0.05, 0.9)]:
            rows.append({
                "correction_step_number": step,
                "delta_r_inf_pu_100MVA_positive_mean": damage,
                "delta_r_inf_pu_100MVA_p95": p95,
                "delta_vm_mae_pu_signed_mean": 0.0,
                "delta_va_mae_rad_signed_mean": 0.0,
                "intervention_nonfinite_rate": 0.0,
            })
        policies = build_policies(
            candidates=candidates, budgets=[1], random_seeds=[7],
            local_scores={2: 0.8, 3: 0.95, 4: 0.9}, global_rows=rows,
            global_constraints={
                "max_nonfinite_rate": 0.0, "max_r_inf_p95": 0.5,
                "max_abs_vm_mae_mean": 0.1, "max_abs_va_mae_mean": 0.1,
            },
        )
        selected = {policy.family: policy.step_numbers for policy in policies}
        self.assertEqual(selected["local"], (3,))
        self.assertEqual(selected["global"], (3,))

    def test_combination_interaction_sign_and_nonfinite(self):
        self.assertAlmostEqual(combination_interaction(5.0, [2.0, 1.0]), 2.0)
        self.assertIsNone(combination_interaction(float("nan"), [1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
