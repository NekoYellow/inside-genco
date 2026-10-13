from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_trajectory_analysis import bootstrap_associations, node_relative_update
from run_operator_substitution import parse_steps, summarize_substitutions
from summarize_trajectory_analysis import summarize_node_activity


class RobustnessStatisticsTests(unittest.TestCase):
    def test_node_update_uses_per_node_relative_norm(self) -> None:
        before = torch.tensor([[3., 4.], [0., 2.]])
        after = torch.tensor([[6., 8.], [0., 3.]])
        torch.testing.assert_close(
            node_relative_update(before, after), torch.tensor([1., .5], dtype=torch.float64)
        )
        with self.assertRaisesRegex(ValueError, "equal-width"):
            node_relative_update(before, after[:, :1])
        with self.assertRaisesRegex(ValueError, "finite"):
            node_relative_update(before, after * float("nan"))

    def test_activity_weights_conditions_equally_and_threshold_is_strict(self) -> None:
        updates = pd.DataFrame([
            dict(scenario_id=scenario, device_id=device, node_type="bus", correction_step=2,
                 graph_update_relative_l2=value, functional_update_relative_l2=value)
            for scenario, device, value in ((1, 1, .01), (1, 2, .02), (2, 1, .03))
        ])
        result = summarize_node_activity(updates)
        graph = result[result.kind == "graph"].set_index("threshold")
        self.assertAlmostEqual(graph.loc[.01, "active_fraction"], .75)
        self.assertAlmostEqual(graph.loc[.02, "active_fraction"], .5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_node_activity(pd.concat([updates, updates.iloc[:1]]))
        updates.loc[0, "graph_update_relative_l2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_node_activity(updates)

    def setUp(self) -> None:
        steps = list(range(2, 12))
        scenarios = list(range(6))
        self.ablations = pd.DataFrame(
            [
                {
                    "scenario_id": scenario,
                    "correction_step": step,
                    "positive_delta_final_r_inf": float(step + scenario / 100),
                }
                for scenario in scenarios
                for step in steps
            ]
        )
        self.local_change = pd.Series(
            {step: float(step) for step in steps}, name="local_change"
        )

    def test_bootstrap_recomputes_paired_condition_summaries(self) -> None:
        result = bootstrap_associations(
            self.ablations,
            self.local_change,
            samples=50,
            seed=7,
        )

        self.assertEqual(result["late_steps"], [7, 8, 9, 10, 11])
        self.assertEqual(result["resampling_unit"], "operating condition")
        for name in ("cka_all", "cka_late"):
            self.assertAlmostEqual(result[name]["spearman"], 1.0)
            self.assertEqual(result[name]["bootstrap_samples"], 50)
            self.assertAlmostEqual(result[name]["bootstrap_ci_low"], 1.0)
            self.assertAlmostEqual(result[name]["bootstrap_ci_high"], 1.0)

    def test_bootstrap_rejects_incomplete_conditions(self) -> None:
        incomplete_ablations = self.ablations.iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "incomplete"):
            bootstrap_associations(
                incomplete_ablations,
                self.local_change,
                samples=5,
            )

    def test_operator_steps_and_summary_are_deterministic(self) -> None:
        self.assertEqual(parse_steps("11,2,7", 12), [2, 7, 11])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "steps must be in"):
            parse_steps("1,2", 12)
        rows = pd.DataFrame(
            [
                {
                    "scenario_id": scenario,
                    "target_step": 2,
                    "source_step": source,
                    "r_inf_pu_100MVA": float(source + scenario),
                    "delta_r_inf_pu_100MVA": float(source - 2),
                    "positive_delta_r_inf_pu_100MVA": float(max(source - 2, 0)),
                    "delta_vm_mae_pu": 0.1,
                    "delta_va_mae_rad": 0.2,
                    "delta_p_mae_all_bus_MW": 0.3,
                    "delta_q_mae_all_bus_Mvar": 0.4,
                }
                for scenario in (1, 2)
                for source in (2, 3)
            ]
        )
        summary = summarize_substitutions(rows)
        self.assertEqual(len(summary), 2)
        diagonal = summary[summary.source_step == 2].iloc[0]
        self.assertEqual(diagonal.conditions, 2)
        self.assertEqual(diagonal.delta_final_r_inf_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
