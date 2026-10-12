from __future__ import annotations

import unittest

import pandas as pd

from run_trajectory_analysis import bootstrap_associations


class RobustnessStatisticsTests(unittest.TestCase):
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
        self.updates = pd.DataFrame(
            [
                {
                    "scenario_id": scenario,
                    "correction_step": step,
                    "node_type": "bus",
                    "update_relative_l2": float(step + scenario / 200),
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
            self.updates,
            self.local_change,
            samples=50,
            seed=7,
        )

        self.assertEqual(result["late_steps"], [7, 8, 9, 10, 11])
        self.assertEqual(result["resampling_unit"], "operating condition")
        for name in ("cka_all", "cka_late", "direct_all", "direct_late"):
            self.assertAlmostEqual(result[name]["spearman"], 1.0)
            self.assertEqual(result[name]["bootstrap_samples"], 50)
            self.assertAlmostEqual(result[name]["bootstrap_ci_low"], 1.0)
            self.assertAlmostEqual(result[name]["bootstrap_ci_high"], 1.0)

    def test_bootstrap_rejects_unpaired_conditions(self) -> None:
        incomplete_updates = self.updates.iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "complete and paired"):
            bootstrap_associations(
                self.ablations,
                incomplete_updates,
                self.local_change,
                samples=5,
            )


if __name__ == "__main__":
    unittest.main()
