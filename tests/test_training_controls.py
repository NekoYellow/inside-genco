from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_training_controls import build_config, validate_invariants, validate_split
from run_training_control import prepare_run_directory


class TrainingControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = yaml.safe_load((ROOT / "configs/training-controls.yaml").read_text())
        source_path = ROOT / self.protocol["source_config"]
        self.source = yaml.safe_load(source_path.read_text())

    def test_all_architectures_preserve_protocol_invariants(self) -> None:
        for architecture in self.protocol["architectures"].values():
            for seed in self.protocol["seeds"]:
                config = build_config(
                    self.source,
                    model_type=architecture["model_type"],
                    seed=seed,
                    split_directory="fixed/split",
                    epochs=self.protocol["budget"]["epochs"],
                    patience=self.protocol["budget"]["early_stopping_patience"],
                )
                validate_invariants(config, self.protocol["invariants"])
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["training"]["epochs"], 200)
                self.assertEqual(config["callbacks"]["patience"], 201)
                self.assertEqual(config["data"]["split_from_existing_files"], "fixed/split")
                self.assertNotIn("split_by_load_scenario_idx", config["data"])

    def test_split_validation_rejects_overlap(self) -> None:
        validate_split({"train": [0, 1], "val": [2], "test": [3]})
        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            validate_split({"train": [0, 1], "val": [1], "test": [3]})

    def test_retry_archives_only_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "seed0"
            run_directory.mkdir()
            manifest_path = run_directory / "manifest.json"
            manifest_path.write_text('{"status": "failed"}\n')

            archived = prepare_run_directory(
                run_directory, manifest_path, retry_failed=True
            )

            self.assertEqual(archived, run_directory / "manifest.attempt-1.json")
            self.assertFalse(manifest_path.exists())
            self.assertEqual(archived.read_text(), '{"status": "failed"}\n')

    def test_retry_refuses_non_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "seed0"
            run_directory.mkdir()
            manifest_path = run_directory / "manifest.json"
            manifest_path.write_text('{"status": "complete"}\n')

            with self.assertRaisesRegex(FileExistsError, "status 'complete'"):
                prepare_run_directory(run_directory, manifest_path, retry_failed=True)

            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
