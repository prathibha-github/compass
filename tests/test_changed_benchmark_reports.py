"""Tests for changed benchmark report validation in pull requests."""

import importlib.util
import pathlib
import tempfile
import unittest

from compass.benchmark.validation import BenchmarkValidationIssue


def _load_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_changed_benchmark_reports.py"
    )
    spec = importlib.util.spec_from_file_location(
        "changed_benchmark_reports",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChangedBenchmarkReportsTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_report_path_filter_matches_evaluation_jsonl_only(self):
        self.assertTrue(self.module.is_benchmark_report_path("evaluations.jsonl"))
        self.assertTrue(
            self.module.is_benchmark_report_path("results/bench/evaluations_gpt-4o.jsonl")
        )
        self.assertFalse(self.module.is_benchmark_report_path("generations.jsonl"))
        self.assertFalse(self.module.is_benchmark_report_path("evaluations.jsonl.bak"))
        self.assertFalse(self.module.is_benchmark_report_path("README.md"))

    def test_validate_changed_reports_ignores_non_report_changes(self):
        errors = self.module.validate_changed_reports(["README.md", "tests/test_cache.py"])
        self.assertEqual(errors, [])

    def test_validate_changed_reports_surfaces_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            report_path.write_text("{}\n")

            def fake_validator(path):
                self.assertEqual(path, report_path)
                return [
                    BenchmarkValidationIssue(
                        code="missing_evaluation_quality_fields",
                        location="evaluation row 1",
                        message="missing quality fields",
                    )
                ]

            errors = self.module.validate_changed_reports(
                [str(report_path)],
                validator=fake_validator,
            )

        self.assertEqual(
            errors,
            [f"{report_path}: evaluation row 1: missing quality fields"],
        )

    def test_validate_changed_reports_passes_valid_reports(self):
        valid_fixture = (
            pathlib.Path(__file__).resolve().parent
            / "fixtures"
            / "benchmark_evaluations_valid.jsonl"
        )
        errors = self.module.validate_changed_reports([str(valid_fixture)])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
