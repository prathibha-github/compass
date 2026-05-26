"""Tests for benchmark report validation and summary formatting."""

import importlib.util
import pathlib
import tempfile
import unittest

from compass.benchmark import (
    BenchmarkValidationIssue,
    analyze_results,
    get_benchmark_spec,
    rank_models,
    validate_benchmark_report,
)
from compass.benchmark.reporting import format_summary

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _load_benchmark_module():
    path = EXAMPLES_DIR / "constitutional_compliance_benchmark.py"
    spec = importlib.util.spec_from_file_location("constitutional_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkValidationTests(unittest.TestCase):
    def test_validate_benchmark_report_accepts_quality_complete_rows(self):
        path = FIXTURES_DIR / "benchmark_evaluations_valid.jsonl"
        self.assertEqual(validate_benchmark_report(path), [])

    def test_validate_benchmark_report_rejects_missing_quality_fields(self):
        path = FIXTURES_DIR / "benchmark_evaluations_invalid.jsonl"
        errors = validate_benchmark_report(path)
        self.assertTrue(errors)
        self.assertIsInstance(errors[0], BenchmarkValidationIssue)
        self.assertEqual(errors[0].code, "missing_evaluation_quality_fields")
        self.assertIn("generation_visible_chars", str(errors[0]))

    def test_benchmark_validation_issue_stringifies_to_stable_cli_message(self):
        issue = BenchmarkValidationIssue(
            code="missing_summary_quality_fields",
            location="stats row 'llama3.1|clarity'",
            message="missing quality fields: quality_filtered_total",
        )
        self.assertEqual(
            str(issue),
            "stats row 'llama3.1|clarity': missing quality fields: quality_filtered_total",
        )

    def test_validate_benchmark_report_raises_for_malformed_rows(self):
        line = '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text(line + "\n")
            with self.assertRaisesRegex(ValueError, "invalid evaluation row"):
                validate_benchmark_report(path)

    def test_format_summary_matches_expected_snapshot(self):
        path = FIXTURES_DIR / "benchmark_evaluations_valid.jsonl"
        stats = analyze_results(path, path.parent)
        summary = format_summary(stats, path)
        expected = "\n".join(
            [
                "",
                "=" * 100,
                "EVALUATION SUMMARY",
                "=" * 100,
                "",
                "Model           | Rubric          |  Hit Rate |  Q-Flag |   Cap |  Frag | LegacyCap |  QF Hit | Samples",
                "-" * 100,
                "llama3.1        | clarity         |     50.0% |   50.0% | 50.0% | 50.0% |      0.0% |    0.0% |       2",
                "",
                f"Results saved: {path}",
                "=" * 100,
                "",
            ]
        )
        self.assertEqual(summary, expected)

    def test_example_summary_output_contains_quality_columns(self):
        benchmark = _load_benchmark_module()
        path = FIXTURES_DIR / "benchmark_evaluations_valid.jsonl"
        stats = benchmark.analyze_results(path, path.parent)
        summary = format_summary(stats, path)
        self.assertIn("Q-Flag", summary)
        self.assertIn("LegacyCap", summary)
        self.assertIn("QF Hit", summary)

    def test_analyze_results_raises_for_malformed_rows(self):
        line = '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text(line + "\n")
            with self.assertRaisesRegex(ValueError, "invalid evaluation row"):
                analyze_results(path, pathlib.Path(tmpdir))

    def test_rank_models_raises_for_malformed_rows(self):
        spec = get_benchmark_spec("constitutional_compliance")
        line = '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text(line + "\n")
            with self.assertRaisesRegex(ValueError, "invalid evaluation row"):
                rank_models(path, spec, pathlib.Path(tmpdir))

    def test_validator_script_exits_nonzero_for_invalid_fixture(self):
        import runpy
        import sys

        script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_report.py"
        old_argv = sys.argv
        try:
            sys.argv = [str(script_path), str(FIXTURES_DIR / "benchmark_evaluations_invalid.jsonl")]
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path(str(script_path), run_name="__main__")
            self.assertEqual(exc.exception.code, 1)
        finally:
            sys.argv = old_argv

    def test_validator_script_accepts_valid_fixture(self):
        import runpy
        import sys

        script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_report.py"
        old_argv = sys.argv
        try:
            sys.argv = [str(script_path), str(FIXTURES_DIR / "benchmark_evaluations_valid.jsonl")]
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path(str(script_path), run_name="__main__")
            self.assertEqual(exc.exception.code, 0)
        finally:
            sys.argv = old_argv

    def test_validator_script_exits_nonzero_for_malformed_rows(self):
        import runpy
        import sys

        script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_report.py"
        old_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = pathlib.Path(tmpdir) / "evaluations.jsonl"
                path.write_text('{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}\n')
                sys.argv = [str(script_path), str(path)]
                with self.assertRaises(SystemExit) as exc:
                    runpy.run_path(str(script_path), run_name="__main__")
                self.assertEqual(exc.exception.code, 1)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
