"""Tests for the benchmark PR quality gate."""

import importlib.util
import pathlib
import unittest


def _load_gate_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_benchmark_test_delta.py"
    spec = importlib.util.spec_from_file_location("benchmark_test_delta_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkTestDeltaGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = _load_gate_module()

    def test_non_benchmark_changes_do_not_require_benchmark_tests(self):
        errors = self.gate.validate_benchmark_test_delta(
            ["README.md", "compass/cache.py"]
        )
        self.assertEqual(errors, [])

    def test_benchmark_logic_change_requires_benchmark_test_change(self):
        errors = self.gate.validate_benchmark_test_delta(
            ["compass/benchmark/runner.py", "tests/test_cache.py"]
        )
        self.assertTrue(errors)
        self.assertIn("Benchmark logic changed without a benchmark test update.", errors[0])

    def test_benchmark_logic_change_accepts_benchmark_test_file(self):
        errors = self.gate.validate_benchmark_test_delta(
            ["compass/benchmark/runner.py", "tests/test_benchmark_runner.py"]
        )
        self.assertEqual(errors, [])

    def test_example_benchmark_change_accepts_example_smoke_update(self):
        errors = self.gate.validate_benchmark_test_delta(
            [
                "examples/constitutional_compliance_benchmark.py",
                "tests/test_examples.py",
            ]
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
