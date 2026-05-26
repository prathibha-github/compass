"""Tests for shared benchmark CLI helpers."""

import logging
import unittest

from compass.benchmark.cli import (
    classify_generation_source,
    classify_judge_source,
    estimate_judge_cost_note,
    log_and_exit,
    log_benchmark_run_summary,
    log_errors_and_exit,
    log_token_budget_policy,
    parse_max_tokens_by_model_args,
    require_available_models,
    require_or_exit,
    resolve_token_budget_overrides,
    run_or_exit,
)


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("compass.benchmark.cli.tests")

    def test_log_and_exit_logs_message_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                log_and_exit(self.logger, "bad config", exit_code=2)

        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:bad config"])

    def test_require_or_exit_passes_when_condition_is_true(self):
        require_or_exit(True, self.logger, "should not fail", exit_code=2)

    def test_require_or_exit_logs_and_exits_when_condition_is_false(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                require_or_exit(False, self.logger, "missing models", exit_code=1)

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:missing models"])

    def test_log_errors_and_exit_logs_all_messages_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                log_errors_and_exit(
                    self.logger,
                    ["bad row 1", "bad row 2"],
                    exit_code=1,
                )

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(
            logs.output,
            [
                "ERROR:compass.benchmark.cli.tests:bad row 1",
                "ERROR:compass.benchmark.cli.tests:bad row 2",
            ],
        )

    def test_run_or_exit_returns_callback_value(self):
        value = run_or_exit(lambda: "ok", self.logger, exit_code=1)
        self.assertEqual(value, "ok")

    def test_run_or_exit_logs_value_error_and_exits(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                run_or_exit(
                    lambda: (_ for _ in ()).throw(ValueError("bad row")),
                    self.logger,
                    exit_code=1,
                )

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:bad row"])

    def test_parse_max_tokens_by_model_args(self):
        budgets = parse_max_tokens_by_model_args(
            ["claude-haiku-4-5=1000", "gpt-5.4-mini=1000"]
        )
        self.assertEqual(
            budgets,
            {
                "claude-haiku-4-5": 1000,
                "gpt-5.4-mini": 1000,
            },
        )

    def test_parse_max_tokens_by_model_args_rejects_invalid_entries(self):
        with self.assertRaisesRegex(ValueError, "expected MODEL=TOKENS"):
            parse_max_tokens_by_model_args(["bad-entry"])
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            parse_max_tokens_by_model_args(["llama3.1=0"])

    def test_resolve_token_budget_overrides_returns_uniform_map(self):
        budgets = resolve_token_budget_overrides(
            ["llama3.1", "mistral"],
            256,
            None,
            self.logger,
            exit_code=2,
        )
        self.assertEqual(budgets, {"llama3.1": 256, "mistral": 256})

    def test_resolve_token_budget_overrides_returns_custom_map(self):
        budgets = resolve_token_budget_overrides(
            ["llama3.1"],
            None,
            ["llama3.1=512"],
            self.logger,
            exit_code=2,
        )
        self.assertEqual(budgets, {"llama3.1": 512})

    def test_resolve_token_budget_overrides_exits_on_invalid_uniform_budget(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                resolve_token_budget_overrides(
                    ["llama3.1"],
                    0,
                    None,
                    self.logger,
                    exit_code=2,
                )

        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:--max-tokens must be > 0"])

    def test_require_available_models_filters_using_probe(self):
        available = require_available_models(
            ["llama3.1", "mistral"],
            lambda model: model == "mistral",
            self.logger,
            exit_code=1,
            unavailable_message="no models",
        )
        self.assertEqual(available, ["mistral"])

    def test_require_available_models_exits_when_none_pass_probe(self):
        with self.assertLogs(self.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exc:
                require_available_models(
                    ["llama3.1"],
                    lambda model: False,
                    self.logger,
                    exit_code=1,
                    unavailable_message="no models",
                )

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(logs.output, ["ERROR:compass.benchmark.cli.tests:no models"])

    def test_log_token_budget_policy_logs_uniform_budget(self):
        with self.assertLogs(self.logger, level="INFO") as logs:
            log_token_budget_policy(self.logger, {"llama3.1": 150, "mistral": 150})

        self.assertEqual(
            logs.output,
            [
                "INFO:compass.benchmark.cli.tests:Token budget policy: uniform max_tokens=150 across 2 models."
            ],
        )

    def test_log_token_budget_policy_logs_mixed_budget_warning(self):
        with self.assertLogs(self.logger, level="WARNING") as logs:
            log_token_budget_policy(
                self.logger,
                {"llama3.1": 150, "gemini-2.5-flash": 2000},
            )

        self.assertEqual(
            logs.output,
            [
                "WARNING:compass.benchmark.cli.tests:Token budget policy: mixed budgets enabled: {'llama3.1': 150, 'gemini-2.5-flash': 2000}"
            ],
        )

    def test_classify_generation_source(self):
        self.assertEqual(classify_generation_source(["llama3.1"]), "LOCAL (Ollama)")
        self.assertEqual(
            classify_generation_source(["gpt-4o-mini", "claude-haiku-4-5"]),
            "CLOUD (OpenAI/Anthropic/Google)",
        )
        self.assertEqual(
            classify_generation_source(["llama3.1", "gpt-4o-mini"]),
            "MIXED (Ollama + Cloud)",
        )

    def test_classify_judge_source(self):
        self.assertEqual(classify_judge_source("llama3.1"), "LOCAL (free)")
        self.assertEqual(classify_judge_source("gemini-2.5-flash"), "CLOUD (free tier)")
        self.assertEqual(classify_judge_source("gpt-4o-mini"), "CLOUD (paid)")

    def test_estimate_judge_cost_note(self):
        self.assertEqual(
            estimate_judge_cost_note(25, "llama3.1"),
            "FULLY FREE (local generation + local judge)",
        )
        self.assertEqual(
            estimate_judge_cost_note(25, "gpt-4o-mini"),
            "$0.03 (judge only, generation is free)",
        )

    def test_log_benchmark_run_summary_logs_shared_banner(self):
        with self.assertLogs(self.logger, level="INFO") as logs:
            log_benchmark_run_summary(
                self.logger,
                benchmark_title="TEST BENCHMARK",
                preset_name="default",
                models=("llama3.1", "gemini-2.5-flash"),
                judge_model="gemini-2.5-flash",
                rubric_names=("clarity", "truthfulness"),
                samples=3,
                output_dir="results/test",
                token_budget_defaults={"default": 150, "gemini": 2000},
                legacy_token_cap_threshold=301,
                total_evaluations=6,
            )

        rendered_logs = "\n".join(logs.output)
        self.assertIn("TEST BENCHMARK", rendered_logs)
        self.assertIn(
            "Generation: llama3.1, gemini-2.5-flash (MIXED (Ollama + Cloud))",
            rendered_logs,
        )
        self.assertIn(
            "Judge: gemini-2.5-flash (CLOUD (free tier))",
            rendered_logs,
        )
        self.assertIn("Cost: $0.01 (judge only, generation is free)", rendered_logs)


if __name__ == "__main__":
    unittest.main()
