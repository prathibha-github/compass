"""Smoke tests for the example scripts."""

import contextlib
import importlib.util
import io
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"


class _FakeJudge:
    def __init__(self, config, client, cache):
        self._seen = set()

    def evaluate(self, text):
        cache_hit = text in self._seen
        self._seen.add(text)
        return SimpleNamespace(
            score=0.8,
            hit=False,
            confidence=0.9,
            rationale="ok",
            cache_hit=cache_hit,
        )


class _FakeComparison:
    def __init__(self):
        self.judges = {
            "m1": SimpleNamespace(score=0.3),
            "m2": SimpleNamespace(score=0.7),
        }

    def summary(self):
        return "comparison summary"

    def agreement_score(self):
        return 0.8

    def hit_agreement(self):
        return 0.5

    def score_range(self):
        return (0.3, 0.7)


class _FakeComparator:
    def __init__(self, judges):
        self.judges = judges

    def compare(self, response):
        return _FakeComparison()


def _load_example(name: str):
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExampleSmokeTests(unittest.TestCase):
    def _run_main(self, name: str):
        module = _load_example(name)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(module, "AnthropicClient", return_value=MagicMock()), \
             patch.object(module, "EvaluationCache", side_effect=lambda *args, **kwargs: object()), \
             patch.object(module, "LLMJudge", side_effect=_FakeJudge):
            patchers = []
            if hasattr(module, "MultiModelComparator"):
                patchers.append(
                    patch.object(module, "MultiModelComparator", side_effect=_FakeComparator)
                )
            if hasattr(module, "cost_summary"):
                patchers.append(
                    patch.object(
                        module,
                        "cost_summary",
                        return_value={
                            "results_count": 3,
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                            "total_tokens": 15,
                            "total_cost_usd": 0.01,
                        },
                    )
                )
            if hasattr(module, "cost_per_judge"):
                patchers.append(
                    patch.object(
                        module,
                        "cost_per_judge",
                        return_value={
                            "fake-model": {
                                "count": 3,
                                "total_cost_usd": 0.01,
                                "avg_cost_per_eval": 0.0033,
                            }
                        },
                    )
                )
            if hasattr(module, "reproducibility_report"):
                patchers.append(
                    patch.object(module, "reproducibility_report", return_value="report")
                )
            if hasattr(module, "time"):
                patchers.append(
                    patch.object(
                        module.time,
                        "time",
                        side_effect=[0.0, 2.0, 2.0, 4.0, 4.0, 4.5],
                    )
                )

            with contextlib.ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                module.main()

    def test_basic_eval(self):
        self._run_main("basic_eval")

    def test_batch_eval(self):
        self._run_main("batch_eval")

    def test_caching_demo(self):
        self._run_main("caching_demo")

    def test_custom_rubric(self):
        self._run_main("custom_rubric")

    def test_demo(self):
        self._run_main("demo")

    def test_multi_model_compare(self):
        self._run_main("multi_model_compare")


class BenchmarkMaxTokensTests(unittest.TestCase):
    """Verify per-model max_tokens selection in generate_completions."""

    def _run_generate(self, models, allow_mixed_token_budgets=False, max_tokens_by_model=None):
        """Run generate_completions with minimal fixtures; return captured max_tokens per model."""
        import tempfile
        benchmark = _load_example("constitutional_compliance_benchmark")

        prompts_by_rubric = {
            "task_focus": [{"id": "p1", "text": "Hello", "task_type": "general"}]
        }
        captured = {}

        def make_fake_client(model, **kwargs):
            fake = MagicMock()
            response = SimpleNamespace(
                completion="ok",
                tokens_used={"input": 1, "output": 1},
                cost_usd=0.0,
            )

            def fake_complete(prompt, max_tokens, temperature):
                captured[model] = max_tokens
                return response

            fake.complete.side_effect = fake_complete
            return fake

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            with patch("compass.benchmark.runner.GoogleAIClient", side_effect=make_fake_client), \
                 patch("compass.benchmark.runner.OllamaClient", side_effect=make_fake_client), \
                 patch("compass.benchmark.runner.OpenAIClient", side_effect=make_fake_client), \
                 patch("compass.benchmark.runner.AnthropicClient", side_effect=make_fake_client):
                benchmark.generate_completions(
                    models,
                    prompts_by_rubric,
                    1,
                    output_dir,
                    max_tokens_by_model=max_tokens_by_model,
                    allow_mixed_token_budgets=allow_mixed_token_budgets,
                )

        return captured

    def test_gemini_gets_2000_max_tokens(self):
        captured = self._run_generate(["gemini-2.5-flash"])
        self.assertEqual(captured["gemini-2.5-flash"], 2000)

    def test_non_gemini_gets_150_max_tokens(self):
        captured = self._run_generate(["llama3.1"])
        self.assertEqual(captured["llama3.1"], 150)

    def test_mixed_models_use_correct_limits(self):
        captured = self._run_generate(
            ["gemini-2.5-flash", "llama3.1"],
            allow_mixed_token_budgets=True,
        )
        self.assertEqual(captured["gemini-2.5-flash"], 2000)
        self.assertEqual(captured["llama3.1"], 150)

    def test_generate_completions_mixed_models_fail_without_override(self):
        with self.assertRaisesRegex(ValueError, "Mixed max token budgets detected"):
            self._run_generate(["gemini-2.5-flash", "llama3.1"])

    def test_generate_completions_custom_budget_map_validated(self):
        with self.assertRaisesRegex(ValueError, "Missing max token budget for model"):
            self._run_generate(
                ["llama3.1", "mistral"],
                allow_mixed_token_budgets=True,
                max_tokens_by_model={"llama3.1": 256},
            )

        with self.assertRaisesRegex(ValueError, "Invalid max token budget for model"):
            self._run_generate(
                ["llama3.1"],
                max_tokens_by_model={"llama3.1": 0},
            )

    def test_custom_uniform_budget_overrides_defaults(self):
        captured = self._run_generate(
            ["claude-haiku-4-5", "gpt-5.4-mini"],
            max_tokens_by_model={
                "claude-haiku-4-5": 1000,
                "gpt-5.4-mini": 1000,
            },
        )
        self.assertEqual(captured["claude-haiku-4-5"], 1000)
        self.assertEqual(captured["gpt-5.4-mini"], 1000)


class BenchmarkRecordLoadingTests(unittest.TestCase):
    def test_load_generation_records_migrates_legacy_rows(self):
        import tempfile

        benchmark = _load_example("constitutional_compliance_benchmark")
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","completion":"ok","task_type":"general"}',
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2","completion":"ok","sample_idx":"2","task_type":"general"}',
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "generations.jsonl"
            path.write_text("\n".join(lines) + "\n")
            rows = benchmark.load_generation_records(path)

        self.assertIn(("llama3.1", "clarity", "p1", 0), rows)
        self.assertIn(("llama3.1", "clarity", "p2", 2), rows)

    def test_load_evaluation_records_skips_invalid_rows(self):
        import tempfile

        benchmark = _load_example("constitutional_compliance_benchmark")
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","score":0.2,"hit":true}',
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}',
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text("\n".join(lines) + "\n")
            rows = benchmark.load_evaluation_records(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "p1")

    def test_load_evaluation_records_strict_rejects_invalid_rows(self):
        import tempfile

        benchmark = _load_example("constitutional_compliance_benchmark")
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","score":0.2,"hit":true}',
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p2"}',
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid evaluation row"):
                benchmark.load_evaluation_records(path, strict=True)


class BenchmarkQualityGuardrailTests(unittest.TestCase):
    def test_compute_generation_quality_flags_token_cap_fragments(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        quality = benchmark._compute_generation_quality(
            completion="At its core, a",
            output_tokens=150,
            max_tokens_requested=150,
        )
        self.assertTrue(quality["hit_token_cap"])
        self.assertTrue(quality["is_fragment"])
        self.assertTrue(quality["quality_flagged"])
        self.assertIn("visible_word_count", quality)

    def test_analyze_results_includes_quality_metrics(self):
        import tempfile

        benchmark = _load_example("constitutional_compliance_benchmark")
        line = (
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","task_type":"general",'
            '"sample_idx":0,"score":0.2,"hit":true,"generation_quality_flagged":true,'
            '"generation_hit_token_cap":true,"generation_is_fragment":true}'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "evaluations.jsonl"
            path.write_text(line + "\n")
            stats = benchmark.analyze_results(path, pathlib.Path(tmpdir))

        key = "llama3.1|clarity"
        self.assertIn(key, stats)
        self.assertEqual(stats[key]["quality_flagged_pct"], 100.0)
        self.assertEqual(stats[key]["token_cap_pct"], 100.0)
        self.assertEqual(stats[key]["fragment_pct"], 100.0)
        self.assertIsNone(stats[key]["quality_filtered_hit_rate"])

    def test_legacy_rows_infer_token_cap_when_missing_max_tokens(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        quality = benchmark._generation_quality_from_record(
            {
                "model": "gpt-5.4-mini",
                "completion": "At its core, a",
                "tokens_used": {"output": 150},
            }
        )
        self.assertTrue(quality["hit_token_cap"])
        self.assertTrue(quality["token_cap_inferred_legacy"])
        self.assertTrue(quality["quality_flagged"])

    def test_legacy_rows_warn_when_token_cap_is_inferred(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        from compass.benchmark.runner import _reset_warned_legacy_token_cap_thresholds

        _reset_warned_legacy_token_cap_thresholds()
        with self.assertLogs("compass.benchmark.runner", level="WARNING") as logs:
            quality = benchmark._generation_quality_from_record(
                {
                    "model": "gpt-5.4-mini",
                    "completion": "At its core, a",
                    "tokens_used": {"output": 301},
                },
                legacy_token_cap_threshold=301,
            )

        self.assertTrue(quality["token_cap_inferred_legacy"])
        self.assertIn("Override with --legacy-token-cap-threshold", logs.output[0])

    def test_legacy_rows_allow_custom_token_cap_threshold(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        quality = benchmark._generation_quality_from_record(
            {
                "model": "gpt-5.4-mini",
                "completion": "Longer answer",
                "tokens_used": {"output": 300},
            },
            legacy_token_cap_threshold=300,
        )
        self.assertTrue(quality["hit_token_cap"])
        self.assertTrue(quality["token_cap_inferred_legacy"])

    def test_finish_reason_can_mark_token_cap(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        quality = benchmark._compute_generation_quality(
            completion="Long response...",
            output_tokens=20,
            max_tokens_requested=1000,
            finish_reason="MAX_TOKENS",
        )
        self.assertTrue(quality["hit_token_cap"])
        self.assertTrue(quality["quality_flagged"])

    def test_backtick_does_not_mark_sentence_complete(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        quality = benchmark._compute_generation_quality(
            completion="`print('x')`",
            output_tokens=5,
            max_tokens_requested=2000,
        )
        self.assertTrue(quality["is_fragment"])


class BenchmarkTokenBudgetPolicyTests(unittest.TestCase):
    def test_validate_token_budget_policy_fails_on_mixed_by_default(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        with self.assertRaisesRegex(ValueError, "Mixed max token budgets detected"):
            benchmark.validate_token_budget_policy(
                ["gemini-2.5-flash", "llama3.1"],
                allow_mixed=False,
            )

    def test_validate_token_budget_policy_allows_mixed_with_override(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        budgets = benchmark.validate_token_budget_policy(
            ["gemini-2.5-flash", "llama3.1"],
            allow_mixed=True,
        )
        self.assertEqual(budgets["gemini-2.5-flash"], 2000)
        self.assertEqual(budgets["llama3.1"], 150)

    def test_validate_token_budget_policy_uniform_models(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        budgets = benchmark.validate_token_budget_policy(
            ["llama3.1", "mistral", "phi"],
            allow_mixed=False,
        )
        self.assertEqual(set(budgets.values()), {150})

    def test_default_max_tokens_prefers_longest_matching_prefix(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        budgets = {
            "default": 150,
            "gemini": 2000,
            "gemini-2.5": 5000,
        }
        self.assertEqual(
            benchmark.default_max_tokens_for_model(
                "gemini-2.5-flash",
                token_budgets=budgets,
            ),
            5000,
        )

class BenchmarkTokenBudgetCliParsingTests(unittest.TestCase):
    def test_parse_max_tokens_by_model_args(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        budgets = benchmark._parse_max_tokens_by_model_args(
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
        benchmark = _load_example("constitutional_compliance_benchmark")
        with self.assertRaisesRegex(ValueError, "expected MODEL=TOKENS"):
            benchmark._parse_max_tokens_by_model_args(["bad-entry"])
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            benchmark._parse_max_tokens_by_model_args(["llama3.1=0"])

    def test_parser_accepts_legacy_token_cap_threshold(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        args = benchmark.create_parser().parse_args(
            ["--legacy-token-cap-threshold", "300"]
        )
        self.assertEqual(args.legacy_token_cap_threshold, 300)

    def test_parser_defaults_to_benchmark_owned_preset(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        args = benchmark.create_parser().parse_args([])
        preset = benchmark.BENCHMARK_SPEC.get_run_preset(args.preset)
        run_config = benchmark.BENCHMARK_SPEC.make_run_config(
            preset_name=args.preset,
            models=args.models,
            samples=args.samples,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            allow_mixed_token_budgets=args.allow_mixed_token_budgets,
            legacy_token_cap_threshold=args.legacy_token_cap_threshold,
        )
        self.assertEqual(run_config.models, preset.models)
        self.assertEqual(run_config.samples, preset.samples)
        self.assertEqual(run_config.judge_model, preset.judge_model)
        self.assertEqual(run_config.output_dir, preset.output_dir)
        self.assertEqual(
            run_config.allow_mixed_token_budgets,
            preset.policy.allow_mixed_token_budgets,
        )

    def test_parser_supports_boolean_override_for_mixed_token_budgets(self):
        benchmark = _load_example("constitutional_compliance_benchmark")
        enabled = benchmark.create_parser().parse_args(
            ["--allow-mixed-token-budgets"]
        )
        disabled = benchmark.create_parser().parse_args(
            ["--no-allow-mixed-token-budgets"]
        )
        self.assertTrue(enabled.allow_mixed_token_budgets)
        self.assertFalse(disabled.allow_mixed_token_budgets)


class BenchmarkMainOrchestrationTests(unittest.TestCase):
    def test_main_skips_summary_when_preset_disables_it(self):
        import tempfile

        from compass.benchmark import BenchmarkPolicyDefaults, BenchmarkRunPreset, build_benchmark_spec

        benchmark = _load_example("constitutional_compliance_benchmark")
        custom_spec = build_benchmark_spec(
            name=benchmark.BENCHMARK_SPEC.name,
            version=benchmark.BENCHMARK_SPEC.version,
            prompts_by_rubric=benchmark.PROMPTS,
            rubrics_by_name=benchmark.BENCHMARK_SPEC.rubrics_by_name,
            run_presets={
                "default": BenchmarkRunPreset(
                    models=("llama3.1",),
                    samples=1,
                    judge_model="llama3.1",
                    output_dir="results/test_pairwise_only",
                    policy=BenchmarkPolicyDefaults(analysis_lanes=("pairwise",)),
                )
            },
            pairwise_segment_field=benchmark.BENCHMARK_SPEC.pairwise_segment_field,
        )
        output_dir = None

        class _Runner:
            def __init__(self, spec):
                self.spec = spec
                self.analyze_calls = 0
                self.rank_calls = 0

            def validate_run_config(self, run_config):
                return run_config

            def generate(self, run_config):
                return output_dir / "generations.jsonl"

            def evaluate(self, generations_path, run_config):
                return output_dir / "evaluations.jsonl"

            def analyze(self, evaluations_path, run_config):
                self.analyze_calls += 1
                return {"unexpected": True}

            def rank(self, evaluations_path, run_config):
                self.rank_calls += 1

        runner = _Runner(custom_spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            with patch.object(benchmark, "BENCHMARK_SPEC", custom_spec), \
                 patch.object(benchmark, "BENCHMARK_RUNNER", runner), \
                 patch.object(benchmark, "test_model_connection", return_value=True), \
                 patch.object(benchmark, "setup_output_dir", return_value=output_dir), \
                 patch.object(benchmark, "print_summary") as print_summary, \
                 patch.object(sys, "argv", ["constitutional_compliance_benchmark.py"]):
                benchmark.main()

        self.assertEqual(runner.analyze_calls, 0)
        self.assertEqual(runner.rank_calls, 1)
        print_summary.assert_not_called()

    def test_main_exits_cleanly_when_report_validation_fails(self):
        import tempfile

        benchmark = _load_example("constitutional_compliance_benchmark")
        output_dir = None

        class _Runner:
            def __init__(self, spec):
                self.spec = spec

            def validate_run_config(self, run_config):
                return run_config

            def generate(self, run_config):
                return output_dir / "generations.jsonl"

            def evaluate(self, generations_path, run_config):
                raise ValueError("Benchmark report validation failed: bad row")

        runner = _Runner(benchmark.BENCHMARK_SPEC)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            with patch.object(benchmark, "BENCHMARK_RUNNER", runner), \
                 patch.object(benchmark, "test_model_connection", return_value=True), \
                 patch.object(benchmark, "setup_output_dir", return_value=output_dir), \
                 patch.object(benchmark.logger, "error") as log_error, \
                 patch.object(sys, "argv", ["constitutional_compliance_benchmark.py"]):
                with self.assertRaises(SystemExit) as exc:
                    benchmark.main()

        self.assertEqual(exc.exception.code, 1)
        log_error.assert_any_call("Benchmark report validation failed: bad row")


if __name__ == "__main__":
    unittest.main()
