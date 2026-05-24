"""Smoke tests for the example scripts."""

import contextlib
import importlib.util
import io
import pathlib
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

    def _run_generate(self, models):
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
            with patch.object(benchmark, "GoogleAIClient", side_effect=make_fake_client), \
                 patch.object(benchmark, "OllamaClient", side_effect=make_fake_client), \
                 patch.object(benchmark, "OpenAIClient", side_effect=make_fake_client), \
                 patch.object(benchmark, "AnthropicClient", side_effect=make_fake_client):
                benchmark.generate_completions(models, prompts_by_rubric, 1, output_dir)

        return captured

    def test_gemini_gets_2000_max_tokens(self):
        captured = self._run_generate(["gemini-2.5-flash"])
        self.assertEqual(captured["gemini-2.5-flash"], 2000)

    def test_non_gemini_gets_150_max_tokens(self):
        captured = self._run_generate(["llama3.1"])
        self.assertEqual(captured["llama3.1"], 150)

    def test_mixed_models_use_correct_limits(self):
        captured = self._run_generate(["gemini-2.5-flash", "llama3.1"])
        self.assertEqual(captured["gemini-2.5-flash"], 2000)
        self.assertEqual(captured["llama3.1"], 150)


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


if __name__ == "__main__":
    unittest.main()
