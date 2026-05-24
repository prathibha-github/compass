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


if __name__ == "__main__":
    unittest.main()
