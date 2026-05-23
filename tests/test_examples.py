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


if __name__ == "__main__":
    unittest.main()
