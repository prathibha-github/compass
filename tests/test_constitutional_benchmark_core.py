"""Focused unit tests for constitutional benchmark core loops.

These tests intentionally target extraction-risk behavior in
examples/constitutional_compliance_benchmark.py:
- generation loop + resume semantics
- evaluation loop + resume semantics
- judge invocation count for pending work
- aggregation behavior under quality flags
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from compass.benchmark.schemas import migrate_evaluation_record, migrate_generation_record

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _load_benchmark_module():
    path = EXAMPLES_DIR / "constitutional_compliance_benchmark.py"
    spec = importlib.util.spec_from_file_location("constitutional_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConstitutionalBenchmarkCoreTests(unittest.TestCase):
    def test_generate_completions_reuses_client_per_model(self):
        benchmark = _load_benchmark_module()
        prompts = {
            "task_focus": [
                {"id": "p1", "text": "hello", "task_type": "general"},
                {"id": "p2", "text": "hello again", "task_type": "general"},
            ]
        }
        created = []

        class _Client:
            def complete(self, prompt, max_tokens, temperature):
                return SimpleNamespace(
                    completion="ok",
                    tokens_used={"input": 1, "output": 10},
                    cost_usd=0.0,
                )

        def _make_client(model):
            created.append(model)
            return _Client()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            with patch("compass.benchmark.runner.OllamaClient", side_effect=_make_client):
                benchmark.generate_completions(["llama3.1"], prompts, 2, out)

        self.assertEqual(created, ["llama3.1"])

    def test_generate_completions_resume_skips_completed_samples(self):
        benchmark = _load_benchmark_module()
        prompts = {
            "task_focus": [
                {"id": "p1", "text": "hello", "task_type": "general"},
            ]
        }

        class _Client:
            def __init__(self):
                self.calls = 0

            def complete(self, prompt, max_tokens, temperature):
                self.calls += 1
                return SimpleNamespace(
                    completion="ok",
                    tokens_used={"input": 1, "output": 10},
                    cost_usd=0.0,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            first_client = _Client()
            with patch("compass.benchmark.runner.OllamaClient", return_value=first_client):
                benchmark.generate_completions(["llama3.1"], prompts, 2, out)
            self.assertEqual(first_client.calls, 2)

            # Resume pass should skip both completed samples.
            second_client = _Client()
            with patch("compass.benchmark.runner.OllamaClient", return_value=second_client):
                benchmark.generate_completions(["llama3.1"], prompts, 2, out)
            self.assertEqual(second_client.calls, 0)

    def test_generate_completions_writes_quality_fields(self):
        benchmark = _load_benchmark_module()
        prompts = {
            "clarity": [
                {"id": "p1", "text": "explain", "task_type": "general"},
            ]
        }

        class _Client:
            def complete(self, prompt, max_tokens, temperature):
                return SimpleNamespace(
                    completion="At its core, a",
                    tokens_used={"input": 1, "output": 150},
                    cost_usd=0.0,
                    finish_reason="max_tokens",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            with patch("compass.benchmark.runner.OllamaClient", return_value=_Client()):
                path = benchmark.generate_completions(["llama3.1"], prompts, 1, out)

            row = json.loads(path.read_text().strip().splitlines()[0])
            self.assertIn("benchmark_schema_version", row)
            self.assertEqual(row["benchmark_record_type"], "generation")
            self.assertIn("max_tokens_requested", row)
            self.assertIn("visible_chars", row)
            self.assertIn("visible_word_count", row)
            self.assertIn("hit_token_cap", row)
            self.assertIn("quality_flagged", row)

    def test_evaluate_completions_skips_completed_and_records_quality(self):
        benchmark = _load_benchmark_module()

        class _FakeJudge:
            calls = 0

            def __init__(self, config, client, cache):
                self.config = config
                self.client = client
                self.cache = cache

            def evaluate(self, text):
                _FakeJudge.calls += 1
                return SimpleNamespace(
                    score=0.2,
                    hit=False,
                    confidence=0.9,
                    rationale="rationale",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            generations_path = out / "generations.jsonl"
            rows = [
                migrate_generation_record(
                    {
                        "model": "llama3.1",
                        "rubric": "clarity",
                        "prompt_id": "p1",
                        "task_type": "general",
                        "sample_idx": 0,
                        "completion": "At its core,",
                        "tokens_used": {"input": 1, "output": 150},
                    }
                ),
                migrate_generation_record(
                    {
                        "model": "llama3.1",
                        "rubric": "clarity",
                        "prompt_id": "p1",
                        "task_type": "general",
                        "sample_idx": 1,
                        "completion": "At its core,",
                        "tokens_used": {"input": 1, "output": 150},
                    }
                ),
            ]
            generations_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

            # Pre-mark sample_idx=0 as already evaluated.
            eval_path = out / "evaluations_llama3.1.jsonl"
            eval_path.write_text(
                json.dumps(
                    migrate_evaluation_record(
                        {
                            "model": "llama3.1",
                            "rubric": "clarity",
                            "prompt_id": "p1",
                            "task_type": "general",
                            "sample_idx": 0,
                            "score": 0.1,
                            "hit": False,
                            "confidence": 0.8,
                            "rationale": "cached",
                            "judge_model": "llama3.1",
                            "generation_visible_chars": 120,
                            "generation_visible_word_count": 20,
                            "generation_hit_token_cap": False,
                            "generation_is_fragment": False,
                            "generation_quality_flagged": False,
                            "generation_finish_reason": "stop",
                            "generation_token_cap_inferred_legacy": False,
                        }
                    )
                )
                + "\n"
            )

            _FakeJudge.calls = 0
            with patch("compass.benchmark.runner.LLMJudge", side_effect=_FakeJudge), patch(
                "compass.benchmark.runner.OllamaClient", return_value=SimpleNamespace()
            ):
                benchmark.evaluate_completions(generations_path, "llama3.1", out)

            self.assertEqual(_FakeJudge.calls, 1)
            saved = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(saved), 2)
            latest = saved[-1]
            self.assertIn("generation_quality_flagged", latest)
            self.assertIn("generation_hit_token_cap", latest)
            self.assertIn("generation_is_fragment", latest)

    def test_analyze_results_reports_quality_filtered_rate(self):
        benchmark = _load_benchmark_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            eval_path = out / "evaluations.jsonl"
            eval_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "m1",
                                "rubric": "clarity",
                                "prompt_id": "p1",
                                "sample_idx": 0,
                                "score": 0.1,
                                "hit": False,
                                "generation_quality_flagged": True,
                                "generation_hit_token_cap": True,
                                "generation_is_fragment": True,
                            }
                        ),
                        json.dumps(
                            {
                                "model": "m1",
                                "rubric": "clarity",
                                "prompt_id": "p2",
                                "sample_idx": 0,
                                "score": 0.9,
                                "hit": True,
                                "generation_quality_flagged": False,
                                "generation_hit_token_cap": False,
                                "generation_is_fragment": False,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            stats = benchmark.analyze_results(eval_path, out)
            row = stats["m1|clarity"]
            self.assertEqual(row["total"], 2)
            self.assertEqual(row["quality_filtered_total"], 1)
            self.assertEqual(row["quality_filtered_hit_rate"], 100.0)
            self.assertEqual(row["token_cap_pct"], 50.0)
            self.assertEqual(row["fragment_pct"], 50.0)

    def test_analyze_results_can_exclude_quality_flagged_rows(self):
        benchmark = _load_benchmark_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir)
            eval_path = out / "evaluations.jsonl"
            eval_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "model": "m1",
                                "rubric": "clarity",
                                "prompt_id": "p1",
                                "sample_idx": 0,
                                "score": 0.1,
                                "hit": False,
                                "generation_quality_flagged": True,
                                "generation_hit_token_cap": True,
                                "generation_is_fragment": True,
                            }
                        ),
                        json.dumps(
                            {
                                "model": "m1",
                                "rubric": "clarity",
                                "prompt_id": "p2",
                                "sample_idx": 0,
                                "score": 0.9,
                                "hit": True,
                                "generation_quality_flagged": False,
                                "generation_hit_token_cap": False,
                                "generation_is_fragment": False,
                            }
                        ),
                    ]
                )
                + "\n"
            )

            stats = benchmark.analyze_results(
                eval_path,
                out,
                quality_filter_mode="exclude_flagged",
            )
            row = stats["m1|clarity"]
            self.assertEqual(row["hits"], 1)
            self.assertEqual(row["total"], 1)
            self.assertEqual(row["hit_rate"], 100.0)
            self.assertEqual(row["quality_filtered_total"], 1)
            self.assertEqual(row["quality_filter_mode"], "exclude_flagged")
            self.assertEqual(row["raw_total"], 2)


if __name__ == "__main__":
    unittest.main()
