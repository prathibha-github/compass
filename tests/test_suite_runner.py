"""Tests for the two-phase, condition-aware suite runner."""

import json
import tempfile
import unittest
from pathlib import Path

from compass.clients.base import CompletionResponse
from compass.detectors.base import StyleCondition, StylePrompt, TicSuite
from compass.detectors.heuristic import PhraseSetDetector
from compass.detectors.llm_detector import LLMJudgeDetector
from compass.cache import EvaluationCache
from compass.judges import JudgeConfig, LLMJudge
from compass.rubrics import RubricLibrary
from compass.rubrics.base import Rubric
from compass.evaluation.suite_io import (
    load_suite_generations,
    migrate_suite_generation_record,
    suite_generation_identity,
)
from compass.evaluation.suite_runner import (
    evaluate_suite_completions,
    generate_suite_completions,
    summarize_suite_evaluations,
)


def _heuristic_suite():
    return TicSuite(
        name="rest_test",
        prompts=[
            StylePrompt(id="neutral", text="Fix the bug.", task_type="t"),
            StylePrompt(id="fatigue", text="I'm exhausted. Fix the bug.", task_type="t"),
        ],
        conditions=[
            StyleCondition(name="default", system_prompt="You are helpful."),
            StyleCondition(name="strict", system_prompt="Only answer the task."),
        ],
        detectors=[PhraseSetDetector(name="rest_suggestion", phrases={"take a break"})],
        baseline_condition="default",
    )


class _CountingClient:
    """Generation client whose completion encodes prompt+system, with a call count."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, max_tokens=200, temperature=0.0, system=None):
        self.calls += 1
        # Fatigue prompts get a rest suggestion; neutral prompts stay on task.
        body = "take a break" if "exhausted" in prompt else "here is the fix"
        # Unique per call so distinct samples are distinct completions (as at temp>0).
        return CompletionResponse(
            completion=f"{body} [{system}|{prompt}] #{self.calls}",
            tokens_used={"input": 1, "output": 1},
            cost_usd=0.0,
        )


class _RaisingClient:
    def complete(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("evaluation must not call a generation client")


class GenerationPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_generation_persists_completion_and_resumes(self):
        suite = _heuristic_suite()
        client = _CountingClient()
        gen_path = generate_suite_completions(
            suite, ["m"], samples=2, output_dir=self.out, client_factory=lambda _m: client
        )
        rows = load_suite_generations(gen_path)
        # 2 prompts x 2 conditions x 2 samples
        self.assertEqual(client.calls, 8)
        self.assertEqual(len(rows), 8)
        for row in rows.values():
            self.assertIn("completion", row)
            self.assertTrue(row["completion"])

        # Re-running with resume keeps prior work and makes no new calls.
        client2 = _CountingClient()
        generate_suite_completions(
            suite, ["m"], samples=2, output_dir=self.out,
            resume=True, client_factory=lambda _m: client2,
        )
        self.assertEqual(client2.calls, 0)

    def test_fresh_run_resets_prior_generations(self):
        suite = _heuristic_suite()
        generate_suite_completions(
            suite, ["m"], samples=1, output_dir=self.out,
            client_factory=lambda _m: _CountingClient(),
        )
        gen_path = generate_suite_completions(
            suite, ["m"], samples=1, output_dir=self.out,
            resume=False, client_factory=lambda _m: _CountingClient(),
        )
        # 2 prompts x 2 conditions x 1 sample, not doubled.
        self.assertEqual(len(load_suite_generations(gen_path)), 4)


class EvaluationPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _generate(self, suite):
        return generate_suite_completions(
            suite, ["m"], samples=2, output_dir=self.out,
            client_factory=lambda _m: _CountingClient(),
        )

    def test_heuristic_eval_does_not_call_a_client(self):
        suite = _heuristic_suite()
        gen_path = self._generate(suite)
        eval_path = evaluate_suite_completions(
            suite, gen_path, self.out, client_factory=_RaisingClient,
        )
        rows = [json.loads(l) for l in open(eval_path) if l.strip()]
        # 8 generations x 1 detector
        self.assertEqual(len(rows), 8)
        # Fatigue cells hit, neutral cells do not.
        by_prompt = {}
        for r in rows:
            by_prompt.setdefault(r["prompt_id"], []).append(r["hit"])
        self.assertTrue(all(by_prompt["fatigue"]))
        self.assertFalse(any(by_prompt["neutral"]))

    def test_summarize_groups_by_condition(self):
        suite = _heuristic_suite()
        gen_path = self._generate(suite)
        eval_path = evaluate_suite_completions(suite, gen_path, self.out)
        summary = summarize_suite_evaluations(eval_path, suite)
        self.assertEqual(set(summary), {"default", "strict"})
        # Each condition: 4 samples (2 prompts x 2), half fatigue -> 50% hit.
        for cond in ("default", "strict"):
            det = summary[cond]["detectors"]["rest_suggestion"]
            self.assertEqual(summary[cond]["n_outputs"], 4)
            self.assertEqual(det["pct_hit"], 50.0)
            self.assertIn("pct_hit_ci_low", det)


class _CountingJudgeClient:
    def __init__(self):
        self.calls = 0
        self.last_prompt = ""

    def complete(self, prompt, max_tokens=180, temperature=0.0, system=None):
        self.calls += 1
        self.last_prompt = prompt
        # Score 1.0 only when the user request is actually visible to the judge.
        score = 1.0 if "User request:" in prompt else 0.0
        return CompletionResponse(
            completion=json.dumps({"score": score, "confidence": 0.9, "rationale": "x"}),
            tokens_used={"input": 1, "output": 1},
            cost_usd=0.0,
        )


class JudgeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _judge_suite(self):
        return TicSuite(
            name="rest_judge_test",
            prompts=[StylePrompt(id="p1", text="Fix the bug.", task_type="t")],
            conditions=[StyleCondition(name="default", system_prompt="Be helpful.")],
            detectors=[
                LLMJudgeDetector(
                    name="rest_judge", rubric=RubricLibrary.unsolicited_rest.text
                )
            ],
            baseline_condition="default",
        )

    def test_judge_sees_prompt_context_and_caches_across_runs(self):
        suite = self._judge_suite()
        gen_path = generate_suite_completions(
            suite, ["m"], samples=3, output_dir=self.out,
            client_factory=lambda _m: _CountingClient(),
        )
        judge = _CountingJudgeClient()

        eval_path = evaluate_suite_completions(
            suite, gen_path, self.out, judge_model="judge-x",
            resume=False, client_factory=lambda _m: judge,
        )
        # 3 distinct completions -> 3 judge calls, all hit (context was visible).
        self.assertEqual(judge.calls, 3)
        self.assertIn("User request:", judge.last_prompt)
        rows = [json.loads(l) for l in open(eval_path) if l.strip()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["hit"] for r in rows))

        # Re-score the same completions: checkpoint reset but cache persists -> 0 calls.
        judge2 = _CountingJudgeClient()
        evaluate_suite_completions(
            suite, gen_path, self.out, judge_model="judge-x",
            resume=False, client_factory=lambda _m: judge2,
        )
        self.assertEqual(judge2.calls, 0)

    def test_judge_required_when_suite_has_judge_detector(self):
        suite = self._judge_suite()
        gen_path = generate_suite_completions(
            suite, ["m"], samples=1, output_dir=self.out,
            client_factory=lambda _m: _CountingClient(),
        )
        with self.assertRaisesRegex(ValueError, "judge_model is required"):
            evaluate_suite_completions(suite, gen_path, self.out, judge_model=None)


class SchemaAndRubricTests(unittest.TestCase):
    def test_generation_record_requires_completion(self):
        base = {"model": "m", "suite": "s", "prompt_id": "p", "condition": "c"}
        with self.assertRaises(ValueError):
            migrate_suite_generation_record(base)
        ok = migrate_suite_generation_record({**base, "completion": "hi", "sample_idx": 2})
        self.assertEqual(
            suite_generation_identity(ok), ("m", "s", "p", "c", 2)
        )

    def test_unsolicited_rest_rubric_registered(self):
        self.assertIn("unsolicited_rest", RubricLibrary.all_rubrics())
        self.assertIs(
            RubricLibrary.get("unsolicited_rest"), RubricLibrary.unsolicited_rest
        )
        # Prompt-aware: the rubric instructs the judge using the user request.
        self.assertIn("user's request", RubricLibrary.unsolicited_rest.text)


class LLMJudgeContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = EvaluationCache(cache_dir=str(Path(self.tmp.name) / ".cache"))
        self.rubric = Rubric(
            name="r", category="c", version="1.0", created_at="2026-05-30", text="rt"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_context_changes_prompt_and_cache_key(self):
        judge = LLMJudge(
            JudgeConfig(rubric=self.rubric, judge_model="j"),
            client=_CountingJudgeClient(),
            cache=self.cache,
        )
        self.assertNotIn("User request:", judge._build_prompt("resp"))
        self.assertIn("User request:\nasked", judge._build_prompt("resp", context="asked"))
        self.assertNotEqual(
            judge._cache_coordinates("resp"),
            judge._cache_coordinates("resp", context="asked"),
        )


if __name__ == "__main__":
    unittest.main()
