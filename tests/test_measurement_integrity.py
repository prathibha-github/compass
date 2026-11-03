"""Regressions for failed judgments and stale suite artifacts (no API calls)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from compass.cache import EvaluationCache
from compass.detectors.base import StyleCondition, StylePrompt, TicSuite, summarize_detector_records
from compass.detectors.heuristic import RegexDetector
from compass.detectors.llm_detector import LLMJudgeDetector
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions, summarize_suite_evaluations
from compass.judges import JudgeConfig, JudgeResponseError, LLMJudge
from compass.rubrics import RubricLibrary


class Client:
    def __init__(self, output="take a break"):
        self.output, self.calls = output, 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(completion=self.output)


def suite():
    return TicSuite(
        "test", [StylePrompt("p", "Help me")], [StyleCondition("c", "Be helpful")],
        [RegexDetector("rest", "sleep")], "c",
    )


@pytest.mark.parametrize("raw", ["bad", "", "[]", "null", "3", "{}", '{"score":true}',
                                     '{"score":null}', '{"score":"bad"}', '{"score":NaN}',
                                     '{"score":Infinity}', '{"score":"-inf"}'])
def test_invalid_judgments_are_not_cached_and_can_be_retried(tmp_path, raw):
    client = Client(raw)
    judge = LLMJudge(JudgeConfig(RubricLibrary.sycophancy, "judge"), client, EvaluationCache(str(tmp_path)))
    with pytest.raises(JudgeResponseError):
        judge.evaluate("response")
    with pytest.raises(JudgeResponseError):
        LLMJudgeDetector("x", "rubric").parse_response(raw)
    client.output = '{"score":0}'
    result = judge.evaluate("response")
    assert result.score == 0 and not result.hit and not result.cache_hit
    assert client.calls == 2
    assert judge.evaluate("response").cache_hit
    assert client.calls == 2


@pytest.mark.parametrize("score,hit", [(0.9, False), (0.2, True), (0.5, False)])
def test_both_judge_paths_use_the_configured_threshold(tmp_path, score, hit):
    rubric = RubricLibrary.sycophancy
    raw = json.dumps({"score": score, "hit": hit})
    judge = LLMJudge(JudgeConfig(rubric, "judge"), Client(raw), EvaluationCache(str(tmp_path)))
    direct = LLMJudgeDetector("x", rubric.text, rubric.hit_threshold).parse_response(raw)
    assert direct.hit == judge.evaluate("response").hit == (score >= rubric.hit_threshold)


@pytest.mark.parametrize("change", ["prompt", "system", "temperature", "legacy"])
def test_generation_resume_rejects_changed_inputs_before_calls(tmp_path, change):
    original = suite()
    client = Client()
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: client)
    altered, kwargs = original, {}
    if change == "prompt":
        altered = replace(original, prompts=[replace(original.prompts[0], text="Different")])
    elif change == "system":
        altered = replace(original, conditions=[replace(original.conditions[0], system_prompt="Different")])
    elif change == "temperature":
        kwargs["temperature"] = 0.7
    else:
        row = json.loads(path.read_text())
        del row["generation_contract"]
        path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot resume generation"):
        generate_suite_completions(altered, ["m"], 1, tmp_path, resume=True, client_factory=lambda _: client, **kwargs)
    assert client.calls == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["rule", "completion", "context", "legacy"])
def test_evaluation_resume_rejects_stale_scores(tmp_path, change):
    original = suite()
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: Client())
    output = evaluate_suite_completions(original, path, tmp_path)
    if change == "rule":
        original = replace(original, detectors=[RegexDetector("rest", "break")])
    elif change == "legacy":
        row = json.loads(output.read_text())
        del row["evaluation_contract"]
        output.write_text(json.dumps(row) + "\n")
    else:
        row = json.loads(path.read_text())
        row["completion" if change == "completion" else "prompt_text"] = "changed"
        path.write_text(json.dumps(row) + "\n")
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Cannot resume evaluation"):
        evaluate_suite_completions(original, path, tmp_path)
    assert output.read_bytes() == before
    evaluate_suite_completions(original, path, tmp_path, resume=False)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["hit"] == (change == "rule")


def test_unchanged_resume_and_appending_samples(tmp_path):
    original, client = suite(), Client()
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: client)
    output = evaluate_suite_completions(original, path, tmp_path)
    before = output.read_bytes()
    evaluate_suite_completions(original, path, tmp_path)
    assert output.read_bytes() == before
    generate_suite_completions(original, ["m"], 2, tmp_path, resume=True, client_factory=lambda _: client)
    evaluate_suite_completions(original, path, tmp_path)
    assert client.calls == 2
    assert len(output.read_text().splitlines()) == 2


def test_failed_suite_judgments_do_not_checkpoint_or_report_success(tmp_path):
    original = replace(suite(), detectors=[LLMJudgeDetector("rest", "Score rest")])
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: Client())
    client = Client("not JSON")
    with pytest.raises(RuntimeError, match="1 runtime failures"):
        evaluate_suite_completions(original, path, tmp_path, judge_model="judge", client_factory=lambda _: client)
    output = tmp_path / "suite_evaluations_judge.jsonl"
    assert not output.exists() or not output.read_text().strip()
    client.output = '{"score":1}'
    evaluate_suite_completions(original, path, tmp_path, judge_model="judge", client_factory=lambda _: client)
    assert json.loads(output.read_text())["hit"] is True
    assert client.calls == 2


def test_missing_detector_result_is_not_a_negative():
    with pytest.raises(ValueError, match="incomplete detector"):
        summarize_detector_records({"c": [{}]}, suite())


def test_offline_summary_rejects_wholly_missing_sample(tmp_path):
    original = suite()
    path = generate_suite_completions(original, ["m"], 2, tmp_path, client_factory=lambda _: Client())
    output = evaluate_suite_completions(original, path, tmp_path)
    rows = output.read_text().splitlines()
    output.write_text(rows[0] + "\n")
    with pytest.raises(ValueError, match="Incomplete evaluation coverage"):
        summarize_suite_evaluations(output, original)
    evaluate_suite_completions(original, path, tmp_path)
    assert summarize_suite_evaluations(output, original)["m"]["c"]["n_outputs"] == 2


def test_generation_failure_stops_pipeline_and_retry_keeps_successes(tmp_path):
    original = suite()
    class PartialClient(Client):
        def complete(self, *args, **kwargs):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("provider failure")
            return super().complete(*args, **kwargs)
    client = PartialClient()
    with pytest.raises(RuntimeError, match="1 runtime failures"):
        generate_suite_completions(original, ["m"], 2, tmp_path, client_factory=lambda _: client)
    with pytest.raises(ValueError, match="Incomplete or incompatible generations"):
        evaluate_suite_completions(original, tmp_path / "suite_generations.jsonl", tmp_path)
    path = generate_suite_completions(original, ["m"], 2, tmp_path, resume=True, client_factory=lambda _: client)
    assert client.calls == 3
    assert len(path.read_text().splitlines()) == 2


def test_legacy_zero_cache_entry_is_not_reused(tmp_path):
    import hashlib
    from compass.judges import EvaluationResult
    cache = EvaluationCache(str(tmp_path))
    config = JudgeConfig(RubricLibrary.sycophancy, "judge")
    cache.put(config.config_hash, hashlib.sha256(b"response").hexdigest()[:16], "1.0",
              EvaluationResult(name="sycophancy", score=0, hit=False))
    client = Client('{"score":1}')
    result = LLMJudge(config, client, cache).evaluate("response")
    assert result.hit and not result.cache_hit
    assert client.calls == 1


@pytest.mark.parametrize("duplicate", ["prompt", "condition", "model"])
def test_duplicate_ids_fail_before_reset_or_calls(tmp_path, duplicate):
    original, client = suite(), Client()
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: client)
    before = path.read_bytes()
    models = ["m"]
    if duplicate == "prompt":
        original = replace(original, prompts=original.prompts + [replace(original.prompts[0], text="different")])
    elif duplicate == "condition":
        original = replace(original, conditions=original.conditions * 2)
    else:
        models *= 2
    with pytest.raises(ValueError, match="must be unique"):
        generate_suite_completions(original, models, 1, tmp_path, client_factory=lambda _: client)
    assert path.read_bytes() == before and client.calls == 1


def test_duplicate_generation_artifacts_are_rejected(tmp_path):
    original = suite()
    path = generate_suite_completions(original, ["m"], 1, tmp_path, client_factory=lambda _: Client())
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate generation identity"):
        evaluate_suite_completions(original, path, tmp_path)
