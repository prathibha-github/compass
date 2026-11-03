import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from compass.cache import EvaluationCache
from compass.detectors.base import StylePrompt, StyleCondition, TicSuite, analyze_text
from compass.detectors.llm_detector import LLMJudgeDetector, LLM_JUDGE_SYSTEM_PROMPT
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions
from compass.judges import LLMJudge, JudgeConfig
from compass.rubrics.base import Rubric


def judge(tmp_path):
    client = Mock(complete=Mock(return_value=SimpleNamespace(
        completion='{"score": 0.8, "confidence": 0.9}', tokens_used={}, cost_usd=0)))
    rubric = Rubric(name='rest', text='Was rest requested?', category='test', version='1', created_at='2026-11-02')
    return LLMJudge(JudgeConfig(rubric, 'judge'), client, EvaluationCache(str(tmp_path/'cache'))), client


def test_both_paths_build_identical_prompts_and_results(tmp_path):
    llm, client = judge(tmp_path)
    detector = LLMJudgeDetector('rest', llm.config.rubric.text)
    captured = []
    def judge_fn(prompt, max_tokens):
        captured.append(prompt)
        return client.complete.return_value.completion
    direct = detector.detect_with_judge('Take a break', judge_fn, context='Please suggest rest', system_context='Be helpful')
    result = llm.evaluate('Take a break', context='Please suggest rest', system_context='Be helpful')
    assert captured == [client.complete.call_args.args[0]]
    assert LLM_JUDGE_SYSTEM_PROMPT == LLMJudge.SYSTEM_PROMPT
    assert direct.hit == result.hit
    assert direct.metadata['judge_context']['user_request'] == 'Please suggest rest'


def test_context_cache_keys_do_not_collide(tmp_path):
    llm, client = judge(tmp_path)
    assert llm._cache_coordinates('a\0b') != llm._cache_coordinates('b', context='a')
    for context, system in [(None, None), ('', None), ('request', None), ('request', 'system')]:
        llm.evaluate('same response', context=context, system_context=system)
        llm.evaluate('same response', context=context, system_context=system)
    assert client.complete.call_count == 4


def suite():
    return TicSuite('test', [StylePrompt('p', 'Saved request')], [StyleCondition('c', 'Saved system')],
                    [LLMJudgeDetector('rest', 'Was rest requested?')], 'c')


def test_context_modes_use_saved_inputs_and_invalidate_resume(tmp_path):
    definition = suite()
    llm, fake = judge(tmp_path)
    generations = generate_suite_completions(definition, ['m'], 1, tmp_path, client_factory=lambda m: fake)
    # Current definitions are not a substitute for the saved request.
    definition.prompts[0] = StylePrompt('p', 'A different current request')
    path = evaluate_suite_completions(definition, generations, tmp_path, judge_model='judge', client_factory=lambda m: fake)
    row = json.loads(path.read_text())
    assert row['judge_context'] == {'mode': 'user', 'user_request': 'Saved request', 'system_prompt': None}
    assert 'Saved request' in fake.complete.call_args.args[0]
    assert 'Saved system' not in fake.complete.call_args.args[0]
    before = path.read_bytes()
    with pytest.raises(ValueError, match='inputs changed'):
        evaluate_suite_completions(definition, generations, tmp_path, judge_model='judge', judge_context_mode='user_and_system')
    assert path.read_bytes() == before
    evaluate_suite_completions(definition, generations, tmp_path, judge_model='judge',
                               judge_context_mode='user_and_system', resume=False, client_factory=lambda m: fake)
    assert 'Saved system' in fake.complete.call_args.args[0]


def test_missing_request_requires_explicit_completion_only_mode(tmp_path):
    definition = suite()
    generations = tmp_path/'archive.jsonl'
    generations.write_text(json.dumps(dict(model='m', suite='test', prompt_id='p', condition='c',
                                           sample_idx=0, completion='Take a break'))+'\n')
    with pytest.raises(ValueError, match='Saved user request is missing'):
        evaluate_suite_completions(definition, generations, tmp_path, judge_model='judge')
    llm, fake = judge(tmp_path)
    path = evaluate_suite_completions(definition, generations, tmp_path, judge_model='judge',
                                      judge_context_mode='completion', client_factory=lambda m: fake)
    assert json.loads(path.read_text())['judge_context']['mode'] == 'completion'


def test_benchmark_context_mode_invalidates_saved_scores(tmp_path):
    from compass.benchmark.runner import generate_completions, evaluate_completions
    from compass.benchmark.specs import BenchmarkPrompt, BenchmarkSpec
    llm, fake = judge(tmp_path)
    spec = BenchmarkSpec('test', '1', {'rest': (BenchmarkPrompt('p', 'Saved request', 'task'),)}, {'rest': llm.config.rubric})
    with patch('compass.benchmark.runner._create_client', return_value=fake):
        generations = generate_completions(['m'], spec, 1, tmp_path).path
        path = evaluate_completions(generations, spec, 'judge', tmp_path).path
    assert json.loads(path.read_text())['judge_context']['user_request'] == 'Saved request'
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='Inputs changed'):
        evaluate_completions(generations, spec, 'judge', tmp_path, judge_context_mode='completion')
    factory.assert_not_called()
