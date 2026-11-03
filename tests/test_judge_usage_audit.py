import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import pytest

from compass import EvaluationCache, RubricLibrary
from compass.clients.base import CompletionResponse
from compass.clients.openai_responses import OpenAIResponsesClient
from compass.clients.usage import CompletionFailure
from compass.judges import LLMJudge, JudgeConfig, JudgeResponseError
from compass.reproducibility.versioning import cost_summary, cost_per_judge, EvaluationMetadata


def make_judge(tmp_path, text='{"score": 0.3}'):
    response = CompletionResponse(text, {'input': 100, 'output': 50}, .003,
        request_settings={'effective': {'seed': None, 'temperature': 1}}, usage_complete=True)
    return LLMJudge(JudgeConfig(RubricLibrary.clarity, 'm', seed=123),
                    Mock(complete=Mock(return_value=response)), EvaluationCache(str(tmp_path)))


def test_fresh_memory_and_disk_cache_have_independent_usage(tmp_path):
    judge = make_judge(tmp_path)
    fresh = judge.evaluate('answer')
    memory = judge.evaluate('answer')
    judge.cache = EvaluationCache(str(tmp_path))
    disk = judge.evaluate('answer')
    assert fresh.cache_hit is False
    assert fresh.cost_usd == .003
    for reused in [memory, disk]:
        assert reused.cache_hit is True
        assert reused.cost_usd == 0
        assert reused.tokens_used == {'input': 0, 'output': 0}
        assert reused.source_cost_usd == .003
        assert reused.source_tokens_used == {'input': 100, 'output': 50}
    memory.request_settings['effective']['temperature'] = 99
    fresh.tokens_used['input'] = 999
    assert judge.evaluate('answer').source_tokens_used['input'] == 100
    assert judge.evaluate('answer').request_settings['effective']['temperature'] == 1
    fresh.tokens_used['input'] = 100
    assert cost_summary([fresh, memory, disk])['total_cost_usd'] == .003
    assert cost_summary([fresh, memory, disk])['total_input_tokens'] == 100
    assert cost_per_judge([fresh, memory, disk])['m']['total_cost_usd'] == .003
    metadata = EvaluationMetadata.from_result(fresh, 'test')
    assert metadata.seed is None and metadata.requested_seed == 123
    assert json.loads(json.dumps(disk.to_dict()))['requested_seed'] == 123


def test_invalid_judgment_keeps_raw_response_and_usage_but_no_cache(tmp_path):
    judge = make_judge(tmp_path, 'not a score')
    with pytest.raises(JudgeResponseError) as caught:
        judge.evaluate('answer')
    assert caught.value.audit_record['raw_response'] == 'not a score'
    assert caught.value.audit_record['cost_usd'] == .003
    assert caught.value.audit_record['tokens_used']['input'] == 100
    assert judge.cache.stats()['cached_files'] == 0


def responses_client():
    sdk = MagicMock()
    sdk.RateLimitError = type('RateLimitError', (Exception,), {})
    sdk.APIError = type('APIError', (Exception,), {})
    with patch.dict('sys.modules', {'openai': sdk}):
        return OpenAIResponsesClient('gpt-5-mini', api_key='fake')


def response(text, tokens):
    return SimpleNamespace(output_text=text, usage=SimpleNamespace(input_tokens=10, output_tokens=tokens),
                           status='completed', incomplete_details=None)


def test_empty_retry_usage_is_separate_from_final_response_tokens():
    client = responses_client()
    client.client.responses.create.side_effect = [response('', 7), response('answer', 3)]
    with patch('time.sleep'):
        actual = client.complete('task')
    assert actual.tokens_used == {'input': 10, 'output': 3}
    assert actual.billing_tokens == client.total_tokens == {'input': 20, 'output': 10}
    assert actual.cost_usd == pytest.approx(client.total_cost_usd)
    assert len(actual.usage_attempts) == 2 and actual.usage_complete


def test_exhausted_empty_retries_preserve_usage():
    client = responses_client()
    client.client.responses.create.return_value = response('', 7)
    with patch('time.sleep'), pytest.raises(CompletionFailure) as caught:
        client.complete('task')
    audit = caught.value.audit_record
    assert audit['billing_tokens'] == {'input': 100, 'output': 70}
    assert len(audit['usage_attempts']) == 10
    assert audit['usage_complete'] is True


def test_suite_failed_parse_writes_audit_and_retry_writes_full_result(tmp_path):
    from compass.detectors import TicSuite, StylePrompt, StyleCondition, LLMJudgeDetector
    from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions
    suite = TicSuite('s', [StylePrompt('p','Task')], [StyleCondition('c','')],
                     [LLMJudgeDetector('r','Rubric')], 'c')
    fake = Mock(complete=Mock(return_value=CompletionResponse('answer')))
    generations = generate_suite_completions(suite, ['m'], 1, tmp_path, client_factory=lambda _: fake)
    fake.complete.return_value = CompletionResponse('bad JSON', {'input':10, 'output':3}, .001)
    with pytest.raises(RuntimeError):
        evaluate_suite_completions(suite, generations, tmp_path, judge_model='j', client_factory=lambda _:fake)
    failure = json.loads(next(tmp_path.glob('suite_evaluations*.failures.jsonl')).read_text())
    assert failure['audit']['raw_response'] == 'bad JSON'
    assert failure['audit']['cost_usd'] == .001
    fake.complete.return_value = CompletionResponse('{"score":0}', {'input':10, 'output':3}, .001)
    scored = evaluate_suite_completions(suite, generations, tmp_path, judge_model='j', resume=True, client_factory=lambda _:fake)
    row = json.loads(scored.read_text())
    assert row['judge_result']['raw_response'] == '{"score":0}'
    assert row['judge_result']['cost_usd'] == .001
    assert failure['audit']['raw_response'] in next(tmp_path.glob('suite_evaluations*.failures.jsonl')).read_text()


@pytest.mark.parametrize('status', [400, 500, 429])
def test_chat_provider_errors_keep_usage_and_judge_provenance(tmp_path, status):
    from compass.clients.openai import OpenAIClient
    sdk = MagicMock()
    class APIError(Exception):
        status_code = status
    class RateLimitError(Exception):
        response = None
    sdk.APIError, sdk.RateLimitError = APIError, RateLimitError
    with patch.dict('sys.modules', {'openai': sdk}):
        client = OpenAIClient('gpt-4o-mini', api_key='fake')
    client.client.chat.completions.with_raw_response.create.side_effect = RateLimitError('limited') if status==429 else APIError('failed')
    judge = LLMJudge(JudgeConfig(RubricLibrary.clarity, 'gpt-4o-mini', seed=123), client, EvaluationCache(str(tmp_path)))
    with patch('time.sleep'), pytest.raises(CompletionFailure) as caught:
        judge.evaluate('answer')
    audit = caught.value.audit_record
    assert audit['usage_complete'] is False
    assert len(audit['usage_attempts']) == (1 if status == 400 else 10)
    assert audit['request_settings']['model'] == 'gpt-4o-mini'
    assert audit['requested_seed'] == 123
    assert audit['rubric_hash'] == RubricLibrary.clarity.hash


def test_billed_response_followed_by_rate_limits_is_incomplete():
    client = responses_client()
    error = client._openai.RateLimitError('limited')
    client.client.responses.create.side_effect = [response('', 7)] + [error]*9
    with patch('time.sleep'), pytest.raises(CompletionFailure) as caught:
        client.complete('task')
    audit = caught.value.audit_record
    assert len(audit['usage_attempts']) == 10
    assert audit['billing_tokens'] == {'input': 10, 'output': 7}
    assert audit['cost_usd'] > 0
    assert audit['usage_complete'] is False
