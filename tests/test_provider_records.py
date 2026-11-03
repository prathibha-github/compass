import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from compass.clients.base import CompletionResponse
from compass.clients.settings import client_request_settings, default_request_settings
from compass.clients.openai_responses import OpenAIResponsesClient
from compass.detectors.base import TicSuite, StylePrompt, StyleCondition
from compass.detectors.heuristic import RegexDetector
from compass.evaluation.suite_runner import generate_suite_completions
from compass.benchmark.runner import _generation_quality_from_record


def test_responses_effective_settings_and_finish_reason():
    sdk = MagicMock()
    sdk.RateLimitError = type('RateLimitError', (Exception,), {})
    sdk.APIError = type('APIError', (Exception,), {})
    with patch.dict('sys.modules', {'openai': sdk}):
        client = OpenAIResponsesClient('gpt-5-mini', api_key='fake',
            required_temperature=1, max_output_token_multiplier=10)
    client.client.responses.create.return_value = SimpleNamespace(
        output_text='A complete sentence.', usage=SimpleNamespace(input_tokens=10, output_tokens=180),
        status='incomplete', incomplete_details=SimpleNamespace(reason='max_output_tokens'))
    response = client.complete('task', max_tokens=20, temperature=0)
    sent = client.client.responses.create.call_args.kwargs
    assert response.request_settings['effective']['temperature'] == sent['temperature'] == 1
    assert response.request_settings['effective']['max_tokens'] == sent['max_output_tokens'] == 200
    assert response.finish_reason == 'max_output_tokens'
    assert response.request_settings == default_request_settings('gpt-5-mini', 20, 0, responses_api=True)


def test_suite_quality_uses_effective_budget(tmp_path):
    suite = TicSuite('s', [StylePrompt('p','task')], [StyleCondition('c','')], [RegexDetector('r','rest')], 'c')
    settings = default_request_settings('gpt-5-mini', 150, 0, responses_api=True)
    response = CompletionResponse('A complete sentence.', {'input': 10, 'output': 200},
                                  request_settings=settings)
    path = generate_suite_completions(suite, ['gpt-5-mini'], 1, tmp_path,
        client_factory=lambda _: Mock(complete=Mock(return_value=response)))
    row = json.loads(path.read_text())
    assert row['max_tokens_requested'] == 150
    assert row['max_tokens_effective'] == 1500
    assert row['request_settings'] == settings
    assert row['hit_token_cap'] is False
    assert _generation_quality_from_record(row)['hit_token_cap'] is False


def test_custom_client_settings_are_unknown():
    assert client_request_settings(Mock(), 10, 0)['effective'] is None
    assert client_request_settings(Mock(), 10, 0)['verified'] is False


def test_provider_policy_changes_judge_cache_coordinates(tmp_path):
    from compass.judges import LLMJudge, JudgeConfig
    from compass import RubricLibrary, EvaluationCache
    sdk = MagicMock()
    with patch.dict('sys.modules', {'openai': sdk}):
        client = OpenAIResponsesClient('gpt-5-mini', api_key='fake')
    judge = LLMJudge(JudgeConfig(RubricLibrary.clarity, 'gpt-5-mini'), client,
                     EvaluationCache(str(tmp_path)))
    original = judge._cache_coordinates('text')
    client._max_output_token_multiplier = 10
    assert judge._cache_coordinates('text') != original
    changed_budget = judge._cache_coordinates('text')
    client._required_temperature = 1
    assert judge._cache_coordinates('text') != changed_budget
