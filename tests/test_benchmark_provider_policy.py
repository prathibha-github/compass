from unittest.mock import patch

from compass import RubricLibrary
from compass.benchmark.resume import generation_contracts
from compass.benchmark.runner import _generation_quality_from_record
from compass.benchmark.specs import BenchmarkSpec, BenchmarkPrompt


def test_benchmark_uses_effective_budget_when_rescoring():
    row = {'completion': 'This is a complete sentence.', 'tokens_used': {'output': 200},
           'max_tokens_requested': 150, 'max_tokens_effective': 1500, 'finish_reason': 'completed'}
    assert _generation_quality_from_record(row)['hit_token_cap'] is False
    row['finish_reason'] = 'max_output_tokens'
    assert _generation_quality_from_record(row)['hit_token_cap'] is True


def test_benchmark_provider_policy_changes_generation_contract():
    spec = BenchmarkSpec('b', '1', {'clarity': (BenchmarkPrompt('p', 'Task', 'task'),)},
                         {'clarity': RubricLibrary.clarity})
    original = generation_contracts(spec, ['gpt-5-mini'], 1, {'gpt-5-mini': 150})
    with patch('compass.benchmark.resume.default_request_settings', return_value={'policy_version': 'changed'}):
        assert generation_contracts(spec, ['gpt-5-mini'], 1, {'gpt-5-mini': 150}) != original
