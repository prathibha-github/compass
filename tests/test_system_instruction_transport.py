from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from compass.clients.google_ai import GoogleAIClient
from compass.clients.ollama import OllamaClient
from compass.clients.settings import default_request_settings
from compass.evaluation.contracts import fingerprint


@pytest.mark.parametrize('system', [None, '', 'Follow this condition.'])
@pytest.mark.parametrize('provider', ['google_ai', 'ollama'])
def test_native_system_fields(provider, system):
    sdk = MagicMock()
    if provider == 'google_ai':
        pkg = MagicMock(genai=sdk)
        sdk.types.GenerateContentConfig.side_effect = lambda **kw: kw
        with patch.dict('sys.modules', {'google': pkg, 'google.genai': sdk}):
            client = GoogleAIClient('gemini-test', api_key='fake', request_interval=0)
        call = client.client.models.generate_content
        call.return_value = SimpleNamespace(text='Answer.', candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=2, candidates_token_count=2))
    else:
        with patch.dict('sys.modules', {'ollama': sdk}):
            client = OllamaClient('local-test', request_interval=0)
        call = client.api_client.generate
        call.return_value = {'response': 'Answer.'}
    response = client.complete('User task', system=system)
    sent = call.call_args.kwargs
    instructions = sent['config'] if provider == 'google_ai' else sent
    field = 'system_instruction' if provider == 'google_ai' else 'system'
    assert sent['contents' if provider == 'google_ai' else 'prompt'] == 'User task'
    if system is None:
        assert field not in instructions
    else:
        assert instructions[field] == system
    settings = client.request_settings(180, 0)
    assert settings == default_request_settings(client.model, 180, 0)
    assert response.request_settings['policy']['system_instruction_transport'] == 'native_v1'
    old = {**settings, 'policy': {k:v for k,v in settings['policy'].items()
                               if k != 'system_instruction_transport'}}
    assert fingerprint(old) != fingerprint(settings)
