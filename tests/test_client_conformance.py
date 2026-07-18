"""Shared conformance tests for completion clients."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from compass.clients.base import CompletionResponse


class ClientConformanceTests(unittest.TestCase):
    def _assert_response_contract(self, response, input_tokens, output_tokens):
        self.assertIsInstance(response, CompletionResponse)
        self.assertIsInstance(response.completion, str)
        self.assertTrue(response.completion)
        self.assertEqual(response.tokens_used, {"input": input_tokens, "output": output_tokens})
        self.assertEqual(set(response.tokens_used.keys()), {"input", "output"})
        self.assertIsInstance(response.tokens_used["input"], int)
        self.assertIsInstance(response.tokens_used["output"], int)
        self.assertGreaterEqual(response.tokens_used["input"], 0)
        self.assertGreaterEqual(response.tokens_used["output"], 0)
        self.assertIsInstance(response.cost_usd, float)
        self.assertGreaterEqual(response.cost_usd, 0.0)

    def test_openai_client_conforms(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient

            client = OpenAIClient(model="gpt-4o-mini", api_key="fake")

        resp_mock = MagicMock()
        resp_mock.choices = [MagicMock(message=MagicMock(content="answer"), logprobs=None)]
        resp_mock.usage = MagicMock(prompt_tokens=11, completion_tokens=7)
        raw_mock = MagicMock()
        raw_mock.parse.return_value = resp_mock
        raw_mock.headers = {}
        client.client.chat.completions.with_raw_response.create.return_value = raw_mock
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=33,
                temperature=0.4,
                system="system prompt",
            )

        kwargs = client.client.chat.completions.with_raw_response.create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 33)
        self.assertEqual(kwargs["temperature"], 0.4)
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(kwargs["messages"][0]["content"], "system prompt")
        self._assert_response_contract(response, 11, 7)
        self.assertEqual(client.total_tokens, {"input": 11, "output": 7})

    def test_openai_responses_client_conforms(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai_responses import OpenAIResponsesClient

            client = OpenAIResponsesClient(
                model="gpt-5-mini",
                api_key="fake",
                max_output_token_multiplier=10,
            )

        resp_mock = MagicMock()
        resp_mock.output_text = "answer"
        resp_mock.usage = MagicMock(input_tokens=13, output_tokens=5)
        client.client.responses.create.return_value = resp_mock
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=12,
                temperature=0.0,
                system="system prompt",
            )

        kwargs = client.client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["instructions"], "system prompt")
        self.assertEqual(kwargs["input"], "prompt")
        self.assertEqual(kwargs["max_output_tokens"], 120)
        self._assert_response_contract(response, 13, 5)
        self.assertEqual(client.total_tokens, {"input": 13, "output": 5})

    def test_anthropic_client_conforms(self):
        anthropic_mock = MagicMock()
        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            from compass.clients.anthropic import AnthropicClient

            client = AnthropicClient(model="claude-haiku-4-5", api_key="fake")

        response_mock = MagicMock()
        response_mock.usage = MagicMock(input_tokens=9, output_tokens=4)
        response_mock.content = [MagicMock(type="text", text="answer")]
        client.client.messages.create.return_value = response_mock
        anthropic_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        anthropic_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=22,
                temperature=0.3,
                system="system prompt",
            )

        kwargs = client.client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"], "system prompt")
        self.assertEqual(kwargs["max_tokens"], 22)
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["messages"][0]["content"], "prompt")
        self._assert_response_contract(response, 9, 4)
        self.assertEqual(client.total_tokens, {"input": 9, "output": 4})

    def test_anthropic_client_omits_deprecated_temperature(self):
        anthropic_mock = MagicMock()
        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            from compass.clients.anthropic import AnthropicClient

            client = AnthropicClient(model="claude-opus-4-8", api_key="fake")

        response_mock = MagicMock()
        response_mock.usage = MagicMock(input_tokens=9, output_tokens=4)
        response_mock.content = [MagicMock(type="text", text="answer")]
        client.client.messages.create.return_value = response_mock
        anthropic_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        anthropic_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=22,
                temperature=0.3,
                system="system prompt",
            )

        kwargs = client.client.messages.create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["system"], "system prompt")
        self.assertEqual(kwargs["max_tokens"], 22)
        self._assert_response_contract(response, 9, 4)

    def test_google_ai_client_conforms(self):
        google_mock = MagicMock()
        google_genai_types = MagicMock()
        google_genai_types.GenerateContentConfig.side_effect = lambda **kwargs: kwargs
        google_genai_types.SafetySetting.side_effect = lambda **kwargs: kwargs
        google_pkg = MagicMock()
        google_pkg.genai = google_mock
        google_mock.types = google_genai_types
        with patch.dict(
            "sys.modules",
            {
                "google": google_pkg,
                "google.genai": google_mock,
                "google.genai.types": google_genai_types,
            },
        ):
            from compass.clients.google_ai import GoogleAIClient

            client = GoogleAIClient(model="gemini-2.0-flash", api_key="fake")

        response_mock = MagicMock()
        response_mock.text = "answer"
        response_mock.usage_metadata = MagicMock(
            prompt_token_count=8,
            candidates_token_count=6,
        )
        client.client.models.generate_content.return_value = response_mock

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=40,
                temperature=0.1,
                system="system prompt",
            )

        kwargs = client.client.models.generate_content.call_args.kwargs
        self.assertIn("system prompt", kwargs["contents"])
        self.assertIn("prompt", kwargs["contents"])
        self.assertEqual(kwargs["config"]["max_output_tokens"], 40)
        self.assertEqual(kwargs["config"]["temperature"], 0.1)
        self._assert_response_contract(response, 8, 6)
        self.assertEqual(client.total_tokens, {"input": 8, "output": 6})

    def test_ollama_client_conforms(self):
        ollama_module = MagicMock()
        api_client = MagicMock()
        ollama_module.Client.return_value = api_client
        with patch.dict("sys.modules", {"ollama": ollama_module}):
            from compass.clients.ollama import OllamaClient

            client = OllamaClient(model="llama3.1")

        api_client.generate.return_value = {"response": "answer"}

        with patch("time.sleep"):
            response = client.complete(
                "prompt",
                max_tokens=18,
                temperature=0.7,
                system="system prompt",
            )

        kwargs = api_client.generate.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama3.1")
        self.assertEqual(kwargs["options"]["num_predict"], 18)
        self.assertEqual(kwargs["options"]["temperature"], 0.7)
        self.assertIn("<system>", kwargs["prompt"])
        self.assertIn("system prompt", kwargs["prompt"])
        self.assertIn("prompt", kwargs["prompt"])
        self.assertEqual(response.completion, "answer")
        self._assert_response_contract(
            response,
            response.tokens_used["input"],
            response.tokens_used["output"],
        )
        self.assertGreater(response.tokens_used["input"], 0)
        self.assertGreater(response.tokens_used["output"], 0)
        self.assertEqual(client.total_tokens, response.tokens_used)

    def test_complete_signature_matches_base_shape(self):
        from compass.clients.base import CompletionClient
        from compass.clients.anthropic import AnthropicClient
        from compass.clients.google_ai import GoogleAIClient
        from compass.clients.ollama import OllamaClient
        from compass.clients.openai import OpenAIClient
        from compass.clients.openai_responses import OpenAIResponsesClient

        expected = tuple(inspect.signature(CompletionClient.complete).parameters)
        for client_cls in (
            OpenAIClient,
            OpenAIResponsesClient,
            AnthropicClient,
            GoogleAIClient,
            OllamaClient,
        ):
            self.assertEqual(tuple(inspect.signature(client_cls.complete).parameters), expected)

    def test_unsupported_logprobs_are_explicit(self):
        anthropic_mock = MagicMock()
        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            from compass.clients.anthropic import AnthropicClient
            anthropic_client = AnthropicClient(model="claude-haiku-4-5", api_key="fake")

        google_mock = MagicMock()
        google_genai_types = MagicMock()
        google_genai_types.GenerateContentConfig.side_effect = lambda **kwargs: kwargs
        google_genai_types.SafetySetting.side_effect = lambda **kwargs: kwargs
        google_pkg = MagicMock()
        google_pkg.genai = google_mock
        google_mock.types = google_genai_types
        with patch.dict(
            "sys.modules",
            {
                "google": google_pkg,
                "google.genai": google_mock,
                "google.genai.types": google_genai_types,
            },
        ):
            from compass.clients.google_ai import GoogleAIClient
            google_client = GoogleAIClient(model="gemini-2.0-flash", api_key="fake")

        ollama_module = MagicMock()
        ollama_module.Client.return_value = MagicMock()
        with patch.dict("sys.modules", {"ollama": ollama_module}):
            from compass.clients.ollama import OllamaClient
            ollama_client = OllamaClient(model="llama3.1")

        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai_responses import OpenAIResponsesClient
            responses_client = OpenAIResponsesClient(model="gpt-5-mini", api_key="fake")

        for client in (
            anthropic_client,
            google_client,
            ollama_client,
            responses_client,
        ):
            with self.assertRaisesRegex(ValueError, "does not support logprobs"):
                client.complete("prompt", logprobs=True)


if __name__ == "__main__":
    unittest.main()
