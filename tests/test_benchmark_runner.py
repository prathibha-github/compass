"""Tests for benchmark runner client routing helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from compass.benchmark import runner


class BenchmarkRunnerClientRoutingTests(unittest.TestCase):
    def test_create_client_routes_models_to_expected_client_types(self):
        cases = (
            ("gemini-2.0-flash", "GoogleAIClient"),
            ("gpt-4o-mini", "OpenAIClient"),
            ("claude-haiku-4-5", "AnthropicClient"),
            ("llama3.1", "OllamaClient"),
        )

        for model, client_name in cases:
            sentinel = object()
            with self.subTest(model=model):
                with patch.object(runner, client_name, return_value=sentinel) as client_ctor:
                    self.assertIs(runner._create_client(model), sentinel)
                client_ctor.assert_called_once_with(model=model)

    def test_model_connection_uses_client_complete(self):
        fake_client = SimpleNamespace(
            complete=MagicMock(
                return_value=SimpleNamespace(tokens_used={"input": 1, "output": 2})
            )
        )
        with patch.object(runner, "_create_client", return_value=fake_client) as create_client:
            self.assertTrue(runner.test_model_connection("gpt-4o-mini"))

        create_client.assert_called_once_with("gpt-4o-mini")
        fake_client.complete.assert_called_once_with("test", max_tokens=10)

    def test_model_connection_returns_false_when_client_raises(self):
        fake_client = SimpleNamespace(
            complete=MagicMock(side_effect=RuntimeError("unavailable"))
        )
        with patch.object(runner, "_create_client", return_value=fake_client):
            self.assertFalse(runner.test_model_connection("llama3.1"))


if __name__ == "__main__":
    unittest.main()
