"""Tests for benchmark runner client routing helpers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from compass.benchmark import runner


class BenchmarkRunnerClientRoutingTests(unittest.TestCase):
    def test_create_client_routes_models_to_expected_client_types(self):
        cases = (
            ("gemini-2.0-flash", "GoogleAIClient"),
            ("gpt-4o-mini", "OpenAIClient"),
            ("gpt-5-mini", "OpenAIClient"),
            ("o4-mini", "OpenAIClient"),
            ("claude-haiku-4-5", "AnthropicClient"),
            ("llama3.1", "OllamaClient"),
        )

        for model, client_name in cases:
            sentinel = object()
            with self.subTest(model=model):
                with patch.object(runner, client_name, return_value=sentinel) as client_ctor:
                    self.assertIs(runner._create_client(model), sentinel)
                if client_name == "OpenAIClient" and model in {"gpt-5-mini", "o4-mini"}:
                    client_ctor.assert_called_once_with(
                        model=model,
                        required_temperature=1.0,
                    )
                elif client_name == "OpenAIClient":
                    client_ctor.assert_called_once_with(
                        model=model,
                        required_temperature=None,
                    )
                else:
                    client_ctor.assert_called_once_with(model=model)

    def test_model_connection_for_openai_uses_lightweight_env_check(self):
        fake_client = SimpleNamespace()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}, clear=True):
            with patch.object(runner, "_create_client", return_value=fake_client) as create_client:
                self.assertTrue(runner.test_model_connection("gpt-4o-mini"))

        create_client.assert_called_once_with("gpt-4o-mini")

    def test_model_connection_for_o4_uses_lightweight_env_check(self):
        fake_client = SimpleNamespace()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}, clear=True):
            with patch.object(runner, "_create_client", return_value=fake_client) as create_client:
                self.assertTrue(runner.test_model_connection("o4-mini"))

        create_client.assert_called_once_with("o4-mini")

    def test_model_connection_for_ollama_uses_model_listing(self):
        fake_client = SimpleNamespace(
            api_client=SimpleNamespace(
                list=MagicMock(
                    return_value={"models": [{"name": "llama3.1:latest"}]}
                )
            )
        )
        with patch.object(runner, "_create_client", return_value=fake_client) as create_client:
            self.assertTrue(runner.test_model_connection("llama3.1"))

        create_client.assert_called_once_with("llama3.1")
        fake_client.api_client.list.assert_called_once_with()

    def test_model_connection_returns_false_when_probe_raises(self):
        fake_client = SimpleNamespace(
            api_client=SimpleNamespace(
                list=MagicMock(side_effect=RuntimeError("unavailable"))
            )
        )
        with patch.object(runner, "_create_client", return_value=fake_client):
            self.assertFalse(runner.test_model_connection("llama3.1"))

    def test_model_connection_returns_false_when_required_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(runner, "_create_client", return_value=SimpleNamespace()):
                self.assertFalse(runner.test_model_connection("claude-haiku-4-5"))


if __name__ == "__main__":
    unittest.main()
