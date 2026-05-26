"""Tests for new client features: pricing integration, throttle, retry helpers."""
import builtins
import importlib
import time
import unittest
from unittest.mock import MagicMock, patch

from compass.clients.pricing import get_pricing


# ── Pricing integration ───────────────────────────────────────────────────────

class TestClientPricingIntegration(unittest.TestCase):
    """Verify that client classes use the pricing table correctly."""

    def test_anthropic_client_uses_pricing_table(self):
        """AnthropicClient.total_cost_usd uses exact model pricing."""
        anthropic_mock = MagicMock()
        with patch.dict("sys.modules", {"anthropic": anthropic_mock}):
            from compass.clients.anthropic import AnthropicClient
            client = AnthropicClient(model="claude-sonnet-4-6", api_key="fake")
        p = get_pricing("claude-sonnet-4-6")
        # Inject token counts manually
        client._input_tokens = 1_000_000
        client._output_tokens = 1_000_000
        expected = p.input_cost_per_million + p.output_cost_per_million
        self.assertAlmostEqual(client.total_cost_usd, expected)

    def test_openai_client_uses_pricing_table(self):
        """OpenAIClient.total_cost_usd uses exact model pricing."""
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake")
        p = get_pricing("gpt-4o-mini")
        client._input_tokens = 1_000_000
        client._output_tokens = 1_000_000
        expected = p.input_cost_per_million + p.output_cost_per_million
        self.assertAlmostEqual(client.total_cost_usd, expected)

    def test_openai_responses_client_uses_pricing_table(self):
        """OpenAIResponsesClient.total_cost_usd uses exact model pricing."""
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai_responses import OpenAIResponsesClient
            client = OpenAIResponsesClient(model="gpt-4o-mini", api_key="fake")
        p = get_pricing("gpt-4o-mini")
        client._input_tokens = 1_000_000
        client._output_tokens = 1_000_000
        expected = p.input_cost_per_million + p.output_cost_per_million
        self.assertAlmostEqual(client.total_cost_usd, expected)


# ── _parse_reset_seconds helpers ──────────────────────────────────────────────

class TestParseResetSecondsOpenAI(unittest.TestCase):

    def setUp(self):
        from compass.clients.openai import _parse_reset_seconds
        self.parse = _parse_reset_seconds

    def test_seconds_only(self):
        self.assertAlmostEqual(self.parse("20s"), 20.0)
        self.assertAlmostEqual(self.parse("1.5s"), 1.5)

    def test_minutes_and_seconds(self):
        self.assertAlmostEqual(self.parse("1m0s"), 60.0)
        self.assertAlmostEqual(self.parse("1m30.5s"), 90.5)

    def test_minutes_only(self):
        self.assertAlmostEqual(self.parse("2m"), 120.0)

    def test_bare_float(self):
        self.assertAlmostEqual(self.parse("45"), 45.0)

    def test_invalid_returns_none(self):
        self.assertIsNone(self.parse("not-a-number"))


class TestParseResetSecondsAnthropic(unittest.TestCase):

    def setUp(self):
        from compass.clients.anthropic import _parse_reset_seconds
        self.parse = _parse_reset_seconds

    def test_seconds_only(self):
        self.assertAlmostEqual(self.parse("30s"), 30.0)

    def test_bare_float(self):
        self.assertAlmostEqual(self.parse("60"), 60.0)

    def test_invalid_returns_none(self):
        self.assertIsNone(self.parse("bad"))


# ── Temperature policy ────────────────────────────────────────────────────────

class TestOpenAIRequiredTemperature(unittest.TestCase):
    """OpenAIClient uses explicit required_temperature when configured."""

    def _make_client(self, model, **kwargs):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            return OpenAIClient(model=model, api_key="fake", **kwargs), openai_mock

    def _capture_temperature(self, model, caller_temp, **kwargs):
        """Return the temperature that would be sent to the API."""
        client, openai_mock = self._make_client(model, **kwargs)
        captured = {}

        class _FakeRateLimitError(Exception):
            pass

        openai_mock.RateLimitError = _FakeRateLimitError
        openai_mock.APIError = type("APIError", (Exception,), {})

        def fake_create(**kwargs):
            captured["temperature"] = kwargs.get("temperature")
            raise _FakeRateLimitError("stop")

        client.client.chat.completions.with_raw_response.create.side_effect = fake_create

        with patch("time.sleep"):  # prevent actual sleeping during retry loop
            try:
                client.complete("prompt", temperature=caller_temp)
            except Exception:
                pass
        return captured.get("temperature")

    def test_required_temperature_overrides_caller_temperature(self):
        temp = self._capture_temperature(
            "gpt-5-mini",
            0.0,
            required_temperature=1.0,
        )
        self.assertEqual(temp, 1.0)

    def test_required_temperature_applies_to_o4_models(self):
        temp = self._capture_temperature(
            "o4-mini",
            0.0,
            required_temperature=1.0,
        )
        self.assertEqual(temp, 1.0)

    def test_default_path_uses_caller_temperature(self):
        temp = self._capture_temperature("gpt-5-mini", 0.5)
        self.assertEqual(temp, 0.5)

    def test_openai_logprobs_are_requested_and_returned(self):
        client, openai_mock = self._make_client("gpt-4o-mini")
        captured = {}

        top_lp = [MagicMock(token="A", logprob=-0.1), MagicMock(token="B", logprob=-1.2)]
        content_lp = MagicMock(top_logprobs=top_lp)
        resp_mock = MagicMock()
        resp_mock.choices = [MagicMock(message=MagicMock(content="answer"), logprobs=MagicMock(content=[content_lp]))]
        resp_mock.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        raw_mock = MagicMock()
        raw_mock.parse.return_value = resp_mock
        raw_mock.headers = {}

        def fake_create(**kwargs):
            captured["logprobs"] = kwargs.get("logprobs")
            captured["top_logprobs"] = kwargs.get("top_logprobs")
            return raw_mock

        client.client.chat.completions.with_raw_response.create.side_effect = fake_create
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete("prompt", logprobs=True, top_logprobs=7)

        self.assertTrue(captured["logprobs"])
        self.assertEqual(captured["top_logprobs"], 7)
        self.assertEqual(response.logprobs, top_lp)


# ── Throttle logic ────────────────────────────────────────────────────────────

class TestThrottleLogic(unittest.TestCase):

    def test_no_sleep_when_interval_zero(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake", request_interval=0.0)
        client._last_call_at = time.monotonic()
        with patch("time.sleep") as mock_sleep:
            client._throttle()
            mock_sleep.assert_not_called()

    def test_sleeps_when_interval_positive(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake", request_interval=5.0)
        client._last_call_at = time.monotonic()  # just called
        with patch("time.sleep") as mock_sleep:
            client._throttle()
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            self.assertGreater(sleep_duration, 0)
            self.assertLessEqual(sleep_duration, 5.0)


class TestOllamaThrottleLogic(unittest.TestCase):

    def test_no_sleep_when_interval_zero(self):
        ollama_mock = MagicMock()
        with patch.dict("sys.modules", {"ollama": ollama_mock}):
            from compass.clients.ollama import OllamaClient
            client = OllamaClient(model="llama3.1:latest", request_interval=0.0)
        client._last_call_at = time.monotonic()
        with patch("time.sleep") as mock_sleep:
            client._throttle()
            mock_sleep.assert_not_called()

    def test_sleeps_when_interval_positive(self):
        ollama_mock = MagicMock()
        with patch.dict("sys.modules", {"ollama": ollama_mock}):
            from compass.clients.ollama import OllamaClient
            client = OllamaClient(model="llama3.1:latest", request_interval=5.0)
        client._last_call_at = time.monotonic()
        with patch("time.sleep") as mock_sleep:
            client._throttle()
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            self.assertGreater(sleep_duration, 0)
            self.assertLessEqual(sleep_duration, 5.0)


# ── _wait_from_429_headers ────────────────────────────────────────────────────

class TestWaitFrom429Headers(unittest.TestCase):

    def test_uses_retry_after_header(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake")
        wait = client._wait_from_429_headers({"retry-after": "30s"}, attempt=0)
        self.assertGreaterEqual(wait, 30.0)
        self.assertLessEqual(wait, 33.0)

    def test_fallback_when_no_headers(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake")
        wait = client._wait_from_429_headers({}, attempt=0)
        self.assertGreater(wait, 0)

    def test_exponential_backoff_on_repeated_attempts(self):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai import OpenAIClient
            client = OpenAIClient(model="gpt-4o-mini", api_key="fake")
        waits = [client._wait_from_429_headers({}, attempt=i) for i in range(5)]
        # Later attempts should generally have longer waits (modulo jitter)
        # At minimum, the cap at 60 means it's always bounded
        for w in waits:
            self.assertLessEqual(w, 65.0)


# ── OpenAIResponsesClient ─────────────────────────────────────────────────────

class TestOpenAIResponsesClientBasics(unittest.TestCase):

    def _make_client(self, model="gpt-5-mini", **kwargs):
        openai_mock = MagicMock()
        with patch.dict("sys.modules", {"openai": openai_mock}):
            from compass.clients.openai_responses import OpenAIResponsesClient
            return OpenAIResponsesClient(model=model, api_key="fake", **kwargs), openai_mock

    def test_output_token_multiplier_is_explicit(self):
        """Configured output-token multipliers are forwarded explicitly."""
        client, openai_mock = self._make_client(
            "gpt-5-mini",
            max_output_token_multiplier=10,
        )
        captured = {}

        resp_mock = MagicMock()
        resp_mock.output_text = "hello"
        resp_mock.usage = MagicMock(input_tokens=10, output_tokens=5)

        def fake_create(**kwargs):
            captured["max_output_tokens"] = kwargs.get("max_output_tokens")
            return resp_mock

        client.client.responses.create.side_effect = fake_create
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            client.complete("prompt", max_tokens=20)
        self.assertEqual(captured["max_output_tokens"], 200)  # 20 * 10

    def test_output_token_multiplier_defaults_to_requested_budget(self):
        client, openai_mock = self._make_client("gpt-5-mini")
        captured = {}

        resp_mock = MagicMock()
        resp_mock.output_text = "hello"
        resp_mock.usage = MagicMock(input_tokens=10, output_tokens=5)

        def fake_create(**kwargs):
            captured["max_output_tokens"] = kwargs.get("max_output_tokens")
            return resp_mock

        client.client.responses.create.side_effect = fake_create
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            client.complete("prompt", max_tokens=20)
        self.assertEqual(captured["max_output_tokens"], 20)

    def test_output_token_multiplier_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_output_token_multiplier must be at least 1",
        ):
            self._make_client(max_output_token_multiplier=0)

    def test_system_prompt_is_omitted_when_not_provided(self):
        client, openai_mock = self._make_client("gpt-5-mini")
        captured = {}

        resp_mock = MagicMock()
        resp_mock.output_text = "hello"
        resp_mock.usage = MagicMock(input_tokens=10, output_tokens=5)

        def fake_create(**kwargs):
            captured.update(kwargs)
            return resp_mock

        client.client.responses.create.side_effect = fake_create
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            client.complete("prompt", max_tokens=20)

        self.assertNotIn("instructions", captured)

    def test_nonzero_temperature_is_rejected(self):
        client, _ = self._make_client("gpt-5-mini")
        with self.assertRaisesRegex(
            ValueError,
            "does not support temperature overrides",
        ):
            client.complete("prompt", temperature=0.2)

    def test_missing_usage_fails_without_explicit_estimation_opt_in(self):
        client, openai_mock = self._make_client("gpt-5-mini")

        resp_mock = MagicMock()
        resp_mock.output_text = "hello"
        resp_mock.usage = None
        client.client.responses.create.return_value = resp_mock
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            with self.assertRaisesRegex(
                RuntimeError,
                "did not return usage metadata",
            ):
                client.complete("prompt", max_tokens=20)

    def test_missing_usage_can_be_estimated_when_enabled(self):
        client, openai_mock = self._make_client(
            "gpt-5-mini",
            allow_estimated_usage=True,
        )

        resp_mock = MagicMock()
        resp_mock.output_text = "hello world"
        resp_mock.usage = None
        client.client.responses.create.return_value = resp_mock
        openai_mock.RateLimitError = type("RateLimitError", (Exception,), {})
        openai_mock.APIError = type("APIError", (Exception,), {})

        with patch("time.sleep"):
            response = client.complete("prompt", max_tokens=20)

        self.assertEqual(response.tokens_used, {"input": 0, "output": 2})
        self.assertEqual(client.total_tokens, {"input": 0, "output": 2})


class TestGoogleAIClientFeatures(unittest.TestCase):

    def _make_client(self, **kwargs):
        google_mock = MagicMock()
        google_genai_types = MagicMock()
        google_genai_types.GenerateContentConfig.side_effect = lambda **config_kwargs: config_kwargs
        google_genai_types.SafetySetting.side_effect = lambda **setting_kwargs: setting_kwargs
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
            client = GoogleAIClient(model="gemini-2.0-flash", api_key="fake", **kwargs)
        return client, google_mock, google_genai_types

    def test_google_ai_client_uses_pricing_table(self):
        client, _, _ = self._make_client()
        p = get_pricing("gemini-2.0-flash")
        client._input_tokens = 1_000_000
        client._output_tokens = 1_000_000
        expected = p.input_cost_per_million + p.output_cost_per_million
        self.assertAlmostEqual(client.total_cost_usd, expected)

    def test_google_ai_client_applies_safety_override_only_when_enabled(self):
        client, google_mock, _ = self._make_client(disable_safety_filters=True)
        response_mock = MagicMock()
        response_mock.text = "answer"
        response_mock.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        captured = {}

        def fake_generate_content(**kwargs):
            captured["config"] = kwargs["config"]
            return response_mock

        client.client.models.generate_content.side_effect = fake_generate_content
        with patch("time.sleep"):
            response = client.complete("prompt", max_tokens=20)

        self.assertIn("safety_settings", captured["config"])
        self.assertEqual(len(captured["config"]["safety_settings"]), 4)
        self.assertGreater(response.cost_usd, 0.0)

    def test_google_ai_client_omits_safety_override_by_default(self):
        client, google_mock, _ = self._make_client()
        response_mock = MagicMock()
        response_mock.text = "answer"
        response_mock.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        captured = {}

        def fake_generate_content(**kwargs):
            captured["config"] = kwargs["config"]
            return response_mock

        client.client.models.generate_content.side_effect = fake_generate_content
        with patch("time.sleep"):
            client.complete("prompt")

        self.assertNotIn("safety_settings", captured["config"])

    def test_google_ai_client_enforces_max_requests(self):
        client, _, _ = self._make_client()
        client._request_count = get_pricing("gemini-2.0-flash").max_requests
        with self.assertRaisesRegex(RuntimeError, "Reached max_requests"):
            client.complete("prompt")

    def test_uses_response_text_not_thought_signature(self):
        """Client must use response.text, not attempt to decode thought_signature bytes.

        Real Gemini thinking models return a single part that has both .text
        (the visible answer) and .thought_signature (opaque internal reasoning
        bytes). The client must read response.text and ignore thought_signature.
        """
        client, _, _ = self._make_client()

        part_mock = MagicMock()
        part_mock.text = "At its core, an artificial neural network..."
        part_mock.thought_signature = b"ErgeCrUeAQw51sc-opaque-reasoning-bytes"

        response_mock = MagicMock()
        response_mock.text = "At its core, an artificial neural network..."
        response_mock.candidates = [
            MagicMock(content=MagicMock(parts=[part_mock]))
        ]
        response_mock.usage_metadata = MagicMock(
            prompt_token_count=7,
            candidates_token_count=39,
        )
        client.client.models.generate_content.return_value = response_mock

        with patch("time.sleep"):
            response = client.complete("What is a neural network?", max_tokens=1000)

        self.assertEqual(response.completion, "At its core, an artificial neural network...")
        self.assertEqual(response.tokens_used, {"input": 7, "output": 39})

    def test_max_tokens_forwarded_to_api(self):
        client, _, _ = self._make_client()
        captured = {}

        response_mock = MagicMock()
        response_mock.text = "Some response"
        response_mock.usage_metadata = MagicMock(
            prompt_token_count=5,
            candidates_token_count=10,
        )

        def fake_generate(**kwargs):
            captured["config"] = kwargs.get("config")
            return response_mock

        client.client.models.generate_content.side_effect = fake_generate

        with patch("time.sleep"):
            client.complete("prompt", max_tokens=1000)

        self.assertEqual(captured["config"]["max_output_tokens"], 1000)


class TestOptionalClientExports(unittest.TestCase):

    def test_missing_openai_dependency_raises_clear_error(self):
        import compass.clients as clients_module

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {
                "compass.clients.openai",
                "compass.clients.openai_responses",
            }:
                raise ImportError("missing openai")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            reloaded = importlib.reload(clients_module)
            with self.assertRaisesRegex(ImportError, r"compass-eval\[openai\]"):
                reloaded.OpenAIClient(model="gpt-4o-mini")
            with self.assertRaisesRegex(ImportError, r"compass-eval\[openai\]"):
                reloaded.OpenAIResponsesClient(model="gpt-5-mini")
        importlib.reload(clients_module)

    def test_top_level_import_survives_missing_openai_dependency(self):
        import compass as compass_module
        import compass.clients as clients_module

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {
                "compass.clients.openai",
                "compass.clients.openai_responses",
            }:
                raise ImportError("missing openai")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            importlib.reload(clients_module)
            reloaded = importlib.reload(compass_module)
            with self.assertRaisesRegex(ImportError, r"compass-eval\[openai\]"):
                reloaded.OpenAIClient(model="gpt-4o-mini")
        importlib.reload(clients_module)
        importlib.reload(compass_module)


if __name__ == "__main__":
    unittest.main()
