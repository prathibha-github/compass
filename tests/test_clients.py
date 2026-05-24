"""Tests for client abstraction and implementations."""
import unittest

from compass.clients import CompletionClient, CompletionResponse


class MockClient(CompletionClient):
    """Mock implementation for testing."""

    def __init__(self, response: str = "test", tokens: dict = None, cost: float = 0.0):
        self.response = response
        self.tokens = tokens or {"input": 100, "output": 50}
        self.cost = cost
        self.calls = []

    def complete(
        self,
        prompt,
        max_tokens=180,
        temperature=0.0,
        system=None,
        logprobs=False,
        top_logprobs=0,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
            }
        )
        return CompletionResponse(
            completion=self.response,
            tokens_used=self.tokens,
            cost_usd=self.cost,
        )


class TestCompletionResponse(unittest.TestCase):
    """Test CompletionResponse dataclass."""

    def test_response_creation_minimal(self):
        """Create response with minimal fields."""
        response = CompletionResponse(completion="test")
        self.assertEqual(response.completion, "test")
        self.assertIsNone(response.tokens_used)
        self.assertEqual(response.cost_usd, 0.0)

    def test_response_creation_full(self):
        """Create response with all fields."""
        response = CompletionResponse(
            completion="test",
            tokens_used={"input": 100, "output": 50},
            cost_usd=0.001,
        )
        self.assertEqual(response.completion, "test")
        self.assertEqual(response.tokens_used["input"], 100)
        self.assertEqual(response.cost_usd, 0.001)

    def test_response_tokens_dict(self):
        """Response tokens are properly stored."""
        tokens = {"input": 200, "output": 100}
        response = CompletionResponse(
            completion="test",
            tokens_used=tokens,
        )
        self.assertEqual(response.tokens_used["input"], 200)
        self.assertEqual(response.tokens_used["output"], 100)

    def test_response_cost_calculation(self):
        """Response cost is accurate."""
        response = CompletionResponse(
            completion="test",
            tokens_used={"input": 1000, "output": 500},
            cost_usd=0.002,
        )
        self.assertEqual(response.cost_usd, 0.002)

    def test_response_empty_completion(self):
        """Response can have empty completion."""
        response = CompletionResponse(completion="")
        self.assertEqual(response.completion, "")


class TestCompletionClient(unittest.TestCase):
    """Test CompletionClient interface."""

    def test_client_is_abstract(self):
        """CompletionClient cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            CompletionClient()

    def test_client_subclass_must_implement_complete(self):
        """Subclass must implement complete method."""

        class IncompleteClient(CompletionClient):
            pass

        with self.assertRaises(TypeError):
            IncompleteClient()

    def test_mock_client_implements_interface(self):
        """Mock client properly implements interface."""
        client = MockClient()
        self.assertTrue(hasattr(client, "complete"))
        self.assertTrue(callable(client.complete))

    def test_mock_client_response_format(self):
        """Mock client returns CompletionResponse."""
        client = MockClient("test response")
        response = client.complete("test prompt")

        self.assertIsInstance(response, CompletionResponse)
        self.assertEqual(response.completion, "test response")

    def test_mock_client_tracks_calls(self):
        """Mock client tracks completion calls."""
        client = MockClient()

        client.complete("prompt 1", max_tokens=100)
        client.complete("prompt 2", max_tokens=200, temperature=0.5)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["prompt"], "prompt 1")
        self.assertEqual(client.calls[0]["max_tokens"], 100)
        self.assertEqual(client.calls[1]["temperature"], 0.5)

    def test_mock_client_system_parameter(self):
        """Mock client accepts system parameter."""
        client = MockClient()
        client.complete("prompt", system="You are helpful")

        self.assertEqual(client.calls[0]["system"], "You are helpful")

    def test_mock_client_default_parameters(self):
        """Mock client applies default parameters."""
        client = MockClient()
        client.complete("prompt")

        call = client.calls[0]
        self.assertEqual(call["max_tokens"], 180)
        self.assertEqual(call["temperature"], 0.0)
        self.assertIsNone(call["system"])

    def test_mock_client_custom_tokens(self):
        """Mock client can be configured with custom tokens."""
        tokens = {"input": 500, "output": 250}
        client = MockClient(tokens=tokens)
        response = client.complete("prompt")

        self.assertEqual(response.tokens_used["input"], 500)
        self.assertEqual(response.tokens_used["output"], 250)

    def test_mock_client_custom_cost(self):
        """Mock client can be configured with custom cost."""
        client = MockClient(cost=0.005)
        response = client.complete("prompt")

        self.assertEqual(response.cost_usd, 0.005)

    def test_multiple_mock_clients_independent(self):
        """Multiple mock client instances are independent."""
        client1 = MockClient("response 1")
        client2 = MockClient("response 2")

        response1 = client1.complete("prompt 1")
        response2 = client2.complete("prompt 2")

        self.assertEqual(response1.completion, "response 1")
        self.assertEqual(response2.completion, "response 2")
        self.assertEqual(len(client1.calls), 1)
        self.assertEqual(len(client2.calls), 1)


class TestClientPromptHandling(unittest.TestCase):
    """Test how clients handle different prompt types."""

    def test_empty_prompt(self):
        """Client handles empty prompt."""
        client = MockClient()
        response = client.complete("")
        self.assertIsNotNone(response)

    def test_very_long_prompt(self):
        """Client handles very long prompt."""
        client = MockClient()
        long_prompt = "prompt " * 10000
        response = client.complete(long_prompt)
        self.assertIsNotNone(response)

    def test_prompt_with_special_characters(self):
        """Client handles special characters in prompt."""
        client = MockClient()
        special_prompt = 'Test with "quotes", \\backslashes, \nnewlines'
        response = client.complete(special_prompt)
        self.assertIsNotNone(response)

    def test_prompt_with_unicode(self):
        """Client handles unicode in prompt."""
        client = MockClient()
        unicode_prompt = "Hello 世界 🌍 Привет مرحبا"
        response = client.complete(unicode_prompt)
        self.assertIsNotNone(response)


class TestClientResponseParsing(unittest.TestCase):
    """Test response handling in clients."""

    def test_response_with_json(self):
        """Client returns response with JSON."""
        json_response = '{"score": 0.5, "hit": false}'
        client = MockClient(json_response)
        response = client.complete("prompt")
        self.assertEqual(response.completion, json_response)

    def test_response_with_markdown(self):
        """Client returns response with markdown."""
        md_response = "# Title\nSome content\n- list item"
        client = MockClient(md_response)
        response = client.complete("prompt")
        self.assertEqual(response.completion, md_response)

    def test_response_with_code(self):
        """Client returns response with code."""
        code_response = "def hello():\n    return 'world'"
        client = MockClient(code_response)
        response = client.complete("prompt")
        self.assertEqual(response.completion, code_response)


if __name__ == "__main__":
    unittest.main()
