"""Tests for the client policy audit inventory."""

import unittest

from compass.clients import list_client_policy_translations


class ClientPolicyAuditTests(unittest.TestCase):
    def test_inventory_covers_all_builtin_adapters(self):
        adapters = {entry.adapter for entry in list_client_policy_translations()}
        self.assertEqual(
            adapters,
            {
                "AnthropicClient",
                "GoogleAIClient",
                "OllamaClient",
                "OpenAIClient",
                "OpenAIResponsesClient",
            },
        )

    def test_inventory_includes_known_policy_cases(self):
        entries = {
            (entry.adapter, entry.category): entry
            for entry in list_client_policy_translations()
        }
        self.assertIn(("OpenAIClient", "required_temperature"), entries)
        self.assertIn(("OpenAIResponsesClient", "output_token_multiplier"), entries)
        self.assertIn(("OpenAIResponsesClient", "unsupported_temperature"), entries)
        self.assertIn(("GoogleAIClient", "max_request_cap"), entries)
        self.assertIn(("OllamaClient", "token_estimation"), entries)

    def test_explicit_entries_are_marked_explicit(self):
        entries = {
            (entry.adapter, entry.category): entry
            for entry in list_client_policy_translations()
        }
        self.assertTrue(
            entries[("OpenAIClient", "required_temperature")].explicit_to_caller
        )
        self.assertTrue(
            entries[("OpenAIResponsesClient", "output_token_multiplier")].explicit_to_caller
        )
        self.assertTrue(
            entries[("OpenAIResponsesClient", "unsupported_temperature")].explicit_to_caller
        )

    def test_inventory_entries_are_unique_and_named(self):
        seen = set()
        for entry in list_client_policy_translations():
            key = (entry.adapter, entry.category, entry.trigger)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertTrue(entry.behavior)
            self.assertTrue(entry.category)
            self.assertTrue(entry.trigger)


if __name__ == "__main__":
    unittest.main()
