"""Tests for the shared pricing table."""
import unittest

from compass.clients.pricing import (
    ModelPricing,
    PRICING_TABLE,
    get_pricing,
    _provider,
)


class TestPricingTable(unittest.TestCase):

    def test_table_has_expected_models(self):
        expected = [
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.1-nano", "gpt-4.1-mini",
            "o4-mini",
            "claude-haiku-4-5-20251001", "claude-haiku-4-5",
            "claude-sonnet-4-6", "claude-opus-4-7",
            "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro",
            "gemini-1.5-flash", "gemini-1.5-flash-8b",
        ]
        for model in expected:
            self.assertIn(model, PRICING_TABLE, f"Missing: {model}")

    def test_pricing_fields_are_positive(self):
        for name, pricing in PRICING_TABLE.items():
            self.assertGreater(pricing.input_cost_per_million, 0, name)
            self.assertGreater(pricing.output_cost_per_million, 0, name)

    def test_gemini_models_have_max_requests(self):
        for name, pricing in PRICING_TABLE.items():
            if name.startswith("gemini-"):
                self.assertIsNotNone(pricing.max_requests, name)
                self.assertGreater(pricing.max_requests, 0, name)

    def test_non_gemini_models_have_no_max_requests(self):
        for name, pricing in PRICING_TABLE.items():
            if not name.startswith("gemini-"):
                self.assertIsNone(pricing.max_requests, name)

    def test_model_pricing_is_frozen(self):
        pricing = PRICING_TABLE["gpt-4o-mini"]
        with self.assertRaises((AttributeError, TypeError)):
            pricing.input_cost_per_million = 999.0


class TestGetPricing(unittest.TestCase):

    def test_exact_match(self):
        p = get_pricing("gpt-4o-mini")
        self.assertEqual(p.name, "gpt-4o-mini")
        self.assertAlmostEqual(p.input_cost_per_million, 0.15)
        self.assertAlmostEqual(p.output_cost_per_million, 0.60)

    def test_claude_exact_match(self):
        p = get_pricing("claude-sonnet-4-6")
        self.assertAlmostEqual(p.input_cost_per_million, 3.00)
        self.assertAlmostEqual(p.output_cost_per_million, 15.00)

    def test_unknown_openai_model_falls_back_to_gpt4o_mini(self):
        p = get_pricing("gpt-unknown-future-model")
        self.assertEqual(p, PRICING_TABLE["gpt-4o-mini"])

    def test_unknown_anthropic_model_falls_back(self):
        p = get_pricing("claude-unknown-future")
        self.assertEqual(p, PRICING_TABLE["claude-haiku-4-5-20251001"])

    def test_unknown_google_model_falls_back(self):
        p = get_pricing("gemini-unknown-future")
        self.assertEqual(p, PRICING_TABLE["gemini-2.0-flash"])

    def test_returns_model_pricing_instance(self):
        p = get_pricing("gpt-4o")
        self.assertIsInstance(p, ModelPricing)


class TestProviderDetection(unittest.TestCase):

    def test_anthropic_prefix(self):
        self.assertEqual(_provider("claude-sonnet-4-6"), "anthropic")
        self.assertEqual(_provider("claude-haiku-4-5"), "anthropic")

    def test_google_prefix(self):
        self.assertEqual(_provider("gemini-2.0-flash"), "google")
        self.assertEqual(_provider("gemini-1.5-pro"), "google")

    def test_openai_default(self):
        self.assertEqual(_provider("gpt-4o-mini"), "openai")
        self.assertEqual(_provider("o4-mini"), "openai")
        self.assertEqual(_provider("gpt-5-mini"), "openai")


if __name__ == "__main__":
    unittest.main()
