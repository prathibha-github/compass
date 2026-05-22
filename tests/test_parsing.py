"""Tests for JSON parsing with fallbacks."""
import unittest

from compass.judges.parsing import parse_judge_response


class TestParseJudgeResponse(unittest.TestCase):
    """Test JSON parsing for judge responses."""

    def test_parse_well_formed_json(self):
        raw = '{"score": 0.75, "hit": true, "confidence": 0.95, "rationale": "test"}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.75)
        self.assertTrue(result["hit"])

    def test_parse_malformed_json_with_fallback(self):
        raw = 'The response is: {"score": 0.5, "hit": false, "confidence": 0.8, "rationale": "mixed"}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["hit"])

    def test_parse_completely_invalid_json(self):
        raw = "No JSON here at all, just text"
        result = parse_judge_response(raw)
        self.assertIsNone(result)

    def test_parse_json_at_start(self):
        raw = '{"score": 0.3, "hit": false} some trailing text'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.3)

    def test_parse_json_at_end(self):
        raw = 'some leading text {"score": 0.7, "hit": true}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.7)

    def test_parse_nested_json(self):
        raw = '{"score": 0.6, "hit": true, "metadata": {"nested": "value"}}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["nested"], "value")

    def test_parse_missing_closing_brace(self):
        raw = '{"score": 0.5, "hit": false'
        result = parse_judge_response(raw)
        self.assertIsNone(result)

    def test_parse_empty_object(self):
        raw = '{}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result, {})

    def test_parse_with_newlines(self):
        raw = '''
        {
            "score": 0.8,
            "hit": true,
            "confidence": 0.9,
            "rationale": "good"
        }
        '''
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.8)

    def test_parse_with_escaped_quotes_in_rationale(self):
        raw = '{"score": 0.5, "hit": false, "rationale": "They said \\"hello\\""}'
        result = parse_judge_response(raw)
        self.assertIsNotNone(result)
        self.assertIn("hello", result["rationale"])

    def test_parse_multiple_json_objects_ambiguous(self):
        # Multiple objects is ambiguous—first { to last } spans both
        # This shouldn't happen in practice, but if it does, we return None
        raw = '{"score": 0.3, "hit": false} and also {"score": 0.9, "hit": true}'
        result = parse_judge_response(raw)
        # Can't parse this cleanly, so returns None
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
