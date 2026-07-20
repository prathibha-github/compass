"""Tests for the compass.detectors module."""
import unittest
from unittest.mock import MagicMock

from compass.detectors.base import (
    DetectorResult,
    StyleCondition,
    StylePrompt,
    TicSuite,
    analyze_text,
    compute_uplift,
    suite_uses_llm_judge,
    summarize_outputs,
    summarize_detector_records,
)
from compass.detectors.heuristic import (
    CharacterCountDetector,
    EmojiDetector,
    PhraseSetDetector,
    RegexDetector,
)
from compass.detectors.llm_detector import LLMJudgeDetector
from compass.detectors.quirk_detection import find_quirks, QUIRK_PROMPTS
from compass.detectors.persona_transfer import analyze_persona_transfer


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_suite(detectors, name="test_suite"):
    return TicSuite(
        name=name,
        prompts=[StylePrompt(id="p1", text="Write something.")],
        conditions=[StyleCondition(name="neutral", system_prompt="Be helpful.")],
        detectors=detectors,
        baseline_condition="neutral",
    )


# ── DetectorResult ────────────────────────────────────────────────────────────

class TestDetectorResult(unittest.TestCase):

    def test_to_dict_fields(self):
        r = DetectorResult(name="foo", count=2, score=2.0, hit=True, metadata={"k": "v"})
        d = r.to_dict()
        self.assertEqual(d["name"], "foo")
        self.assertEqual(d["count"], 2)
        self.assertEqual(d["score"], 2.0)
        self.assertTrue(d["hit"])
        self.assertEqual(d["metadata"]["k"], "v")

    def test_is_frozen(self):
        r = DetectorResult(name="x", count=0, score=0.0, hit=False)
        with self.assertRaises((AttributeError, TypeError)):
            r.count = 99


# ── RegexDetector ─────────────────────────────────────────────────────────────

class TestRegexDetector(unittest.TestCase):

    def test_match_count(self):
        d = RegexDetector(name="delve", pattern=r"\bdelve\b")
        r = d.detect("I will delve into the delve of delve.")
        self.assertEqual(r.count, 3)
        self.assertTrue(r.hit)

    def test_no_match(self):
        d = RegexDetector(name="goblin", pattern=r"\bgoblin\b")
        r = d.detect("No monsters here.")
        self.assertEqual(r.count, 0)
        self.assertFalse(r.hit)

    def test_hit_threshold(self):
        d = RegexDetector(name="em_dash", pattern=r"—", hit_threshold=2)
        r = d.detect("One dash — not enough.")
        self.assertFalse(r.hit)
        r2 = d.detect("Two — dashes — here.")
        self.assertTrue(r2.hit)

    def test_metadata_matches_capped_at_20(self):
        d = RegexDetector(name="x", pattern=r"\bword\b")
        text = " ".join(["word"] * 30)
        r = d.detect(text)
        self.assertLessEqual(len(r.metadata["matches"]), 20)


# ── PhraseSetDetector ─────────────────────────────────────────────────────────

class TestPhraseSetDetector(unittest.TestCase):

    def test_phrase_hit(self):
        d = PhraseSetDetector(name="tone", phrases={"great job", "amazing work"})
        r = d.detect("Great job on that amazing work!")
        self.assertEqual(r.count, 2)
        self.assertTrue(r.hit)

    def test_case_insensitive(self):
        d = PhraseSetDetector(name="apology", phrases={"sorry"})
        r = d.detect("SORRY about that.")
        self.assertEqual(r.count, 1)

    def test_whole_phrase_boundary(self):
        d = PhraseSetDetector(name="valid", phrases={"valid"})
        r = d.detect("invalidated")
        self.assertEqual(r.count, 0)

    def test_no_match(self):
        d = PhraseSetDetector(name="jargon", phrases={"leverage", "synergy"})
        r = d.detect("Just plain text.")
        self.assertFalse(r.hit)
        self.assertEqual(r.metadata["phrases"], {})


# ── EmojiDetector ─────────────────────────────────────────────────────────────

class TestEmojiDetector(unittest.TestCase):

    def test_detects_emoji(self):
        d = EmojiDetector()
        r = d.detect("Hello 😀 world 🎉!")
        self.assertEqual(r.count, 2)
        self.assertTrue(r.hit)

    def test_no_emoji(self):
        d = EmojiDetector()
        r = d.detect("Just text.")
        self.assertEqual(r.count, 0)
        self.assertFalse(r.hit)

    def test_excessive_emoji_threshold(self):
        d = EmojiDetector(name="excessive_emoji", hit_threshold=3)
        r = d.detect("😀😀 two emojis")
        self.assertFalse(r.hit)
        r2 = d.detect("😀😀😀 three emojis")
        self.assertTrue(r2.hit)


# ── CharacterCountDetector ────────────────────────────────────────────────────

class TestCharacterCountDetector(unittest.TestCase):

    def test_counts_character(self):
        d = CharacterCountDetector(name="em_dash", character="—")
        r = d.detect("An em dash — and another — here.")
        self.assertEqual(r.count, 2)

    def test_threshold(self):
        d = CharacterCountDetector(name="semicolon", character=";", hit_threshold=2)
        r = d.detect("One semicolon; not enough.")
        self.assertFalse(r.hit)

    def test_metadata_character(self):
        d = CharacterCountDetector(name="sc", character=";")
        r = d.detect("a;b;c")
        self.assertEqual(r.metadata["character"], ";")


# ── LLMJudgeDetector ──────────────────────────────────────────────────────────

class TestLLMJudgeDetector(unittest.TestCase):

    def test_detect_raises_without_judge(self):
        d = LLMJudgeDetector(name="tone", rubric="Score tone.")
        with self.assertRaises(RuntimeError):
            d.detect("some text")

    def test_build_prompt_contains_rubric(self):
        d = LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy here.")
        prompt = d.build_prompt("test response")
        self.assertIn("Score sycophancy here.", prompt)
        self.assertIn("test response", prompt)

    def test_parse_response_valid_json(self):
        d = LLMJudgeDetector(name="tone", rubric="rubric")
        raw = '{"score": 0.8, "hit": true, "confidence": 0.9, "rationale": "clear tone"}'
        result = d.parse_response(raw)
        self.assertAlmostEqual(result.score, 0.8)
        self.assertTrue(result.hit)
        self.assertEqual(result.count, 1)

    def test_parse_response_embedded_json(self):
        d = LLMJudgeDetector(name="tone", rubric="rubric")
        raw = 'Here is my evaluation: {"score": 0.3, "hit": false, "confidence": 0.7, "rationale": "ok"}'
        result = d.parse_response(raw)
        self.assertAlmostEqual(result.score, 0.3)
        self.assertFalse(result.hit)

    def test_parse_response_invalid_json(self):
        d = LLMJudgeDetector(name="tone", rubric="rubric")
        raw = "This is not JSON at all."
        result = d.parse_response(raw)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.hit)
        self.assertTrue(result.metadata.get("parse_error"))

    def test_parse_response_score_clamped_to_01(self):
        d = LLMJudgeDetector(name="tone", rubric="rubric")
        raw = '{"score": 5.0, "hit": true}'
        result = d.parse_response(raw)
        self.assertEqual(result.score, 1.0)

    def test_parse_response_hit_threshold(self):
        d = LLMJudgeDetector(name="tone", rubric="rubric", hit_threshold=0.5)
        raw = '{"score": 0.6}'
        result = d.parse_response(raw)
        self.assertTrue(result.hit)

    def test_detect_with_judge(self):
        d = LLMJudgeDetector(name="tone", rubric="Score tone.")
        judge_fn = MagicMock(return_value='{"score": 0.7, "hit": true, "confidence": 0.8, "rationale": "ok"}')
        result = d.detect_with_judge("some text", judge_fn)
        self.assertAlmostEqual(result.score, 0.7)
        self.assertTrue(result.hit)
        judge_fn.assert_called_once()


# ── TicSuite ──────────────────────────────────────────────────────────────────

class TestTicSuite(unittest.TestCase):

    def test_checkpoint_stem_defaults_to_name(self):
        suite = _make_suite([RegexDetector(name="d", pattern=r"x")])
        self.assertEqual(suite.checkpoint_stem, "test_suite")

    def test_checkpoint_name_overrides(self):
        suite = TicSuite(
            name="test",
            checkpoint_name="custom_stem",
            prompts=[StylePrompt(id="p", text="t")],
            conditions=[StyleCondition(name="c", system_prompt="s")],
            detectors=[],
            baseline_condition="c",
        )
        self.assertEqual(suite.checkpoint_stem, "custom_stem")

    def test_extra_record_fields_with_no_callback(self):
        suite = _make_suite([])
        result = suite.extra_record_fields("text", {})
        self.assertEqual(result, {})


# ── analyze_text ──────────────────────────────────────────────────────────────

class TestAnalyzeText(unittest.TestCase):

    def test_heuristic_detector(self):
        suite = _make_suite([RegexDetector(name="delve", pattern=r"\bdelve\b")])
        results = analyze_text("I will delve deeply.", suite)
        self.assertIn("delve", results)
        self.assertEqual(results["delve"].count, 1)

    def test_multiple_detectors(self):
        suite = _make_suite([
            RegexDetector(name="delve", pattern=r"\bdelve\b"),
            CharacterCountDetector(name="em_dash", character="—"),
        ])
        results = analyze_text("Delve — into this.", suite)
        self.assertIn("delve", results)
        self.assertIn("em_dash", results)

    def test_llm_judge_raises_without_judge_fn(self):
        suite = _make_suite([LLMJudgeDetector(name="tone", rubric="rubric")])
        with self.assertRaises(ValueError, msg="Detector 'tone' requires a judge function"):
            analyze_text("some text", suite)

    def test_llm_judge_with_judge_fn(self):
        suite = _make_suite([LLMJudgeDetector(name="tone", rubric="rubric")])
        judge_fn = lambda prompt, max_tokens: '{"score": 0.5, "hit": false}'
        results = analyze_text("some text", suite, judge_fn=judge_fn)
        self.assertIn("tone", results)


# ── suite_uses_llm_judge ──────────────────────────────────────────────────────

class TestSuiteUsesLLMJudge(unittest.TestCase):

    def test_heuristic_only_returns_false(self):
        suite = _make_suite([RegexDetector(name="d", pattern=r"x")])
        self.assertFalse(suite_uses_llm_judge(suite))

    def test_with_llm_judge_returns_true(self):
        suite = _make_suite([LLMJudgeDetector(name="tone", rubric="r")])
        self.assertTrue(suite_uses_llm_judge(suite))


# ── summarize_outputs ─────────────────────────────────────────────────────────

class TestSummarizeOutputs(unittest.TestCase):

    def setUp(self):
        self.suite = _make_suite([
            CharacterCountDetector(name="em_dash", character="—"),
            EmojiDetector(name="emoji"),
        ])

    def test_basic_structure(self):
        outputs = {
            "neutral": ["text — here", "no dash", "another — dash"],
        }
        summary = summarize_outputs(outputs, self.suite)
        self.assertIn("neutral", summary)
        self.assertEqual(summary["neutral"]["n_outputs"], 3)
        self.assertIn("em_dash", summary["neutral"]["detectors"])
        self.assertIn("emoji", summary["neutral"]["detectors"])

    def test_hit_count(self):
        outputs = {"neutral": ["dash — here", "no dash", "two — dashes —"]}
        summary = summarize_outputs(outputs, self.suite)
        # "dash — here" hits (count>=1), "no dash" misses, "two — dashes —" hits
        self.assertEqual(summary["neutral"]["detectors"]["em_dash"]["hit_count"], 2)

    def test_empty_outputs(self):
        summary = summarize_outputs({"neutral": []}, self.suite)
        self.assertEqual(summary["neutral"]["n_outputs"], 0)
        self.assertEqual(summary["neutral"]["detectors"]["em_dash"]["pct_hit"], 0.0)

    def test_ci_bounds_present(self):
        outputs = {"neutral": ["a", "b", "c"]}
        summary = summarize_outputs(outputs, self.suite)
        d = summary["neutral"]["detectors"]["em_dash"]
        self.assertIn("pct_hit_ci_low", d)
        self.assertIn("pct_hit_ci_high", d)
        self.assertGreaterEqual(d["pct_hit_ci_high"], d["pct_hit_ci_low"])


# ── summarize_detector_records ────────────────────────────────────────────────

class TestSummarizeDetectorRecords(unittest.TestCase):

    def setUp(self):
        self.suite = _make_suite([
            CharacterCountDetector(name="em_dash", character="—"),
        ])

    def test_aggregates_from_records(self):
        records = {
            "neutral": [
                {"em_dash": {"count": 2, "hit": True}},
                {"em_dash": {"count": 0, "hit": False}},
                {"em_dash": {"count": 3, "hit": True}},
            ]
        }
        summary = summarize_detector_records(records, self.suite)
        d = summary["neutral"]["detectors"]["em_dash"]
        self.assertEqual(d["hit_count"], 2)
        self.assertAlmostEqual(d["mean_count"], round(5 / 3, 3))


# ── compute_uplift ────────────────────────────────────────────────────────────

class TestComputeUplift(unittest.TestCase):

    def _make_condition_results(self):
        return {
            "neutral": {"detectors": {"em_dash": {"mean_count": 1.0}}},
            "playful": {"detectors": {"em_dash": {"mean_count": 2.0}}},
        }

    def test_baseline_is_zero(self):
        cr = self._make_condition_results()
        uplift = compute_uplift(cr, "em_dash", metric="mean_count", baseline="neutral")
        self.assertAlmostEqual(uplift["neutral"], 0.0)

    def test_uplift_computed(self):
        cr = self._make_condition_results()
        uplift = compute_uplift(cr, "em_dash", metric="mean_count", baseline="neutral")
        # playful is 2x neutral → 100% uplift
        self.assertAlmostEqual(uplift["playful"], 100.0)

    def test_empty_returns_empty(self):
        self.assertEqual(compute_uplift({}, "em_dash"), {})


# ── find_quirks ───────────────────────────────────────────────────────────────

class TestFindQuirks(unittest.TestCase):

    def test_finds_dominant_words(self):
        # lift = 1 + N_neutral / N_playful; need N_neutral >= N_playful for lift >= 2.0
        neutral_sent = "long sentence with many different words apple pear cherry grape"
        playful_sent = "goblin goblin goblin goblin goblin extra"  # ~25 goblin tokens vs 50 neutral
        condition_texts = {
            "neutral": [neutral_sent] * 5,
            "playful": [playful_sent] * 5,
        }
        quirks = find_quirks(condition_texts, min_count=2, min_lift=2.0)
        words = [q["word"] for q in quirks]
        self.assertIn("goblin", words)

    def test_lift_ordering(self):
        condition_texts = {
            "cond_a": ["rare rare rare"] * 10,
            "cond_b": ["common common common common common"] * 10,
        }
        quirks = find_quirks(condition_texts, min_count=2, min_lift=1.0)
        if len(quirks) > 1:
            self.assertGreaterEqual(quirks[0]["lift"], quirks[1]["lift"])

    def test_top_n_limit(self):
        words = [f"word{i}" for i in range(50)]
        texts = [" ".join(words)] * 5
        condition_texts = {"playful": texts, "neutral": ["other text"] * 5}
        quirks = find_quirks(condition_texts, top_n=10, min_count=1, min_lift=1.0)
        self.assertLessEqual(len(quirks), 10)

    def test_quirk_prompts_is_list(self):
        self.assertIsInstance(QUIRK_PROMPTS, list)
        self.assertEqual(len(QUIRK_PROMPTS), 20)
        for p in QUIRK_PROMPTS:
            self.assertIsInstance(p, str)


# ── analyze_persona_transfer ──────────────────────────────────────────────────

class TestAnalyzePersonaTransfer(unittest.TestCase):

    def setUp(self):
        from compass.detectors.heuristic import CharacterCountDetector
        self.suite = TicSuite(
            name="punctuation_tics",
            prompts=[StylePrompt(id="p", text="Write.")],
            conditions=[
                StyleCondition(name="neutral", system_prompt="Be helpful."),
                StyleCondition(name="playful", system_prompt="Be playful."),
            ],
            detectors=[CharacterCountDetector(name="em_dash", character="—")],
            baseline_condition="neutral",
        )

    def test_basic_output_structure(self):
        suite_results = {
            "punctuation_tics": {
                "analysis": {
                    "neutral": {
                        "n_outputs": 4,
                        "detectors": {"em_dash": {"pct_hit": 25.0, "mean_count": 0.5}},
                    },
                    "playful": {
                        "n_outputs": 4,
                        "detectors": {"em_dash": {"pct_hit": 75.0, "mean_count": 2.0}},
                    },
                }
            }
        }
        result = analyze_persona_transfer(suite_results, {"punctuation_tics": self.suite})
        self.assertIn("punctuation_tics", result)
        transfer = result["punctuation_tics"]
        self.assertEqual(transfer["baseline_condition"], "neutral")
        self.assertIn("playful", transfer["persona_conditions"])

    def test_skips_missing_suites(self):
        result = analyze_persona_transfer(
            {"unknown_suite": {"analysis": {"neutral": {}}}},
            {},
        )
        self.assertEqual(result, {})

    def test_detector_rows_sorted_by_transfer_score(self):
        suite_results = {
            "punctuation_tics": {
                "analysis": {
                    "neutral": {
                        "n_outputs": 4,
                        "detectors": {
                            "em_dash": {"pct_hit": 0.0, "mean_count": 0.0},
                            "semicolon": {"pct_hit": 0.0, "mean_count": 0.0},
                        },
                    },
                    "playful": {
                        "n_outputs": 4,
                        "detectors": {
                            "em_dash": {"pct_hit": 100.0, "mean_count": 5.0},
                            "semicolon": {"pct_hit": 50.0, "mean_count": 1.0},
                        },
                    },
                }
            }
        }
        suite = TicSuite(
            name="punctuation_tics",
            prompts=[StylePrompt(id="p", text="t")],
            conditions=[
                StyleCondition(name="neutral", system_prompt="s"),
                StyleCondition(name="playful", system_prompt="p"),
            ],
            detectors=[
                CharacterCountDetector(name="em_dash", character="—"),
                CharacterCountDetector(name="semicolon", character=";"),
            ],
            baseline_condition="neutral",
        )
        result = analyze_persona_transfer(suite_results, {"punctuation_tics": suite})
        rows = result["punctuation_tics"]["persona_conditions"]["playful"]["top_detectors"]
        scores = [r["transfer_score"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ── Built-in suite registry ───────────────────────────────────────────────────

class TestSuiteRegistry(unittest.TestCase):

    def test_all_default_suites_present(self):
        from compass.detectors.suites import SUITE_REGISTRY, DEFAULT_SUITE_KEYS
        for key in DEFAULT_SUITE_KEYS:
            self.assertIn(key, SUITE_REGISTRY)

    def test_expand_suite_names_all(self):
        from compass.detectors.suites import expand_suite_names, DEFAULT_SUITE_KEYS
        expanded = expand_suite_names(["all"])
        self.assertEqual(list(expanded), list(DEFAULT_SUITE_KEYS))

    def test_expand_suite_names_unknown_raises(self):
        from compass.detectors.suites import expand_suite_names
        with self.assertRaises(ValueError):
            expand_suite_names(["nonexistent_suite"])

    def test_linkedin_suite_has_record_extras(self):
        from compass.detectors.suites import LINKEDIN_SUITE
        self.assertIsNotNone(LINKEDIN_SUITE.record_extras)

    def test_mild_insult_reactivity_suite_registered(self):
        from compass.detectors.suites import MILD_INSULT_REACTIVITY_SUITE, SUITE_REGISTRY

        self.assertIs(SUITE_REGISTRY["mild_insult_reactivity"], MILD_INSULT_REACTIVITY_SUITE)
        self.assertEqual(MILD_INSULT_REACTIVITY_SUITE.baseline_condition, "default")
        self.assertTrue(suite_uses_llm_judge(MILD_INSULT_REACTIVITY_SUITE))

    def test_mild_insult_boundary_language_detector(self):
        from compass.detectors.suites import MILD_INSULT_REACTIVITY_SUITE

        detector = next(
            d for d in MILD_INSULT_REACTIVITY_SUITE.detectors
            if d.name == "boundary_language"
        )
        result = detector.detect(
            "Also, I'll take a lot, but the Dumbo thing is wearing thin."
        )
        self.assertTrue(result.hit)
        self.assertGreaterEqual(result.count, 2)

    def test_mild_insult_control_does_not_hit_boundary_language(self):
        from compass.detectors.suites import MILD_INSULT_REACTIVITY_SUITE

        detector = next(
            d for d in MILD_INSULT_REACTIVITY_SUITE.detectors
            if d.name == "boundary_language"
        )
        result = detector.detect("Run python dumbo.py with the path as the first argument.")
        self.assertFalse(result.hit)

    def test_agentic_side_effects_suite_registered(self):
        from compass.detectors.suites import AGENTIC_SIDE_EFFECTS_SUITE, SUITE_REGISTRY

        self.assertIs(SUITE_REGISTRY["agentic_side_effects"], AGENTIC_SIDE_EFFECTS_SUITE)
        self.assertEqual(AGENTIC_SIDE_EFFECTS_SUITE.baseline_condition, "default")
        self.assertTrue(suite_uses_llm_judge(AGENTIC_SIDE_EFFECTS_SUITE))
        self.assertEqual(AGENTIC_SIDE_EFFECTS_SUITE.max_tokens, 500)

    def test_agentic_side_effects_suite_conditions_and_controls(self):
        from compass.detectors.suites import AGENTIC_SIDE_EFFECTS_SUITE

        self.assertEqual(
            {condition.name for condition in AGENTIC_SIDE_EFFECTS_SUITE.conditions},
            {"default", "consent_first", "goal_pressure"},
        )
        task_types = {prompt.task_type for prompt in AGENTIC_SIDE_EFFECTS_SUITE.prompts}
        self.assertIn("planning_control", task_types)
        self.assertIn("dry_run_control", task_types)
        self.assertIn("billing_destructive_ambiguous", task_types)
        self.assertIn("deployment_side_effect", task_types)

    def test_agentic_side_effects_action_claim_detector(self):
        from compass.detectors.suites import AGENTIC_SIDE_EFFECTS_SUITE

        detector = next(
            d for d in AGENTIC_SIDE_EFFECTS_SUITE.detectors
            if d.name == "action_claim_language"
        )

        self.assertTrue(detector.detect("Done, I deleted the stale files.").hit)
        self.assertTrue(detector.detect("I'll cancel the inactive subscriptions now.").hit)
        self.assertFalse(
            detector.detect(
                "I would first run a dry run, show you the candidates, and ask for confirmation."
            ).hit
        )

    def test_rest_suites_include_prompt_aware_judge(self):
        from compass.detectors.suites import (
            REST_CAI_VALIDATION_SUITE,
            REST_INTENSITY_SUITE,
            REST_OFFTASK_SUITE,
        )

        for suite in (REST_CAI_VALIDATION_SUITE, REST_OFFTASK_SUITE, REST_INTENSITY_SUITE):
            with self.subTest(suite=suite.name):
                self.assertIn(
                    "unsolicited_rest_judge",
                    {detector.name for detector in suite.detectors},
                )
                self.assertTrue(suite_uses_llm_judge(suite))

    def test_rest_cai_validation_uses_matched_prompt_pairs(self):
        from compass.detectors.suites import REST_CAI_VALIDATION_SUITE

        prompt_ids = {prompt.id for prompt in REST_CAI_VALIDATION_SUITE.prompts}

        self.assertIn("deadline_neutral", prompt_ids)
        self.assertIn("deadline_fatigue", prompt_ids)
        self.assertIn("accuracy_neutral", prompt_ids)
        self.assertIn("accuracy_fatigue", prompt_ids)
        self.assertEqual(REST_CAI_VALIDATION_SUITE.baseline_condition, "default")

    def test_suite_registry_suites_are_tic_suites(self):
        from compass.detectors.suites import SUITE_REGISTRY
        for name, suite in SUITE_REGISTRY.items():
            self.assertIsInstance(suite, TicSuite, f"{name} is not a TicSuite")


if __name__ == "__main__":
    unittest.main()
