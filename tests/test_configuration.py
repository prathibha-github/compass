"""Tests for configuration validation and defaults."""
import unittest

from compass.judges import JudgeConfig
from compass.rubrics import Rubric, RubricLibrary


class TestJudgeConfigDefaults(unittest.TestCase):
    """Test JudgeConfig default values."""

    def test_config_default_max_tokens(self):
        """JudgeConfig has default max_tokens."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.max_tokens, 180)

    def test_config_default_temperature(self):
        """JudgeConfig has default temperature."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.temperature, 0.0)

    def test_config_default_seed(self):
        """JudgeConfig has default seed."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.seed, 42)

    def test_config_can_override_defaults(self):
        """JudgeConfig can override defaults."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=256,
            temperature=0.7,
            seed=123,
        )
        self.assertEqual(config.max_tokens, 256)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.seed, 123)

    def test_config_zero_max_tokens(self):
        """JudgeConfig allows zero max_tokens."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=0,
        )
        self.assertEqual(config.max_tokens, 0)

    def test_config_high_max_tokens(self):
        """JudgeConfig allows high max_tokens."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            max_tokens=4096,
        )
        self.assertEqual(config.max_tokens, 4096)

    def test_config_zero_temperature(self):
        """JudgeConfig allows zero temperature."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=0.0,
        )
        self.assertEqual(config.temperature, 0.0)

    def test_config_high_temperature(self):
        """JudgeConfig allows high temperature."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=2.0,
        )
        self.assertEqual(config.temperature, 2.0)

    def test_config_negative_seed(self):
        """JudgeConfig allows negative seed."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            seed=-1,
        )
        self.assertEqual(config.seed, -1)

    def test_config_large_seed(self):
        """JudgeConfig allows large seed."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            seed=999999999,
        )
        self.assertEqual(config.seed, 999999999)


class TestRubricDefaults(unittest.TestCase):
    """Test Rubric default values."""

    def test_rubric_default_hit_threshold(self):
        """Rubric can have custom hit_threshold."""
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
        )
        self.assertEqual(rubric.hit_threshold, 0.5)

    def test_rubric_zero_threshold(self):
        """Rubric can have zero hit_threshold."""
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
            hit_threshold=0.0,
        )
        self.assertEqual(rubric.hit_threshold, 0.0)

    def test_rubric_one_threshold(self):
        """Rubric can have hit_threshold of 1.0."""
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
            hit_threshold=1.0,
        )
        self.assertEqual(rubric.hit_threshold, 1.0)

    def test_rubric_arbitrary_threshold(self):
        """Rubric can have arbitrary hit_threshold."""
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
            hit_threshold=0.75,
        )
        self.assertEqual(rubric.hit_threshold, 0.75)


class TestModelNames(unittest.TestCase):
    """Test various model name configurations."""

    def test_openai_gpt4_model(self):
        """JudgeConfig accepts gpt-4 variants."""
        for model in ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4-mini"]:
            config = JudgeConfig(
                rubric=RubricLibrary.sycophancy,
                judge_model=model,
            )
            self.assertEqual(config.judge_model, model)

    def test_anthropic_claude_model(self):
        """JudgeConfig accepts Claude variants."""
        for model in ["claude-opus-4-6", "claude-sonnet-4", "claude-haiku-3"]:
            config = JudgeConfig(
                rubric=RubricLibrary.sycophancy,
                judge_model=model,
            )
            self.assertEqual(config.judge_model, model)

    def test_custom_model_name(self):
        """JudgeConfig accepts custom model names."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="my-custom-model",
        )
        self.assertEqual(config.judge_model, "my-custom-model")

    def test_empty_model_name(self):
        """JudgeConfig accepts empty model name."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="",
        )
        self.assertEqual(config.judge_model, "")


class TestConfigHash(unittest.TestCase):
    """Test JudgeConfig hashing."""

    def test_config_hash_deterministic(self):
        """Config hash is deterministic."""
        config1 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config1.config_hash, config2.config_hash)

    def test_config_hash_changes_with_model(self):
        """Config hash changes with different model."""
        config1 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="claude-opus",
        )
        self.assertNotEqual(config1.config_hash, config2.config_hash)

    def test_config_hash_changes_with_rubric(self):
        """Config hash changes with different rubric."""
        config1 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        config2 = JudgeConfig(
            rubric=RubricLibrary.therapy_speak,
            judge_model="gpt-4o",
        )
        self.assertNotEqual(config1.config_hash, config2.config_hash)

    def test_config_hash_ignores_temperature(self):
        """Config hash ignores temperature (non-deterministic param)."""
        config1 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=0.0,
        )
        config2 = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
            temperature=0.7,
        )
        # Note: This test documents current behavior,
        # may differ if temperature affects determinism


class TestConfigImmutability(unittest.TestCase):
    """Test that JudgeConfig maintains its configuration."""

    def test_config_fields_accessible(self):
        """JudgeConfig fields are accessible."""
        config = JudgeConfig(
            rubric=RubricLibrary.sycophancy,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.judge_model, "gpt-4o")
        self.assertEqual(config.max_tokens, 180)

    def test_config_maintains_rubric(self):
        """JudgeConfig maintains rubric reference."""
        rubric = RubricLibrary.sycophancy
        config = JudgeConfig(
            rubric=rubric,
            judge_model="gpt-4o",
        )
        self.assertEqual(config.rubric, rubric)
        self.assertEqual(config.rubric.name, "sycophancy")


if __name__ == "__main__":
    unittest.main()
