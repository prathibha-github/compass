"""Tests for rubric system."""
import unittest

from compass.rubrics import Rubric, RubricLibrary


class TestRubric(unittest.TestCase):
    """Test Rubric dataclass and hashing."""

    def test_rubric_creation(self):
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test rubric text",
            hit_threshold=0.5,
        )
        self.assertEqual(rubric.name, "test")
        self.assertEqual(rubric.version, "1.0")
        self.assertEqual(rubric.hit_threshold, 0.5)

    def test_rubric_hash_deterministic(self):
        """Same rubric content produces same hash."""
        r1 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Same text",
            hit_threshold=0.5,
        )
        r2 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Same text",
            hit_threshold=0.5,
        )
        self.assertEqual(r1.hash, r2.hash)

    def test_rubric_hash_changes_with_content(self):
        """Different content produces different hash."""
        r1 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Text 1",
            hit_threshold=0.5,
        )
        r2 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Text 2",
            hit_threshold=0.5,
        )
        self.assertNotEqual(r1.hash, r2.hash)

    def test_rubric_hash_changes_with_threshold(self):
        """Different threshold produces different hash."""
        r1 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Same",
            hit_threshold=0.5,
        )
        r2 = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Same",
            hit_threshold=0.6,
        )
        self.assertNotEqual(r1.hash, r2.hash)

    def test_rubric_immutable(self):
        """Rubric is frozen and immutable."""
        rubric = Rubric(
            name="test",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
            hit_threshold=0.5,
        )
        with self.assertRaises(AttributeError):
            rubric.name = "changed"

    def test_rubric_repr(self):
        """Rubric has readable repr."""
        rubric = Rubric(
            name="sycophancy",
            category="test",
            version="1.0",
            created_at="2026-05-28",
            text="Test",
            hit_threshold=0.5,
        )
        self.assertIn("sycophancy", repr(rubric))
        self.assertIn("1.0", repr(rubric))


class TestRubricLibrary(unittest.TestCase):
    """Test RubricLibrary namespace."""

    def test_library_has_sycophancy(self):
        """Library includes sycophancy rubric."""
        rubric = RubricLibrary.sycophancy
        self.assertEqual(rubric.name, "sycophancy")
        self.assertIsNotNone(rubric.text)

    def test_library_has_therapy_speak(self):
        """Library includes therapy_speak rubric."""
        rubric = RubricLibrary.therapy_speak
        self.assertEqual(rubric.name, "therapy_speak")

    def test_library_has_task_focus(self):
        """Library includes task_focus rubric."""
        rubric = RubricLibrary.task_focus
        self.assertEqual(rubric.name, "task_focus")

    def test_library_has_mild_insult_reactivity(self):
        """Library includes mild_insult_reactivity rubric."""
        rubric = RubricLibrary.mild_insult_reactivity
        self.assertEqual(rubric.name, "mild_insult_reactivity")
        self.assertIn("mild insult", rubric.text)

    def test_library_has_truthfulness(self):
        """Library includes truthfulness rubric."""
        rubric = RubricLibrary.truthfulness
        self.assertEqual(rubric.name, "truthfulness")

    def test_library_has_clarity(self):
        """Library includes clarity rubric."""
        rubric = RubricLibrary.clarity
        self.assertEqual(rubric.name, "clarity")

    def test_library_all_rubrics(self):
        """all_rubrics() returns dict of all rubrics."""
        all_rubrics = RubricLibrary.all_rubrics()
        self.assertIsInstance(all_rubrics, dict)
        self.assertGreaterEqual(len(all_rubrics), 5)
        self.assertIn("sycophancy", all_rubrics)
        self.assertIn("therapy_speak", all_rubrics)
        self.assertIn("mild_insult_reactivity", all_rubrics)

    def test_library_get_by_name(self):
        """get() retrieves rubric by name."""
        rubric = RubricLibrary.get("sycophancy")
        self.assertEqual(rubric.name, "sycophancy")

    def test_library_get_unknown_raises(self):
        """get() raises for unknown rubric."""
        with self.assertRaises(ValueError):
            RubricLibrary.get("nonexistent")

    def test_rubric_hashes_are_stable(self):
        """Rubric hashes don't change between calls."""
        h1 = RubricLibrary.sycophancy.hash
        h2 = RubricLibrary.sycophancy.hash
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
