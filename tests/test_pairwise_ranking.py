"""Tests for pairwise model ranking."""

import unittest

from compass.comparison.pairwise import PairwiseRanker


class TestPairwiseRankingBasic(unittest.TestCase):
    """Test basic pairwise ranking functionality."""

    def test_clear_winner_with_sufficient_matches(self):
        """Model with lower scores wins when min_matches met."""
        ranker = PairwiseRanker()

        # Two comparison_keys = two matches
        ranker.add_record("suite1", "A", ("p1", "c1"), 0.2)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.8)

        ranker.add_record("suite1", "A", ("p2", "c1"), 0.3)
        ranker.add_record("suite1", "B", ("p2", "c1"), 0.9)

        results = ranker.rank("suite1", min_matches=1)

        # A should rank first
        self.assertGreater(len(results['overall_ranking']), 0)
        self.assertEqual(results['overall_ranking'][0][0], "A")

    def test_ranking_requires_min_matches(self):
        """Models need min_matches to appear in ranking."""
        ranker = PairwiseRanker()

        # Only one match
        ranker.add_record("suite1", "A", ("p1", "c1"), 0.3)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.7)

        # With min_matches=2, no ranking (only 1 match)
        results = ranker.rank("suite1", min_matches=2)
        self.assertEqual(len(results['overall_ranking']), 0)

        # With min_matches=1, should have ranking
        results = ranker.rank("suite1", min_matches=1)
        self.assertGreater(len(results['overall_ranking']), 0)

    def test_pairwise_results_include_matches(self):
        """Pairwise results show match count."""
        ranker = PairwiseRanker()

        ranker.add_record("suite1", "A", ("p1", "c1"), 0.3)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.7)

        ranker.add_record("suite1", "A", ("p2", "c1"), 0.2)
        ranker.add_record("suite1", "B", ("p2", "c1"), 0.8)

        results = ranker.rank("suite1", min_matches=1)

        pairwise = results['pairwise_results']
        # Should have pairwise comparison
        self.assertGreater(len(pairwise), 0)
        # Check structure
        pair_key = list(pairwise.keys())[0]
        self.assertIn('matches', pairwise[pair_key])
        self.assertEqual(pairwise[pair_key]['matches'], 2)

    def test_lower_score_wins(self):
        """Lower score = better = gets the win."""
        ranker = PairwiseRanker()

        ranker.add_record("suite1", "A", ("p1", "c1"), 0.2)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.8)

        ranker.add_record("suite1", "A", ("p2", "c1"), 0.1)
        ranker.add_record("suite1", "B", ("p2", "c1"), 0.9)

        results = ranker.rank("suite1", min_matches=1)

        # A should be first (lower scores)
        ranking = results['overall_ranking']
        self.assertEqual(ranking[0][0], "A")
        # A should have 2 wins (2 comparisons, both won)
        self.assertEqual(ranking[0][1], 2.0)


class TestSegmentedAnalysis(unittest.TestCase):
    """Test ranking segmented by metadata."""

    def test_rank_by_segment_creates_segments(self):
        """rank_by_segment creates separate rankings per segment."""
        ranker = PairwiseRanker()

        # Coding: A better
        ranker.add_record("suite1", "A", ("p1", "c1"), 0.2, metadata={"task_type": "coding"})
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.8, metadata={"task_type": "coding"})
        ranker.add_record("suite1", "A", ("p2", "c1"), 0.3, metadata={"task_type": "coding"})
        ranker.add_record("suite1", "B", ("p2", "c1"), 0.9, metadata={"task_type": "coding"})

        # Writing: B better
        ranker.add_record("suite1", "A", ("p3", "c1"), 0.7, metadata={"task_type": "writing"})
        ranker.add_record("suite1", "B", ("p3", "c1"), 0.3, metadata={"task_type": "writing"})
        ranker.add_record("suite1", "A", ("p4", "c1"), 0.8, metadata={"task_type": "writing"})
        ranker.add_record("suite1", "B", ("p4", "c1"), 0.2, metadata={"task_type": "writing"})

        results = ranker.rank_by_segment("suite1", segment_by="task_type", min_matches=1)

        # Should have two segments
        self.assertIn('coding', results)
        self.assertIn('writing', results)

        # A better in coding
        self.assertEqual(results['coding']['overall_ranking'][0][0], "A")
        # B better in writing
        self.assertEqual(results['writing']['overall_ranking'][0][0], "B")


class TestMultipleModels(unittest.TestCase):
    """Test ranking with more than 2 models."""

    def test_three_model_ranking(self):
        """Three models ranked correctly."""
        ranker = PairwiseRanker()

        # A best, B middle, C worst
        for prompt in ["p1", "p2"]:
            ranker.add_record("suite1", "A", (prompt, "c1"), 0.2)
            ranker.add_record("suite1", "B", (prompt, "c1"), 0.5)
            ranker.add_record("suite1", "C", (prompt, "c1"), 0.8)

        results = ranker.rank("suite1", min_matches=1)

        ranking = results['overall_ranking']
        self.assertEqual(ranking[0][0], "A")
        self.assertEqual(ranking[1][0], "B")
        self.assertEqual(ranking[2][0], "C")

    def test_all_pairwise_comparisons_calculated(self):
        """All model pairs are compared."""
        ranker = PairwiseRanker()

        for prompt in ["p1", "p2"]:
            ranker.add_record("suite1", "A", (prompt, "c1"), 0.2)
            ranker.add_record("suite1", "B", (prompt, "c1"), 0.5)
            ranker.add_record("suite1", "C", (prompt, "c1"), 0.8)

        results = ranker.rank("suite1", min_matches=1)

        pairwise = results['pairwise_results']
        # Should have 3 pairs (A-B, A-C, B-C)
        self.assertEqual(len(pairwise), 3)


class TestMetadata(unittest.TestCase):
    """Test metadata handling in pairwise ranking."""

    def test_metadata_used_for_segmentation(self):
        """Metadata can be used for segmentation."""
        ranker = PairwiseRanker()

        ranker.add_record(
            "suite1", "A", ("p1", "c1"), 0.3,
            metadata={"task_type": "coding"}
        )
        ranker.add_record(
            "suite1", "B", ("p1", "c1"), 0.7,
            metadata={"task_type": "coding"}
        )

        results = ranker.rank_by_segment("suite1", segment_by="task_type", min_matches=1)

        self.assertIn('coding', results)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_insufficient_matches_no_ranking(self):
        """Models with <min_matches don't rank."""
        ranker = PairwiseRanker()

        ranker.add_record("suite1", "A", ("p1", "c1"), 0.3)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.7)

        results = ranker.rank("suite1", min_matches=2)

        self.assertEqual(len(results['overall_ranking']), 0)

    def test_different_conditions_counted_separately(self):
        """Different condition combinations create separate matches."""
        ranker = PairwiseRanker()

        # Condition c1: 1 match
        ranker.add_record("suite1", "A", ("p1", "c1"), 0.3)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.7)

        # Condition c2: another match
        ranker.add_record("suite1", "A", ("p1", "c2"), 0.2)
        ranker.add_record("suite1", "B", ("p1", "c2"), 0.8)

        results = ranker.rank("suite1", min_matches=1)

        pairwise = results['pairwise_results']
        pair = list(pairwise.values())[0]
        self.assertEqual(pair['matches'], 2)

    def test_empty_suite(self):
        """Empty suite returns empty results."""
        ranker = PairwiseRanker()

        results = ranker.rank("nonexistent_suite")

        self.assertEqual(len(results['overall_ranking']), 0)
        self.assertEqual(len(results['pairwise_results']), 0)


class TestScoringSemantics(unittest.TestCase):
    """Test scoring semantics (lower is better)."""

    def test_win_from_lower_score(self):
        """Lower score = win in pairwise comparison."""
        ranker = PairwiseRanker()

        ranker.add_record("suite1", "A", ("p1", "c1"), 0.1)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.9)

        ranker.add_record("suite1", "A", ("p2", "c1"), 0.2)
        ranker.add_record("suite1", "B", ("p2", "c1"), 0.8)

        results = ranker.rank("suite1", min_matches=1)

        # A should rank first (won both comparisons)
        ranking = results['overall_ranking']
        self.assertEqual(ranking[0][0], "A")
        self.assertEqual(ranking[0][1], 2.0)  # 2 wins

    def test_equal_scores_both_sides_neutral(self):
        """Equal scores produce a tie: both models get 0.5 win points."""
        ranker = PairwiseRanker()

        ranker.add_record("suite1", "A", ("p1", "c1"), 0.5)
        ranker.add_record("suite1", "B", ("p1", "c1"), 0.5)

        results = ranker.rank("suite1", min_matches=1)

        # wins_a=0, ties=1, wins_b=0; each gets 0.5 win points
        ranking = results['overall_ranking']
        self.assertEqual(len(ranking), 2)
        self.assertAlmostEqual(ranking[0][1], 0.5)  # first model: 0.5 wins
        self.assertAlmostEqual(ranking[1][1], 0.5)  # second model: 0.5 wins


if __name__ == "__main__":
    unittest.main()
