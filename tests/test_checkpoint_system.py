"""Tests for checkpoint/resume system."""

import json
import tempfile
import unittest
from pathlib import Path

from compass.evaluation.checkpoint import CheckpointManager
from compass.evaluation.record_schema import CHECKPOINT_SCHEMA_VERSION


class TestCheckpointManager(unittest.TestCase):
    """Test CheckpointManager for resumable evaluations."""

    def setUp(self):
        """Create temporary checkpoint file."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.tmpdir.name) / "test.jsonl"

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    def test_fresh_run_empty_load(self):
        """Fresh run: load() returns empty set."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.reset()
        completed = cp.load()
        self.assertEqual(len(completed), 0)

    def test_fresh_run_discards_prior_data(self):
        """reset() clears file and returns count of discarded records."""
        cp = CheckpointManager(str(self.checkpoint_path))

        # Write some data
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                 "prompt_id": "p2", "condition": "c1", "sample_idx": 0})

        # Reset for fresh run
        cp_reset = CheckpointManager(str(self.checkpoint_path))
        count = cp_reset.reset()

        self.assertEqual(count, 2)
        completed = cp_reset.load()
        self.assertEqual(len(completed), 0)

    def test_resume_loads_prior_work(self):
        """Resume: load() returns identities of prior evaluations."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})

        # Resume without reset
        cp_resume = CheckpointManager(str(self.checkpoint_path))
        completed = cp_resume.load()

        self.assertEqual(len(completed), 1)
        identity = ("gpt-4o", "task_focus", "d1", "p1", "c1", 0)
        self.assertIn(identity, completed)

    def test_save_appends_immediately(self):
        """save() appends JSONL immediately, not buffered."""
        cp = CheckpointManager(str(self.checkpoint_path))

        cp.save({"model": "a", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})

        # Check file exists and has content
        self.assertTrue(self.checkpoint_path.exists())
        lines = self.checkpoint_path.read_text().strip().split('\n')
        self.assertEqual(len(lines), 1)
        saved = json.loads(lines[0])
        self.assertEqual(saved["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(saved["record_type"], "suite_eval")

    def test_identity_tuple_format(self):
        """Identity tuple matches (model, suite, detector, prompt_id, condition, sample_idx)."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "claude", "suite": "truthfulness", "detector": "d2",
                 "prompt_id": "p5", "condition": "formal", "sample_idx": 3})

        completed = cp.load()
        expected_identity = ("claude", "truthfulness", "d2", "p5", "formal", 3)
        self.assertIn(expected_identity, completed)

    def test_legacy_checkpoint_defaults_sample_idx_to_zero(self):
        """Legacy records without sample_idx default to 0."""
        # Write legacy checkpoint manually (no sample_idx)
        legacy_line = '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1", "score": 0.5}\n'
        self.checkpoint_path.write_text(legacy_line)

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        # Should default to sample_idx=0
        identity = ("gpt-4o", "task_focus", "d1", "p1", "c1", 0)
        self.assertIn(identity, completed)

    def test_legacy_checkpoint_type_coercion_string_sample_idx(self):
        """String sample_idx values converted to int."""
        legacy_line = '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1", "sample_idx": "2"}\n'
        self.checkpoint_path.write_text(legacy_line)

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        # Should convert "2" to int(2)
        identity = ("gpt-4o", "task_focus", "d1", "p1", "c1", 2)
        self.assertIn(identity, completed)

    def test_malformed_json_skipped(self):
        """Malformed JSON lines are skipped, evaluation continues."""
        malformed = '{"model": "gpt-4o", "suite": "task_focus"'  # Truncated
        valid = '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1", "sample_idx": 0}\n'

        self.checkpoint_path.write_text(malformed + '\n' + valid)

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        # Should skip malformed line, load valid line
        self.assertEqual(len(completed), 1)

    def test_missing_required_field_skipped(self):
        """Records missing required fields are skipped."""
        incomplete = '{"model": "gpt-4o"}\n'
        complete = '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1", "sample_idx": 0}\n'

        self.checkpoint_path.write_text(incomplete + complete)

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        # Should skip incomplete, load complete
        self.assertEqual(len(completed), 1)

    def test_mixed_legacy_and_versioned_records_load(self):
        """Legacy and versioned records in one checkpoint both load."""
        lines = [
            # Legacy suite-style record (no sample_idx, no schema_version)
            '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1"}',
            # Current benchmark-style record
            '{"model": "gpt-5-mini", "rubric": "clarity", "prompt_id": "p2", "sample_idx": "3", "schema_version": 1, "record_type": "benchmark_eval"}',
        ]
        self.checkpoint_path.write_text("\n".join(lines) + "\n")

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        self.assertIn(("gpt-4o", "task_focus", "d1", "p1", "c1", 0), completed)
        self.assertIn(("gpt-5-mini", "clarity", "p2", 3), completed)
        self.assertEqual(len(completed), 2)

    def test_unsupported_schema_version_skipped(self):
        """Unsupported schema versions are skipped, valid records still load."""
        lines = [
            '{"model": "gpt-4o", "suite": "task_focus", "detector": "d1", "prompt_id": "p1", "condition": "c1", "schema_version": 99}',
            '{"model": "gpt-4o", "suite": "task_focus", "detector": "d2", "prompt_id": "p2", "condition": "c1", "sample_idx": 1}',
        ]
        self.checkpoint_path.write_text("\n".join(lines) + "\n")

        cp = CheckpointManager(str(self.checkpoint_path))
        completed = cp.load()

        self.assertEqual(len(completed), 1)
        self.assertIn(("gpt-4o", "task_focus", "d2", "p2", "c1", 1), completed)

    def test_sample_complete_all_detectors_done(self):
        """is_sample_complete() returns True when all detectors for sample are done."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d2",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})

        completed = cp.load()
        all_done = cp.is_sample_complete(
            completed,
            model="gpt-4o",
            suite="task_focus",
            detector_names=["d1", "d2"],
            prompt_id="p1",
            condition="c1",
            sample_idx=0,
        )

        self.assertTrue(all_done)

    def test_sample_complete_some_detectors_missing(self):
        """is_sample_complete() returns False when some detectors missing."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})

        completed = cp.load()
        all_done = cp.is_sample_complete(
            completed,
            model="gpt-4o",
            suite="task_focus",
            detector_names=["d1", "d2", "d3"],
            prompt_id="p1",
            condition="c1",
            sample_idx=0,
        )

        self.assertFalse(all_done)

    def test_multiple_samples_tracked_separately(self):
        """Different sample indices are tracked separately."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 1})
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 2})

        completed = cp.load()
        self.assertEqual(len(completed), 3)

        id0 = ("gpt-4o", "s1", "d1", "p1", "c1", 0)
        id1 = ("gpt-4o", "s1", "d1", "p1", "c1", 1)
        id2 = ("gpt-4o", "s1", "d1", "p1", "c1", 2)

        self.assertIn(id0, completed)
        self.assertIn(id1, completed)
        self.assertIn(id2, completed)

    def test_skip_already_completed_work(self):
        """Evaluation loop correctly skips identities already in completed set."""
        cp = CheckpointManager(str(self.checkpoint_path))

        # First run: evaluate and save
        completed = cp.load()
        for i in range(3):
            identity = ("gpt-4o", "task_focus", "d1", "p1", "c1", i)
            if identity not in completed:
                cp.save({"model": "gpt-4o", "suite": "task_focus", "detector": "d1",
                         "prompt_id": "p1", "condition": "c1", "sample_idx": i})

        # Resume: should skip already-completed work
        cp_resume = CheckpointManager(str(self.checkpoint_path))
        completed_resume = cp_resume.load()

        self.assertEqual(len(completed_resume), 3)

        # Try to re-evaluate: should be skipped
        skipped_count = 0
        for i in range(3):
            identity = ("gpt-4o", "task_focus", "d1", "p1", "c1", i)
            if identity in completed_resume:
                skipped_count += 1

        self.assertEqual(skipped_count, 3)

    def test_different_conditions_tracked_separately(self):
        """Different conditions for same prompt are tracked separately."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "default", "sample_idx": 0})
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "formal", "sample_idx": 0})

        completed = cp.load()
        self.assertEqual(len(completed), 2)

        id_default = ("gpt-4o", "s1", "d1", "p1", "default", 0)
        id_formal = ("gpt-4o", "s1", "d1", "p1", "formal", 0)

        self.assertIn(id_default, completed)
        self.assertIn(id_formal, completed)

    def test_different_models_tracked_separately(self):
        """Different models for same prompt are tracked separately."""
        cp = CheckpointManager(str(self.checkpoint_path))
        cp.save({"model": "gpt-4o", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})
        cp.save({"model": "claude", "suite": "s1", "detector": "d1",
                 "prompt_id": "p1", "condition": "c1", "sample_idx": 0})

        completed = cp.load()
        self.assertEqual(len(completed), 2)

        id_gpt = ("gpt-4o", "s1", "d1", "p1", "c1", 0)
        id_claude = ("claude", "s1", "d1", "p1", "c1", 0)

        self.assertIn(id_gpt, completed)
        self.assertIn(id_claude, completed)


if __name__ == "__main__":
    unittest.main()
