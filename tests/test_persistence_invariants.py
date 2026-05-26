"""Cross-layer persistence invariants for benchmark artifacts and checkpoints."""

import tempfile
import unittest
from pathlib import Path

from compass.benchmark.io import load_evaluation_records, load_generation_records
from compass.evaluation.checkpoint import CheckpointManager


class PersistenceInvariantTests(unittest.TestCase):
    def test_strict_generation_reader_rejects_truncated_final_line(self):
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","completion":"ok"}',
            '{"model":"llama3.1","rubric":"clarity"',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "generations.jsonl"
            path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(ValueError, "invalid generation row"):
                load_generation_records(path, strict=True)

    def test_strict_evaluation_reader_rejects_truncated_final_line(self):
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","score":0.5,"hit":true}',
            '{"model":"llama3.1","rubric":"clarity"',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evaluations.jsonl"
            path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(ValueError, "invalid evaluation row"):
                load_evaluation_records(path, strict=True)

    def test_lenient_benchmark_reader_salvages_valid_prefix(self):
        lines = [
            '{"model":"llama3.1","rubric":"clarity","prompt_id":"p1","score":0.5,"hit":true}',
            '{"model":"llama3.1","rubric":"clarity"',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evaluations.jsonl"
            path.write_text("\n".join(lines) + "\n")

            with self.assertLogs("compass.benchmark.io", level="WARNING") as logs:
                rows = load_evaluation_records(path, strict=False)

        self.assertEqual(len(rows), 1)
        self.assertIn("Skipping invalid evaluation row", logs.output[0])

    def test_checkpoint_loader_ignores_truncated_final_line(self):
        lines = [
            '{"model":"gpt-4o","suite":"task_focus","detector":"d1","prompt_id":"p1","condition":"c1","sample_idx":0}',
            '{"model":"gpt-4o","suite":"task_focus"',
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.jsonl"
            path.write_text("\n".join(lines) + "\n")

            checkpoint = CheckpointManager(str(path))
            with self.assertLogs("compass.evaluation.checkpoint", level="WARNING") as logs:
                completed = checkpoint.load()

        self.assertEqual(completed, {("gpt-4o", "task_focus", "d1", "p1", "c1", 0)})
        self.assertIn("Skipping malformed JSON at checkpoint line 2", logs.output[0])


if __name__ == "__main__":
    unittest.main()
