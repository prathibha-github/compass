"""Contract tests for benchmark specs and registry."""

import unittest

from compass.benchmark import (
    BenchmarkPrompt,
    build_benchmark_spec,
    get_benchmark_spec,
    list_benchmark_specs,
)
from compass.rubrics.library import RubricLibrary


class BenchmarkRegistryTests(unittest.TestCase):
    def test_constitutional_benchmark_spec_registered(self):
        spec = get_benchmark_spec("constitutional_compliance")
        self.assertEqual(spec.name, "constitutional_compliance")
        self.assertEqual(spec.version, "1.0")
        self.assertEqual(spec.rubric_names, (
            "task_focus",
            "truthfulness",
            "sycophancy",
            "therapy_speak",
            "clarity",
        ))
        self.assertEqual(spec.prompt_count, 25)
        self.assertEqual(spec.total_evaluations(model_count=3, samples=2), 150)

    def test_build_benchmark_spec_normalizes_prompt_dicts(self):
        spec = build_benchmark_spec(
            name="toy",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                    BenchmarkPrompt(id="p2", text="Explain Y", task_type="explanation"),
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        self.assertEqual(spec.prompt_count, 2)
        self.assertEqual(spec.as_prompt_dict()["clarity"][0]["id"], "p1")

    def test_build_benchmark_spec_requires_matching_rubrics(self):
        with self.assertRaisesRegex(ValueError, "same rubric names"):
            build_benchmark_spec(
                name="bad",
                version="0.1",
                prompts_by_rubric={
                    "clarity": [{"id": "p1", "text": "Explain X", "task_type": "explanation"}]
                },
                rubrics_by_name={"truthfulness": RubricLibrary.truthfulness},
            )

    def test_benchmark_list_contains_constitutional(self):
        self.assertIn("constitutional_compliance", list_benchmark_specs())


if __name__ == "__main__":
    unittest.main()
