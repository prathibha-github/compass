"""Contract tests for benchmark specs and registry."""

import unittest

from compass.benchmark import (
    BenchmarkPolicyDefaults,
    BenchmarkPrompt,
    BenchmarkRunPreset,
    SharedBenchmarkRunner,
    build_benchmark_spec,
    get_benchmark_runner,
    get_benchmark_spec,
    list_benchmark_specs,
    register_benchmark_spec,
)
from compass.rubrics.library import RubricLibrary


class BenchmarkRegistryTests(unittest.TestCase):
    def test_constitutional_benchmark_spec_registered(self):
        spec = get_benchmark_spec("constitutional_compliance")
        self.assertEqual(spec.name, "constitutional_compliance")
        self.assertEqual(spec.version, "1.0")
        self.assertEqual(spec.default_preset, "default")
        self.assertEqual(spec.rubric_names, (
            "task_focus",
            "truthfulness",
            "sycophancy",
            "therapy_speak",
            "clarity",
        ))
        self.assertEqual(spec.prompt_count, 25)
        self.assertEqual(spec.total_evaluations(model_count=3, samples=2), 150)
        preset = spec.get_run_preset()
        self.assertEqual(preset.models, ("llama3.1", "mistral", "phi"))
        self.assertEqual(preset.judge_model, "llama3.1")
        self.assertEqual(preset.output_dir, "results/constitutional_compliance_benchmark")
        self.assertEqual(preset.policy.quality_filter_mode, "annotate")

    def test_constitutional_benchmark_runner_registered(self):
        runner = get_benchmark_runner("constitutional_compliance")
        spec = get_benchmark_spec("constitutional_compliance")
        self.assertIs(runner.spec, spec)

    def test_make_run_config_uses_preset_defaults_and_cli_overrides(self):
        spec = get_benchmark_spec("constitutional_compliance")
        run_config = spec.make_run_config(
            models=("gpt-4o-mini",),
            max_tokens_by_model={"gpt-4o-mini": 800},
            allow_mixed_token_budgets=True,
            skip_ranking=True,
        )
        self.assertEqual(run_config.models, ("gpt-4o-mini",))
        self.assertEqual(run_config.samples, 3)
        self.assertEqual(run_config.max_tokens_by_model["gpt-4o-mini"], 800)
        self.assertTrue(run_config.allow_mixed_token_budgets)
        self.assertEqual(run_config.effective_analysis_lanes, ("summary",))

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

    def test_build_benchmark_spec_is_immutable(self):
        spec = build_benchmark_spec(
            name="toy",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        with self.assertRaises(TypeError):
            spec.prompts_by_rubric["clarity"] = ()
        with self.assertRaises(TypeError):
            spec.rubrics_by_name["clarity"] = RubricLibrary.truthfulness

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

    def test_build_benchmark_spec_accepts_benchmark_owned_presets(self):
        spec = build_benchmark_spec(
            name="toy_preset",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
            run_presets={
                "ci": BenchmarkRunPreset(
                    models=("llama3.1",),
                    samples=2,
                    judge_model="llama3.1",
                    output_dir="results/toy_preset",
                    policy=BenchmarkPolicyDefaults(
                        token_budgets={"default": 120, "gpt": 200},
                        analysis_lanes=("summary",),
                    ),
                )
            },
            default_preset="ci",
        )
        run_config = spec.make_run_config()
        self.assertEqual(run_config.preset_name, "ci")
        self.assertEqual(run_config.analysis_lanes, ("summary",))
        self.assertEqual(run_config.token_budget_defaults["gpt"], 200)

    def test_register_benchmark_spec_rejects_runner_without_contract(self):
        spec = build_benchmark_spec(
            name="toy_invalid_runner",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        with self.assertRaisesRegex(TypeError, "BenchmarkRunner contract"):
            register_benchmark_spec(spec, runner=object())

    def test_register_benchmark_spec_rejects_runner_for_different_spec(self):
        spec = build_benchmark_spec(
            name="toy_registration_target",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        other_spec = build_benchmark_spec(
            name="toy_registration_other",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        with self.assertRaisesRegex(ValueError, "does not match registration target"):
            register_benchmark_spec(spec, runner=SharedBenchmarkRunner(other_spec))

    def test_benchmark_list_contains_constitutional(self):
        self.assertIn("constitutional_compliance", list_benchmark_specs())


if __name__ == "__main__":
    unittest.main()
