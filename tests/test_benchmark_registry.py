"""Contract tests for benchmark specs and registry."""

from pathlib import Path
import unittest
from unittest.mock import patch

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

    def test_shared_runner_skips_summary_when_lane_disabled(self):
        spec = build_benchmark_spec(
            name="toy_pairwise_only",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
            run_presets={
                "default": BenchmarkRunPreset(
                    models=("llama3.1",),
                    samples=1,
                    judge_model="llama3.1",
                    output_dir="results/toy_pairwise_only",
                    policy=BenchmarkPolicyDefaults(analysis_lanes=("pairwise",)),
                )
            },
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        with patch("compass.benchmark.runner.analyze_results") as analyze:
            self.assertEqual(runner.analyze(Path("evaluations.jsonl"), run_config), {})
        analyze.assert_not_called()

    def test_shared_runner_skips_pairwise_when_lane_disabled(self):
        spec = build_benchmark_spec(
            name="toy_summary_only",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
            run_presets={
                "default": BenchmarkRunPreset(
                    models=("llama3.1",),
                    samples=1,
                    judge_model="llama3.1",
                    output_dir="results/toy_summary_only",
                    policy=BenchmarkPolicyDefaults(analysis_lanes=("summary",)),
                )
            },
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        with patch("compass.benchmark.runner.rank_models") as rank:
            runner.rank(Path("evaluations.jsonl"), run_config)
        rank.assert_not_called()

    def test_shared_runner_forwards_quality_filter_mode(self):
        spec = build_benchmark_spec(
            name="toy_quality_policy",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
            run_presets={
                "default": BenchmarkRunPreset(
                    models=("llama3.1",),
                    samples=1,
                    judge_model="llama3.1",
                    output_dir="results/toy_quality_policy",
                    policy=BenchmarkPolicyDefaults(
                        quality_filter_mode="exclude_flagged",
                    ),
                )
            },
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        with patch("compass.benchmark.runner.analyze_results", return_value={}) as analyze:
            runner.analyze(Path("evaluations.jsonl"), run_config)
        with patch("compass.benchmark.runner.rank_models") as rank:
            runner.rank(Path("evaluations.jsonl"), run_config)

        self.assertEqual(analyze.call_args.kwargs["quality_filter_mode"], "exclude_flagged")
        self.assertEqual(rank.call_args.kwargs["quality_filter_mode"], "exclude_flagged")

    def test_shared_runner_evaluate_runs_validation_hook(self):
        spec = build_benchmark_spec(
            name="toy_validate_hook",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        evaluations_path = Path("evaluations.jsonl")
        with patch(
            "compass.benchmark.runner.evaluate_completions",
            return_value=evaluations_path,
        ) as evaluate:
            with patch.object(
                runner,
                "_validate_report_artifacts",
                return_value=[],
            ) as validate_report_artifacts:
                self.assertEqual(
                    runner.evaluate(Path("generations.jsonl"), run_config),
                    evaluations_path,
                )

        evaluate.assert_called_once()
        validate_report_artifacts.assert_called_once_with(evaluations_path, run_config)

    def test_shared_runner_evaluate_raises_on_invalid_report(self):
        spec = build_benchmark_spec(
            name="toy_invalid_report",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        with patch(
            "compass.benchmark.runner.evaluate_completions",
            return_value=Path("evaluations.jsonl"),
        ):
            with patch.object(
                runner,
                "_validate_report_artifacts",
                return_value=["evaluation row 1 missing quality fields: generation_visible_chars"],
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Benchmark report validation failed",
                ):
                    runner.evaluate(Path("generations.jsonl"), run_config)

    def test_shared_runner_validate_report_validates_run_config_once(self):
        spec = build_benchmark_spec(
            name="toy_validate_once",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)
        run_config = spec.make_run_config()
        with patch.object(
            runner,
            "validate_run_config",
            wraps=runner.validate_run_config,
        ) as validate_run_config:
            with patch(
                "compass.benchmark.runner.validate_benchmark_report",
                return_value=[],
            ):
                runner.validate_report(Path("evaluations.jsonl"), run_config)

        validate_run_config.assert_called_once_with(run_config)

    def test_benchmark_list_contains_constitutional(self):
        self.assertIn("constitutional_compliance", list_benchmark_specs())


if __name__ == "__main__":
    unittest.main()
