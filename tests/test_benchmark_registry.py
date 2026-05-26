"""Contract tests for benchmark specs and registry."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
        run_config = runner.validate_run_config(spec.make_run_config())
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
        run_config = runner.validate_run_config(spec.make_run_config())
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
        run_config = runner.validate_run_config(spec.make_run_config())
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
        run_config = runner.validate_run_config(spec.make_run_config())
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
        validate_report_artifacts.assert_called_once_with(evaluations_path)

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
        run_config = runner.validate_run_config(spec.make_run_config())
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

    def test_shared_runner_phase_methods_require_validated_run_config(self):
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
        with self.assertRaisesRegex(
            ValueError,
            "must be validated with runner.validate_run_config",
        ):
            runner.validate_report(Path("evaluations.jsonl"), run_config)

    def test_shared_runner_evaluate_applies_custom_legacy_token_cap_threshold(self):
        spec = build_benchmark_spec(
            name="toy_legacy_threshold",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)

        class _FakeJudge:
            def __init__(self, config, client, cache):
                self.config = config

            def evaluate(self, text):
                return SimpleNamespace(
                    score=0.4,
                    hit=False,
                    confidence=0.8,
                    rationale="legacy-threshold",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generations_path = output_dir / "generations.jsonl"
            generations_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": spec.name,
                        "benchmark_version": spec.version,
                        "benchmark_schema_version": 1,
                        "benchmark_record_type": "generation",
                        "model": "llama3.1",
                        "rubric": "clarity",
                        "prompt_id": "p1",
                        "task_type": "explanation",
                        "sample_idx": 0,
                        "completion": "At its core, a",
                        "tokens_used": {"input": 1, "output": 301},
                    }
                )
                + "\n"
            )
            run_config = spec.make_run_config(
                output_dir=str(output_dir),
                legacy_token_cap_threshold=301,
            )
            run_config = runner.validate_run_config(run_config)
            with patch(
                "compass.benchmark.runner.LLMJudge",
                side_effect=_FakeJudge,
            ), patch(
                "compass.benchmark.runner.OllamaClient",
                return_value=SimpleNamespace(),
            ):
                evaluations_path = runner.evaluate(generations_path, run_config)

            saved_rows = [
                json.loads(line)
                for line in evaluations_path.read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(len(saved_rows), 1)
        self.assertTrue(saved_rows[0]["generation_hit_token_cap"])
        self.assertTrue(saved_rows[0]["generation_token_cap_inferred_legacy"])
        self.assertTrue(saved_rows[0]["generation_quality_flagged"])

    def test_shared_runner_generate_writes_run_policy_sidecar(self):
        spec = build_benchmark_spec(
            name="toy_policy_artifact",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_config = runner.validate_run_config(
                spec.make_run_config(
                    output_dir=str(output_dir),
                    models=("llama3.1", "mistral"),
                    max_tokens_by_model={"llama3.1": 321, "mistral": 222},
                    allow_mixed_token_budgets=True,
                    skip_ranking=True,
                )
            )
            generated_path = output_dir / "generations.jsonl"
            with patch(
                "compass.benchmark.runner.generate_completions",
                return_value=generated_path,
            ):
                result = runner.generate(run_config)

            policy_path = output_dir / "benchmark_run_policy.json"
            policy = json.loads(policy_path.read_text())

        self.assertEqual(result, generated_path)
        self.assertEqual(policy["benchmark_name"], spec.name)
        self.assertEqual(policy["benchmark_version"], spec.version)
        self.assertEqual(policy["preset_name"], run_config.preset_name)
        self.assertEqual(policy["quality_filter_mode"], run_config.quality_filter_mode)
        self.assertEqual(policy["analysis_lanes"], list(run_config.analysis_lanes))
        self.assertEqual(
            policy["effective_analysis_lanes"],
            list(run_config.effective_analysis_lanes),
        )
        self.assertEqual(
            policy["effective_max_tokens_by_model"],
            {
                "llama3.1": 321,
                "mistral": 222,
            },
        )

    def test_shared_runner_evaluate_writes_run_policy_sidecar(self):
        spec = build_benchmark_spec(
            name="toy_policy_eval",
            version="0.1",
            prompts_by_rubric={
                "clarity": [
                    {"id": "p1", "text": "Explain X", "task_type": "explanation"},
                ]
            },
            rubrics_by_name={"clarity": RubricLibrary.clarity},
        )
        runner = SharedBenchmarkRunner(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_config = runner.validate_run_config(
                spec.make_run_config(output_dir=str(output_dir))
            )
            evaluations_path = output_dir / "evaluations.jsonl"
            with patch(
                "compass.benchmark.runner.evaluate_completions",
                return_value=evaluations_path,
            ), patch.object(
                runner,
                "_validate_report_artifacts",
                return_value=[],
            ):
                result = runner.evaluate(output_dir / "generations.jsonl", run_config)

            policy_path = output_dir / "benchmark_run_policy.json"
            policy = json.loads(policy_path.read_text())

        self.assertEqual(result, evaluations_path)
        self.assertEqual(policy["judge_model"], run_config.judge_model)
        self.assertEqual(
            policy["legacy_token_cap_threshold"],
            run_config.legacy_token_cap_threshold,
        )

    def test_benchmark_list_contains_constitutional(self):
        self.assertIn("constitutional_compliance", list_benchmark_specs())


if __name__ == "__main__":
    unittest.main()
