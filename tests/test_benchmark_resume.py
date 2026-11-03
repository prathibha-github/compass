import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from compass.benchmark.runner import SharedBenchmarkRunner, generate_completions, evaluate_completions
from compass.benchmark.specs import BenchmarkPrompt, BenchmarkSpec
from compass.rubrics import RubricLibrary


def spec():
    return BenchmarkSpec('test', '1', {'clarity': (BenchmarkPrompt('p1', 'Original task', 'general'),)},
                         {'clarity': RubricLibrary.get('clarity')})


def client():
    return Mock(complete=Mock(return_value=SimpleNamespace(
        completion='{"score": 0.2, "confidence": 0.9, "rationale": "ok"}',
        tokens_used={'input': 3, 'output': 3}, cost_usd=0.01)))


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_changed_prompt_and_budget_preserve_artifacts(tmp_path):
    original = spec()
    runner = SharedBenchmarkRunner(original)
    cfg = runner.validate_run_config(original.make_run_config(output_dir=str(tmp_path)))
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        runner.generate(cfg)
    before = snapshot(tmp_path)
    changed = replace(original, prompts_by_rubric={'clarity': (BenchmarkPrompt('p1', 'Changed task', 'general'),)})
    for definition, budget in [(changed, None), (original, {'llama3.1': 99})]:
        runner = SharedBenchmarkRunner(definition)
        cfg = runner.validate_run_config(definition.make_run_config(output_dir=str(tmp_path), max_tokens_by_model=budget))
        with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='Inputs changed'):
            runner.generate(cfg)
        factory.assert_not_called()
        assert snapshot(tmp_path) == before


def test_resume_partial_generation_and_reject_incomplete_scoring(tmp_path):
    fake = client()
    fake.complete.side_effect = [fake.complete.return_value, RuntimeError('offline')]
    with patch('compass.benchmark.runner._create_client', return_value=fake):
        generate_completions(['llama3.1'], spec(), 2, tmp_path)
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='incomplete'):
        evaluate_completions(tmp_path/'generations.jsonl', spec(), 'judge', tmp_path)
    factory.assert_not_called()
    fake = client()
    with patch('compass.benchmark.runner._create_client', return_value=fake):
        generate_completions(['llama3.1'], spec(), 2, tmp_path)
        generate_completions(['llama3.1'], spec(), 2, tmp_path)
    assert fake.complete.call_count == 1


def test_changed_rubric_requires_new_scores_but_reuses_generations(tmp_path):
    original = spec()
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = generate_completions(['llama3.1'], original, 1, tmp_path).path
        evaluate_completions(generation, original, 'judge', tmp_path)
    before = snapshot(tmp_path)
    changed = replace(original, rubrics_by_name={'clarity': replace(original.rubrics_by_name['clarity'], text='New rubric')})
    with patch('compass.benchmark.runner._create_client') as factory:
        generate_completions(['llama3.1'], changed, 1, tmp_path)
        with pytest.raises(ValueError, match='Inputs changed'):
            evaluate_completions(generation, changed, 'judge', tmp_path)
    factory.assert_not_called()
    assert snapshot(tmp_path) == before
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        result = evaluate_completions(generation, changed, 'judge', tmp_path/'rescore')
    assert result.successful == 1
    assert generation.read_bytes() == before['generations.jsonl']


def test_changed_completion_invalidates_scores(tmp_path):
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = generate_completions(['llama3.1'], spec(), 1, tmp_path).path
        evaluate_completions(generation, spec(), 'judge', tmp_path)
    row = json.loads(generation.read_text())
    row['completion'] = 'Edited completion'
    generation.write_text(json.dumps(row)+'\n')
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='Inputs changed'):
        evaluate_completions(generation, spec(), 'judge', tmp_path)
    factory.assert_not_called()


def test_legacy_resume_and_duplicate_generations_rejected(tmp_path):
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = generate_completions(['llama3.1'], spec(), 1, tmp_path).path
    generation.with_suffix('.manifest.json').unlink()
    with pytest.raises(ValueError, match='no input manifest'):
        generate_completions(['llama3.1'], spec(), 1, tmp_path)
    generation.write_text(generation.read_text()*2)
    with pytest.raises(ValueError, match='duplicate generation'):
        evaluate_completions(generation, spec(), 'judge', tmp_path)


def test_public_evaluate_rejects_wrong_population_and_namespace(tmp_path):
    definition = spec()
    runner = SharedBenchmarkRunner(definition)
    cfg = runner.validate_run_config(definition.make_run_config(output_dir=str(tmp_path)))
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = runner.generate(cfg)
        runner.evaluate(generation, cfg)
    before = snapshot(tmp_path)
    for models, samples in [(['other'], 1), (['llama3.1'], 5)]:
        wrong = runner.validate_run_config(definition.make_run_config(output_dir=str(tmp_path), models=models, samples=samples))
        with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='do not match'):
            runner.evaluate(generation, wrong)
        factory.assert_not_called()
        assert snapshot(tmp_path) == before
    changed = replace(definition, name='other', version='2')
    other = SharedBenchmarkRunner(changed)
    config = other.validate_run_config(changed.make_run_config(output_dir=str(tmp_path)))
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='Inputs changed'):
        other.evaluate(generation, config)
    factory.assert_not_called()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('corruption', ['duplicate', 'json', 'manifest'])
def test_invalid_artifacts_do_not_overwrite_public_run_metadata(tmp_path, corruption):
    definition = spec()
    runner = SharedBenchmarkRunner(definition)
    config = runner.validate_run_config(definition.make_run_config(output_dir=str(tmp_path)))
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = runner.generate(config)
    if corruption == 'duplicate':
        generation.write_text(generation.read_text()*2)
    elif corruption == 'json':
        generation.write_text('{bad json\n')
    else:
        generation.with_suffix('.manifest.json').write_text('{bad json')
    before = snapshot(tmp_path)
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError):
        runner.generate(config)
    factory.assert_not_called()
    assert snapshot(tmp_path) == before


def test_new_generations_cannot_be_scored_without_manifest(tmp_path):
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = generate_completions(['llama3.1'], spec(), 1, tmp_path).path
    generation.with_suffix('.manifest.json').unlink()
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='manifest is missing'):
        evaluate_completions(generation, spec(), 'judge', tmp_path)
    factory.assert_not_called()


def test_archived_generation_scores_record_unknown_coverage(tmp_path):
    generation = tmp_path/'legacy.jsonl'
    generation.write_text(json.dumps(dict(model='m', rubric='clarity', prompt_id='p1',
                                         task_type='general', completion='old reply'))+'\n')
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        evaluation = evaluate_completions(generation, spec(), 'judge', tmp_path).path
    assert json.loads(evaluation.read_text())['generation_coverage_verified'] is False


@pytest.mark.parametrize('change', ['prompt', 'task', 'budget'])
def test_public_evaluate_rejects_changed_generation_inputs(tmp_path, change):
    definition = spec()
    runner = SharedBenchmarkRunner(definition)
    config = runner.validate_run_config(definition.make_run_config(output_dir=str(tmp_path)))
    with patch('compass.benchmark.runner._create_client', return_value=client()):
        generation = runner.generate(config)
        runner.evaluate(generation, config)
    before = snapshot(tmp_path)
    prompt = definition.prompts_by_rubric['clarity'][0]
    if change in ('prompt', 'task'):
        prompt = replace(prompt, **({'text': 'New task'} if change == 'prompt' else {'task_type': 'changed'}))
        definition = replace(definition, prompts_by_rubric={'clarity': (prompt,)})
    runner = SharedBenchmarkRunner(definition)
    config = runner.validate_run_config(definition.make_run_config(
        output_dir=str(tmp_path), max_tokens_by_model={'llama3.1': 99} if change == 'budget' else None))
    with patch('compass.benchmark.runner._create_client') as factory, pytest.raises(ValueError, match='Inputs changed'):
        runner.evaluate(generation, config)
    factory.assert_not_called()
    assert snapshot(tmp_path) == before
