import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from compass.detectors.base import TicSuite, StylePrompt, StyleCondition
from compass.detectors.heuristic import RegexDetector
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions, summarize_suite_evaluations
from compass.benchmark.runner import generate_completions, evaluate_completions
from compass.benchmark.specs import BenchmarkSpec, BenchmarkPrompt
from compass.benchmark.reporting import analyze_results, rank_models
from compass.benchmark.validation import validate_benchmark_report
from compass import RubricLibrary


def suite():
    return TicSuite('s',[StylePrompt('p','Task')],[StyleCondition('c','')],[RegexDetector('r','rest')],'c')


def response(text='rest'):
    return SimpleNamespace(completion=text,tokens_used={},cost_usd=0)


@pytest.mark.parametrize('successes',[0,1])
def test_cannot_drop_failed_planned_generations(tmp_path, successes):
    fake=Mock(complete=Mock(side_effect=([response()] if successes else [])+[RuntimeError('offline')]*2))
    with pytest.raises(RuntimeError):
        generate_suite_completions(suite(),['m'],2,tmp_path,client_factory=lambda _:fake)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    factory=Mock()
    with pytest.raises(ValueError,match='Planned'):
        generate_suite_completions(suite(),['m'],1,tmp_path,resume=True,client_factory=factory)
    factory.assert_not_called()
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_modern_missing_manifest_and_duplicate_manifest_fail(tmp_path):
    path=generate_suite_completions(suite(),['m'],1,tmp_path,client_factory=lambda _:Mock(complete=Mock(return_value=response())))
    manifest=path.with_suffix('.manifest.json')
    data=json.loads(manifest.read_text())
    data['expected']*=2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Duplicate manifest'):
        evaluate_suite_completions(suite(),path,tmp_path)
    manifest.unlink()
    with pytest.raises(ValueError,match="manifest is missing"):
        generate_suite_completions(suite(),["m"],1,tmp_path,resume=True)
    with pytest.raises(ValueError,match='manifest is missing'):
        evaluate_suite_completions(suite(),path,tmp_path)


def test_legacy_suite_summary_exposes_unknown_generation_coverage(tmp_path):
    path=tmp_path/'legacy.jsonl'
    path.write_text(json.dumps(dict(model='m',suite='s',prompt_id='p',condition='c',sample_idx=0,completion='rest'))+'\n')
    scored=evaluate_suite_completions(suite(),path,tmp_path)
    assert summarize_suite_evaluations(scored,suite())['m']['c']['generation_coverage_verified'] is False


def test_incomplete_benchmark_scores_cannot_be_reported(tmp_path):
    spec=BenchmarkSpec('b','1',{'clarity':(BenchmarkPrompt('p','Task','task'),)}, {'clarity':RubricLibrary.clarity})
    fake=Mock(complete=Mock(return_value=response()))
    with patch('compass.benchmark.runner._create_client',return_value=fake):
        generations=generate_completions(['m'],spec,2,tmp_path).path
        fake.complete.side_effect=[response('{"score":0}'),response('invalid')]
        # Different saved responses ensure the second score cannot be a cache hit.
        rows=[json.loads(line) for line in generations.read_text().splitlines()]
        rows[1]['completion']='another response'
        generations.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        evaluations=evaluate_completions(generations,spec,'judge',tmp_path).path
    assert validate_benchmark_report(evaluations)[0].code=='incomplete_evaluation_coverage'
    with pytest.raises(ValueError,match='Incomplete'):
        analyze_results(evaluations,tmp_path)
    with pytest.raises(ValueError,match='Incomplete'):
        rank_models(evaluations,spec,tmp_path)


def test_missing_evaluation_manifest_cannot_hide_failed_detector(tmp_path):
    planned = suite()
    planned.detectors.append(RegexDetector("broken", "["))
    generations = generate_suite_completions(planned, ["m"], 1, tmp_path,
        client_factory=lambda _: Mock(complete=Mock(return_value=response())))
    with pytest.raises(RuntimeError):
        evaluate_suite_completions(planned, generations, tmp_path)
    evaluations = tmp_path / "suite_evaluations_heuristic.jsonl"
    evaluations.with_suffix(".manifest.json").unlink()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="Evaluation manifest is missing"):
        evaluate_suite_completions(suite(), generations, tmp_path, resume=True)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    # An intentional rescore can change the detector population.
    evaluate_suite_completions(suite(), generations, tmp_path, resume=False)
