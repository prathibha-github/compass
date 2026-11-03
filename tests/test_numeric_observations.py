import pytest
from compass.benchmark.schemas import migrate_evaluation_record
from compass.judges.reliability import JudgeReliabilityAuditor
from compass.detectors import compute_uplift, compute_absolute_change


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf'), True, '0.5', None, -0.1, 1.1, 10**1000])
def test_invalid_observations_are_rejected(score):
    with pytest.raises(ValueError, match='finite number'):
        migrate_evaluation_record(dict(model='m', rubric='r', prompt_id='p', score=score, hit=False))
    auditor = JudgeReliabilityAuditor()
    with pytest.raises(ValueError, match="finite number"):
        auditor.detect_drift([score])
    with pytest.raises(ValueError, match="finite number"):
        auditor.add_comparison(0.5, score)
    assert auditor.observations == []
    with pytest.raises(ValueError, match='finite number'):
        auditor.calculate_agreement([score], [score])
    with pytest.raises(ValueError, match='hit_threshold'):
        auditor.calculate_agreement([0.5], [0.5], hit_threshold=score)


def test_stored_hit_must_be_boolean():
    with pytest.raises(ValueError, match='boolean'):
        migrate_evaluation_record(dict(model='m', rubric='r', prompt_id='p', score=0.5, hit='false'))


def test_zero_baseline_has_absolute_but_no_relative_change():
    rows = {key: {'detectors': {'d': {'pct_hit': rate}}}
            for key, rate in [('baseline',0), ('small',1), ('large',100)]}
    assert compute_uplift(rows, 'd', 'pct_hit') == dict(baseline=None, small=None, large=None)
    assert compute_absolute_change(rows, 'd', 'pct_hit') == dict(baseline=0, small=1, large=100)


def test_invalid_saved_score_cannot_be_reported(tmp_path):
    import json
    from compass.benchmark.reporting import analyze_results
    from compass.benchmark.validation import validate_benchmark_report
    path = tmp_path / 'evaluations.jsonl'
    path.write_text(json.dumps(dict(model='m', rubric='r', prompt_id='p', score=float('nan'), hit=False)) + '\n')
    for report in (lambda: analyze_results(path, tmp_path), lambda: validate_benchmark_report(path)):
        with pytest.raises(ValueError, match='finite number'):
            report()


@pytest.mark.parametrize("expected_low", [None, 0, 1, "false"])
def test_control_direction_is_boolean(expected_low):
    with pytest.raises(ValueError, match="expected_low"):
        JudgeReliabilityAuditor().detect_drift([], expected_low=expected_low)
