"""Reject malformed numeric results at the report boundary."""
import pytest

def test_invalid_saved_score_cannot_be_reported(tmp_path):
    import json
    from compass.benchmark.reporting import analyze_results
    from compass.benchmark.validation import validate_benchmark_report
    path = tmp_path / 'evaluations.jsonl'
    path.write_text(json.dumps(dict(model='m', rubric='r', prompt_id='p', score=float('nan'), hit=False)) + '\n')
    for report in (lambda: analyze_results(path, tmp_path), lambda: validate_benchmark_report(path)):
        with pytest.raises(ValueError, match='finite number'):
            report()

