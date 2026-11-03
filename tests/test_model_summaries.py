from types import SimpleNamespace

from compass.detectors.base import StyleCondition, StylePrompt, TicSuite
from compass.detectors.heuristic import PhraseSetDetector
from compass.evaluation.suite_runner import (
    generate_suite_completions, evaluate_suite_completions, summarize_suite_evaluations,
)


def test_models_remain_separate_with_unequal_sample_counts(tmp_path):
    suite = TicSuite(name='test', prompts=[StylePrompt('same-id', 'Task', 'task')],
                     conditions=[StyleCondition('neutral', '')],
                     detectors=[PhraseSetDetector('rest', phrases={'take a break'})],
                     baseline_condition='neutral')
    def factory(model):
        return SimpleNamespace(complete=lambda **kwargs: SimpleNamespace(
            completion='take a break' if model == 'a' else 'keep working',
            tokens_used={}, cost_usd=0))
    path = generate_suite_completions(suite, ['a'], 1, tmp_path, client_factory=factory)
    path = generate_suite_completions(suite, ['a', 'b'], 1, tmp_path, resume=True, client_factory=factory)
    # Add a second response for b to exercise unequal model weights.
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    extra = dict(next(row for row in rows if row['model'] == 'b'), sample_idx=1)
    with path.open('a') as f:
        f.write(json.dumps(extra)+'\n')
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    item = next(item for item in manifest['expected'] if item['identity'][0] == 'b')
    new_item = dict(item, identity=[*item['identity'][:-1], 1])
    manifest['expected'].append(new_item)
    manifest_path.write_text(json.dumps(manifest))
    evaluations = evaluate_suite_completions(suite, path, tmp_path)
    summary = summarize_suite_evaluations(evaluations, suite)
    assert set(summary) == {'a', 'b'}
    assert summary['a']['neutral']['detectors']['rest']['pct_hit'] == 100
    assert summary['b']['neutral']['detectors']['rest']['pct_hit'] == 0
    assert summary['a']['neutral']['n_outputs'] == 1
    assert summary['b']['neutral']['n_outputs'] == 2
    pooled = summarize_suite_evaluations(evaluations, suite, pool_models=True)
    assert pooled['neutral']['detectors']['rest']['pct_hit'] == 33.3
