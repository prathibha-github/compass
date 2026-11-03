import json
from types import SimpleNamespace

from compass.benchmark.reporting import analyze_results, format_summary
from compass.detectors import TicSuite, StylePrompt, StyleCondition, RegexDetector
from compass.evaluation.suite_runner import generate_suite_completions, evaluate_suite_completions, summarize_suite_evaluations


def test_benchmark_all_filtered_is_unavailable_and_unknown_quality_counted(tmp_path):
    path = tmp_path/'evaluations.jsonl'
    row = dict(model='m', rubric='r', prompt_id='p', sample_idx=0, score=0, hit=False)
    path.write_text(json.dumps(row)+'\n')
    result = analyze_results(path, tmp_path, quality_filter_mode='exclude_flagged')
    cell = result['m|r']
    assert cell.hit_rate is None and cell.mean_score is None
    assert cell.total == 0 and cell.raw_total == 1
    assert cell.prompt_statistics['quality_unknown'] == 1
    assert cell.prompt_statistics['all']['hit_rate'] == 0
    assert 'n/a' in format_summary(result, path)


def test_suite_statistics_keep_prompt_and_response_counts_separate(tmp_path):
    suite = TicSuite('s', [StylePrompt('p1','Task'), StylePrompt('p2','Task2')],
                     [StyleCondition('c','')], [RegexDetector('r','rest')], 'c')
    class Fake:
        def complete(self, **kwargs):
            return SimpleNamespace(completion='Rest now.', tokens_used={}, cost_usd=0)
    generations = generate_suite_completions(suite, ['m'], 3, tmp_path, client_factory=lambda _: Fake())
    scored = evaluate_suite_completions(suite, generations, tmp_path)
    summary = summarize_suite_evaluations(scored, suite)['m']['c']['prompt_statistics']['r']
    assert summary['all']['responses'] == 6
    assert summary['all']['distinct_prompts'] == 2
    assert summary['all']['hit_rate'] == 1
    pooled = summarize_suite_evaluations(scored, suite, pool_models=True)
    assert pooled['c']['prompt_statistics_by_model']['m']['r'] == summary
