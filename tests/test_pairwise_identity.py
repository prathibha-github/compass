import itertools
import pytest
from compass.comparison.pairwise import PairwiseRanker


def test_order_invariant_and_minimum_applies_across_keys():
    rows = [('a', 0, .2), ('a', 1, .8), ('b', 0, .3), ('b', 1, .9)]
    for order in itertools.permutations(rows):
        ranker = PairwiseRanker()
        for model, sample, score in order:
            ranker.add_record('s', model, ('p', 'c', sample), score)
        pair = ranker.rank('s')['pairwise_results'][('a', 'b')]
        assert pair['matches'] == pair['wins_a'] == 2


def test_unmatched_samples_and_ties():
    ranker = PairwiseRanker()
    for model, sample in [('a',0), ('a',1), ('a',2), ('b',0), ('b',1), ('b',3)]:
        ranker.add_record('s', model, ('p','c',sample), .5, {'task_type':'task'})
    pair = ranker.rank('s')['pairwise_results'][('a','b')]
    assert pair['ties'] == pair['matches'] == 2
    assert pair['wins_a'] == pair['wins_b'] == 0
    assert pair['unmatched_a'] == pair['unmatched_b'] == 1
    assert ranker.rank('s')['overall_ranking'] == [('a',1,2), ('b',1,2)]
    assert ranker.rank_by_segment('s')['task']['pairwise_results'][('a','b')] == pair
    with pytest.raises(ValueError, match='Duplicate'):
        ranker.add_record('s','a',('p','c',0),.9)


def test_benchmark_adapter_preserves_sample_indices(tmp_path):
    import json
    from compass.benchmark.reporting import rank_models
    from compass.benchmark.specs import BenchmarkSpec, BenchmarkPrompt
    from compass import RubricLibrary
    spec=BenchmarkSpec('b','1',{'clarity':(BenchmarkPrompt('p','Task','task'),)},{'clarity':RubricLibrary.clarity})
    rows=[dict(model=m,rubric='clarity',prompt_id='p',sample_idx=i,task_type='task',score=s,hit=False)
          for m,i,s in [('a',0,.2),('a',1,.8),('b',1,.9),('b',0,.3)]]
    path=tmp_path/'scores.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    pair=rank_models(path,spec,tmp_path).rubrics['clarity'].pairwise_results[('a','b')]
    assert pair.wins_a == pair.matches == 2


def test_invalid_scores_and_conflicting_segments():
    ranker=PairwiseRanker()
    for score in [float('nan'),float('inf'),True,'0.2']:
        with pytest.raises(ValueError):
            ranker.add_record('s','a',('p',0),score)
    ranker.add_record('s','a',('p',0),.2,{'task_type':'a'})
    ranker.add_record('s','b',('p',0),.3,{'task_type':'b'})
    with pytest.raises(ValueError,match='Conflicting'):
        ranker.rank_by_segment('s')
