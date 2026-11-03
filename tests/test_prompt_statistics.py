import pytest
from compass.evaluation.statistics import prompt_rate_summary


def rows():
    return [dict(prompt_id='p1', sample_idx=i, hit=True, generation_quality_flagged=False) for i in range(9)] + [
        dict(prompt_id='p2', sample_idx=0, hit=False, generation_quality_flagged=True)]


def test_response_weighting_and_prompt_cluster_bootstrap():
    result = prompt_rate_summary(rows(), resamples=500, seed=4)
    assert result['all']['hit_rate'] == .9
    assert result['all']['distinct_prompts'] == 2
    assert result['all']['responses'] == 10
    # Drawing whole prompts can select only the single negative observation.
    assert result['all']['interval'] == [0, 1]
    assert result == prompt_rate_summary(list(reversed(rows())), resamples=500, seed=4)
    assert result['quality_filtered']['hit_rate'] == 1
    assert result['quality_filtered']['interval'] is None
    assert result['quality_flagged'] == 1


def test_unknown_quality_and_empty_filtered_population():
    result = prompt_rate_summary([dict(prompt_id='p', hit=False)])
    assert result['quality_unknown'] == 1
    assert result['quality_filtered']['responses'] == 0
    assert result['quality_filtered']['hit_rate'] is None
    assert result['all']['interval'] is None
    assert prompt_rate_summary([])['all']['hit_rate'] is None


def test_duplicate_observations_and_accidental_model_pooling_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        prompt_rate_summary(rows() * 2)
    with pytest.raises(ValueError, match='model'):
        prompt_rate_summary([dict(prompt_id='p', hit=True, model='a'), dict(prompt_id='q', hit=True, model='b')])


@pytest.mark.parametrize('confidence,resamples', [(0, 2000), (1, 2000), (.95, 0), (float('nan'), 2000)])
def test_invalid_bootstrap_settings(confidence, resamples):
    with pytest.raises(ValueError):
        prompt_rate_summary(rows(), confidence=confidence, resamples=resamples)
