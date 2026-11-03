"""Response-weighted rates with uncertainty clustered by prompt."""

import math
import random
from collections import defaultdict


def _clusters(rows):
    clusters = defaultdict(list)
    seen = set()
    for row in rows:
        prompt = row.get('prompt_id')
        sample = row.get('sample_idx', 0)
        if not isinstance(prompt, str) or not prompt or type(sample) is not int or sample < 0:
            raise ValueError('Each observation needs a prompt_id and nonnegative sample_idx')
        if type(row.get('hit')) is not bool:
            raise ValueError('Hits must be booleans')
        identity = (row.get('suite', ''), row.get('rubric', ''), prompt, sample)
        if identity in seen:
            raise ValueError(f'Duplicate observation: {identity}')
        seen.add(identity)
        clusters[identity[:3]].append(row)
    return [sorted(clusters[key], key=lambda r: r.get('sample_idx', 0)) for key in sorted(clusters)]


def _rate(clusters):
    rows = [row for cluster in clusters for row in cluster]
    return sum(row['hit'] for row in rows) / len(rows) if rows else None


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _validate(confidence, resamples):
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between zero and one')
    if type(resamples) is not int or resamples < 100:
        raise ValueError('resamples must be an integer of at least 100')


def prompt_rate_summary(rows, *, confidence=.95, resamples=2000, seed=0):
    """Summarize one model/condition/detector cell, resampling whole prompts.

    Rates weight each response equally. The bootstrap describes variation over
    the observed prompt set; it does not correct prompt-selection bias. Unknown
    quality flags are excluded from the quality-filtered sensitivity analysis.
    """
    _validate(confidence, resamples)
    rows = list(rows)
    for field in ('model', 'condition', 'detector', 'suite', 'rubric'):
        if len({row.get(field) for row in rows}) > 1:
            raise ValueError(f'Pass one {field} group at a time')
    all_clusters = _clusters(rows)
    known_clean = [row for row in rows if row.get('generation_quality_flagged') is False]
    clean_clusters = _clusters(known_clean)

    def summarize(clusters):
        rng = random.Random(seed)
        count = len(clusters)
        interval = None
        if count >= 2:
            estimates = [_rate([clusters[rng.randrange(count)] for _ in range(count)])
                         for _ in range(resamples)]
            tail = (1 - confidence) / 2
            interval = [_quantile(estimates, tail), _quantile(estimates, 1 - tail)]
        return {'responses': sum(len(c) for c in clusters), 'distinct_prompts': count,
                'hit_rate': _rate(clusters), 'interval': interval}

    return {
        'all': summarize(all_clusters), 'quality_filtered': summarize(clean_clusters),
        'quality_flagged': sum(row.get('generation_quality_flagged') is True for row in rows),
        'quality_unknown': sum(type(row.get('generation_quality_flagged')) is not bool for row in rows),
        'method': 'prompt_cluster_percentile_bootstrap', 'weighting': 'response',
        'confidence': confidence, 'resamples': resamples, 'seed': seed,
        'interval_unavailable_below_prompts': 2,
    }
