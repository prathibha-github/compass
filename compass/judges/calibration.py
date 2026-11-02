"""Compare binary detector predictions with independently supplied human labels."""

from compass.judges.reliability import wilson_interval


def compare_human_labels(predictions: dict, labels: dict) -> dict:
    """Return coverage and a confusion matrix keyed by stable item IDs.

    Predictions must be booleans. Human labels may be booleans or None (not
    labeled/undecidable). Missing labels never count as negatives. Sampling and
    adjudication are caller responsibilities; these metrics are descriptive.
    """
    if not predictions:
        raise ValueError("No predictions to calibrate")
    if set(labels) - set(predictions):
        raise ValueError("Human labels contain unknown item IDs")
    if any(type(value) is not bool for value in predictions.values()):
        raise ValueError("Predictions must be booleans")
    if any(value is not None and type(value) is not bool for value in labels.values()):
        raise ValueError("Human labels must be booleans or null")
    pairs = [(pred, labels[key]) for key, pred in predictions.items() if labels.get(key) is not None]
    tp = sum(pred and human for pred, human in pairs)
    tn = sum(not pred and not human for pred, human in pairs)
    fp = sum(pred and not human for pred, human in pairs)
    fn = sum(not pred and human for pred, human in pairs)
    n = len(pairs)
    return {
        "status": "complete" if n == len(predictions) else "incomplete",
        "total": len(predictions), "labeled": n, "unlabeled": len(predictions) - n,
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
        "agreement": (tp + tn) / n if n else None,
        "agreement_wilson_95": list(wilson_interval(tp + tn, n)) if n else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
