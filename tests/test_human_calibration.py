import pytest

from compass.judges.calibration import compare_human_labels


def test_confusion_matrix_and_unlabeled_coverage():
    result = compare_human_labels(
        dict(a=True, b=True, c=False, d=False, e=True, f=False),
        dict(a=True, b=False, c=True, d=False, e=None),
    )
    assert result["status"] == "incomplete"
    assert result["labeled"] == 4 and result["unlabeled"] == 2
    assert [result[k] for k in ("true_positive", "false_positive", "false_negative", "true_negative")] == [1, 1, 1, 1]
    assert result["agreement"] == result["precision"] == result["recall"] == 0.5


def test_unlabeled_data_never_reports_perfect_agreement():
    result = compare_human_labels({"a": False}, {})
    assert result["agreement"] is None
    assert result["precision"] is None
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("predictions,labels", [({}, {}), ({"a": True}, {"b": False}),
                                                ({"a": 1}, {}), ({"a": True}, {"a": "false"})])
def test_rejects_ambiguous_or_misaligned_labels(predictions, labels):
    with pytest.raises(ValueError):
        compare_human_labels(predictions, labels)
