import sys
from unittest.mock import patch
import pytest
from compass.judges.reliability import wilson_interval


def test_confidence_levels_do_not_depend_on_scipy():
    intervals=[]
    for level in [.8,.9,.95,.99]:
        expected=wilson_interval(5,10,level)
        with patch.dict(sys.modules, {'scipy':None,'scipy.stats':None}):
            assert wilson_interval(5,10,level)==expected
        intervals.append(expected)
    assert all(a[0]>b[0] and a[1]<b[1] for a,b in zip(intervals,intervals[1:]))
    assert intervals[-1] == pytest.approx((.1842255182472355,.8157744817527646))


@pytest.mark.parametrize('args',[(-1,0,.95),(2,1,.95),(0,-1,.95),(True,1,.95),(1.5,2,.95),
                                (0,0,0),(0,0,1),(0,0,float('nan')),(0,1,float('inf'))])
def test_invalid_inputs_rejected(args):
    with pytest.raises(ValueError): wilson_interval(*args)


def test_extremes_and_empty_sample():
    assert wilson_interval(0,0)==(0,1)
    assert wilson_interval(0,10)[0]==0
    assert wilson_interval(10,10)[1]==1
