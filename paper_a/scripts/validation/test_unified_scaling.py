import numpy as np

from plot_unified_scaling import redundancy, summarize


def test_thresholds_precede_averaging():
    q = np.array([[.91,.5], [.92,.6], [.95,.91], [.99,.95]])
    r = redundancy(q)
    np.testing.assert_allclose(r, [8, 8/3])
    assert np.mean(r) != redundancy(np.mean(q, axis=1, keepdims=True))[0]


def test_noncrossing_is_undefined_and_counted():
    q = np.array([[.9,.1], [.95,.2], [.97,.3], [.99,.8]])
    r = redundancy(q)
    assert r[0] == 8 and np.isnan(r[1])
    result = summarize(r, np.random.default_rng(1))
    assert result['defined'] == 1 and result['total'] == 2 and result['mean'] == 8
    assert summarize(np.array([np.nan]), np.random.default_rng(1))['mean'] is None
