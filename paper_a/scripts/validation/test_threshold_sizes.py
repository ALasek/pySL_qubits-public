import numpy as np

from plot_threshold_sizes import threshold_size, summarize


def test_average_sizes_not_inverse_average_redundancy():
    q = np.array([[.91, .5], [.92, .6], [.95, .91], [.99, .95]])
    sizes = threshold_size(q)
    np.testing.assert_array_equal(sizes, [1, 3])
    assert np.mean(sizes) == 2
    assert np.mean(sizes) != 8 / np.mean(8 / sizes)


def test_noncrossing_stays_undefined():
    sizes = threshold_size(np.array([[.9, .1], [.95, .8]]))
    assert sizes[0] == 1 and np.isnan(sizes[1])
    stats = summarize(sizes, np.random.default_rng(1))
    assert stats['mean'] == 1 and stats['defined'] == 1 and stats['total'] == 2
