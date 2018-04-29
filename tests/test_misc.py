import numpy as np
from misc.vectors import normalize


def test_vector_normalization():

    # non-zero norm
    s = np.random.randint(low=1,high=10,size=2)
    v = np.ones(s)
    np.testing.assert_almost_equal(
        np.linalg.norm(normalize(v), ord=1), 1.0)

    # zero norm
    s = np.random.randint(low=1, high=10, size=2)
    v = np.zeros(s)
    np.testing.assert_almost_equal(
        np.linalg.norm(normalize(v), ord=1), 0)