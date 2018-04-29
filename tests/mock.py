import numpy as np

from data.query import Query, Utility
from data.database import Universe


def generate_database(m, n, uni_range_low=None, uni_range_high=None, exact_number=False):
    """
    - Generate Universe by picking n random integers from low (inclusive) to high (exclusive).
      If exact_number, then Universe.size == n
    - Generate a Database of m records, over the Universe
    """

    # generate Universe
    if exact_number:
        objects = range(n)
    else:
        objects = list(np.random.randint(uni_range_low, uni_range_high, size=n))
    uni = Universe(objects)

    # generate Database
    db = uni.random_db(m)

    return db


def linear_query(uni, return_underlying_vector=False):

    indices = np.random.uniform(low=0.0, high=1.0, size=uni.size)
    func = lambda db: np.inner(db.data, indices)

    if return_underlying_vector:
        # for testing
        return Query(uni, func, sensitivity=1), indices
    else:
        return Query(uni, func, sensitivity=1)


def categorical_linear_query(uni, return_underlying_matrix=False):

    num_categories = np.random.randint(low=1, high=50)
    categories_to_uni = np.random.uniform(low=0.0, high=1.0, size=(uni.size, num_categories))
    func = lambda db, cat: np.inner(db.data, categories_to_uni[:, cat])

    if return_underlying_matrix:
        # for testing
        return Utility(uni, list(range(num_categories)), func, sensitivity=1), categories_to_uni
    else:
        return Utility(uni, list(range(num_categories)), func, sensitivity=1)