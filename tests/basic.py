import numpy as np
from tests.mock import generate_database, linear_query


def test_database_rep_change():

    # generate random database
    r = np.random.randint(0, 1000, 4)
    exact_number = np.random.choice([True, False])

    db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, exact_number)

    # test rep change
    db.change_representation(rep='probability')
    assert (np.isclose(1.0, np.sum(db.data)))


def test_generate_database(iters=1000):

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        exact_number = np.random.choice([True, False])

        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, exact_number)

        if exact_number:
            assert (r[1] + 1 == db.uni.size)
        else:
            assert (r[1] + 1 >= db.uni.size)

        assert (db.data.shape == db.uni.shape)
        assert (r[0] == np.sum(db.data))


def test_linear_query_evaluation(iters=1000):

    for i in range(iters):

        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
        query, qvec = linear_query(db.uni, return_underlying_vector=True)

        assert (np.inner(qvec, db.data) == query.value(db))


def test_linear_query_sensitivity(func_iters=10, db_iters=10):

    for i in range(func_iters):

        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
        query = linear_query(db.uni)

        assert (1 >= query.eval_sensitivity(db_iters, random_db_size=r[0]))
