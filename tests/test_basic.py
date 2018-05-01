import numpy as np
from tests.mock import generate_database, linear_query, categorical_linear_query
from mechanism.basic import laplace, exponential, report_noisy_max
from data.query import Query


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


def test_query_copy():

    # generate database
    db = generate_database(m=20, n=10, exact_number=True)
    uni = db.uni

    # generate query
    query = linear_query(uni)

    # copy query
    cquery = query.copy()

    # change and verify independence
    query._uni.objects = [1,]
    assert (1 != len(cquery._uni.objects))


def test_query_dim_evaluation(iters=10):

    # random query dimension
    query_dim = np.random.randint(1, 20)

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)

        # generate multiple one-dimensional linear query
        queries = []
        for j in range(query_dim):
            queries.append(linear_query(db.uni))

        # generate multiple-dimension linear query
        func = lambda x: np.array([q._func(x) for q in queries])
        multidim_query = Query(db.uni, func, dim=query_dim, sensitivity=query_dim)

        # test eval dim
        assert (multidim_query.dim == query_dim)


def test_utility_copy():
    # generate database
    db = generate_database(m=20, n=10, exact_number=True)
    uni = db.uni

    r = np.random.randint(0, 1000, 4)
    db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
    utility, umat = categorical_linear_query(db.uni, return_underlying_matrix=True)

    # generate utility
    utility = categorical_linear_query(uni)

    # copy query
    cutility = utility.copy()

    # change universe and verify independence
    utility._uni.objects = [1, ]
    assert (1 != len(cutility._uni.objects))


def test_database_copy():
    # generate database
    db = generate_database(m=20, n=10, exact_number=True)

    # copy database
    cdb = db.copy()

    # change and verify independence
    db.uni.objects = [1, ]
    assert (10 == len(cdb.uni.objects))


def test_linear_query_evaluation(iters=1000):

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)

        query, qvec = linear_query(db.uni, return_underlying_vector=True)

        assert (np.inner(qvec, db.data) == query.value(db))


def test_linear_query_sensitivity(func_iters=10, db_iters=10):

    for i in range(func_iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
        query = linear_query(db.uni)

        assert (1 >= query._eval_sensitivity(db_iters, random_db_size=r[0]))


def test_utility_evaluation(iters=10):

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
        utility, umat = categorical_linear_query(db.uni, return_underlying_matrix=True)

        for c in utility.categories:
            assert (np.inner(db.data, umat[:, c]) == utility.value(db, c))


def test_utility_sensitivity(func_iters=10, db_iters=10):

    for i in range(func_iters):

        # generate random database
        r = np.random.randint(0, 100, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)
        utility = categorical_linear_query(db.uni)

        assert (1 >= utility._eval_sensitivity(db_iters, random_db_size=r[0]))


def test_laplace_accuracy_on_linear_queries(iters=10, lap_samples=10, eps=1e-5, beta=1e-5):

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)

        # generate random query
        query = linear_query(db.uni)

        # verify Laplace mechanism accuracy guarantee
        acceptable_err = np.log(query.dim / beta) * (query.sensitivity / eps)

        # sample laplace values to estimate the probability of error
        bad_case_cntr = 0
        for j in range(lap_samples):
            y = laplace(db, query, eps)
            ind_actual_err = np.abs(np.subtract(y, query.value(db)))
            actual_err = ind_actual_err if isinstance(ind_actual_err, (int, float)) else max(ind_actual_err)
            if actual_err > acceptable_err:
                bad_case_cntr += 1

        # We know from the Laplace Mechanism guaranties that
        # P[max_i |f(x)_i - y_i| >= log(query.dim/beta)(query.sensitivity/eps)] <= beta
        # we use (bad_case_cntr / lap_samples) as an estimation for the probability of a too large error
        assert (beta >= (bad_case_cntr / lap_samples))


def test_exponential_utility_on_linear_queries(iters=10, exp_samples=10, eps=1e-5, t=5):

    for i in range(iters):

        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)

        # generate random utility
        utility = categorical_linear_query(db.uni)

        # verify Exponential mechanism accuracy guarantee
        acceptable_val = utility.optimal(db) - ((2 * utility.sensitivity) / eps) * (np.log(len(utility.categories)) + t)

        # sample exponential values to estimate the probability of error
        bad_case_cntr = 0
        for j in range(exp_samples):
            c = exponential(db, utility, eps)
            actual_val = utility.value(db, c)
            if actual_val <= acceptable_val:
                bad_case_cntr += 1

        # We know from the Exponential Mechanism guaranties that
        # P[u(db,c) <= utility.optimal(db) - (2 * utility.sensitivity / eps) (ln(|utility.categories|) + t)] <= e^-t
        # we use (bad_case_cntr / lap_samples) as an estimation for the probability of a too large error
        assert (np.exp(-t) >= (bad_case_cntr / exp_samples))


def test_report_noise_max(iters=10, rnm_samples=10, query_dim=5, eps=1e-5, beta=1e-5):

    for i in range(iters):
        # generate random database
        r = np.random.randint(0, 1000, 4)
        db = generate_database(r[0], r[1] + 1, r[2], r[2] + r[3] + 1, False)

        # generate multiple one-dimensional linear query
        queries = []
        for j in range(query_dim):
            queries.append(linear_query(db.uni))

        # test one dimensional RNM
        assert (0 == report_noisy_max(db, queries[0], eps))

        # generate multiple-dimension linear query
        func = lambda x: np.array([q._func(x) for q in queries])
        multidim_query = Query(db.uni, func, dim=query_dim, sensitivity=query_dim)
        true_values = [q.value(db) for q in queries]
        true_argmax = np.argmax(true_values)

        # verify mechanism accuracy guarantee
        acceptable_err = np.log(1 / beta) / eps

        # sample multi-dimensional RNM values to estimate the probability of error
        bad_case_cntr = 0
        for j in range(rnm_samples):
            noise_argmax = report_noisy_max(db, multidim_query, eps)
            if true_values[noise_argmax] > true_values[true_argmax]:
                bad_case_cntr += 1

        # We know from the Report Noise Max guaranties that
        # P[max_i |y[true_argmax] - t[rnm_argmax]| >= log(1/beta)(1/eps)] <= beta
        # we use (bad_case_cntr / lap_samples) as an estimation for the probability of a too large error
        assert (beta >= (bad_case_cntr / rnm_samples))
