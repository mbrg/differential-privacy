import numpy as np

from mechanism.multiqueries import small_db, AT
from misc.vectors import normalize, norm
from tests.mock import generate_database, linear_query


def test_smalldb(iters=10, num_queries=5, smalldb_samples=10, eps=1e-5, alpha=1e-0, beta=1e-5):

    for i in range(iters):
        # generate random database
        r = np.random.randint(0, 15, 2)
        db = generate_database(r[0] + 1, r[1] + 2, exact_number=True)

        # generate multiple one-dimensional linear query
        queries = []
        for j in range(num_queries):
            queries.append(linear_query(db.uni))

        # verify mechanism accuracy guarantee
        lg_u = np.log(db.uni.size)
        lg_q = np.log(len(queries))
        lg_b = np.log(1/beta)
        db_nrm = np.linalg.norm(db.data, ord=1)
        acceptable_err = np.power(db_nrm, 2/3) * \
                         np.power((1/eps) * (16*lg_u*lg_q + 4*lg_b), 1/3)

        # sample smalldb
        bad_case_cntr = 0
        for j in range(smalldb_samples):
            y = small_db(db, queries, eps, alpha)
            err = max([norm(np.subtract(q.value(db), q.value(y)), ord=1)
                       for q in queries])
            if err > acceptable_err:
                bad_case_cntr += 1

        # We know from the Small DB guaranties that
        # P[ M > db.size**(2/3) * ((16*log(|U|)*log(|Q|) + 4*log(1/beta)))/eps)**(1/3) ] <= beta
        # we use (bad_case_cntr / lap_samples) as an estimation for the probability of a too large error
        assert (beta >= (bad_case_cntr / smalldb_samples))


def test_at(iters=10, at_samples=10, eps=1e-5, alpha=1e-2):

    for i in range(iters):
        # generate random database
        r = np.random.randint(0, 1000, 3)
        db = generate_database(r[0] + 1, r[1] + 2, exact_number=True)

        # generate multiple one-dimensional linear query
        queries = []
        for j in range(r[2]):
            queries.append(linear_query(db.uni))

        # AT arguments
        y = np.array([q.value(db) for q in queries])
        thresh = np.percentile(y, 80)
        num_at_queries = np.sum(y >= (thresh - alpha))
        num_queries = r[2]
        beta = (4 * num_at_queries) / np.exp(eps * alpha * (1 / (9*num_at_queries)) - np.log(num_queries))
        at = AT(db, num_queries, num_at_queries, thresh, eps, 0)

        # sample AT
        bad_case_cntr = 0
        for j in range(at_samples):
            for k in range(num_queries):
                try:
                    y_hat = at.value(queries[k])
                    if ((y_hat is None and y[k] > thresh + alpha) or
                        (y_hat is not None and np.abs(y[k]-y_hat) > alpha)):
                        bad_case_cntr += 1
                except AssertionError:
                    # AT exceeded max queries or max AT queries
                    bad_case_cntr += 1

        assert (beta >= (bad_case_cntr / (at_samples*num_queries)))

