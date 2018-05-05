import numpy as np
from itertools import combinations_with_replacement
from math import ceil

from data.database import Database
from data.query import Query, Utility
from misc.vectors import normalize, norm
from mechanism.basic import exponential


def small_db(db, queries, eps=1e-5, alpha=1e-5):
    """
    Goal:
        The Small DB mechanism is a DP method for answering multiple nomerical queries,
        by sampling a small db which estimates the original db, and answering queries
        using the small db.
    Guaranties:
        1. (eps,0) Differential Privacy
        2. Accuracy:
           Let utility: N ^ |Universe| x categories -> R
           Denote y = small_db(db, queries, eps, alpha)
                  |U| = db.uni.size
                  |Q| = len(queries)
                  M = max_{ |q(db) - q(y)| | q \in queries}
           Then forall beta > 0
                  P[ M > db.size**(2/3) * ((16*log(|U|)*log(|Q|) + 4*log(1/beta)))/eps)**(1/3) ] <= beta
    """

    assert (isinstance(db, Database))
    assert (db.uni.size >= 2)
    for q in queries: assert (isinstance(q, Query))
    assert (db.rep == 'histogram')
    assert (eps > 0)
    assert (alpha > 0)

    # create utility function
    small_db_size =  ceil(np.log(len(queries)) / (alpha ** 2))
    categories = [Database(sum(c), 'histogram', db.uni)
                  for c in combinations_with_replacement(
                      np.identity(db.uni.size, dtype=int),
                      small_db_size)]
    func = lambda x, y: - max([norm(np.subtract(q.value(x), q.value(y)), ord=1)
                               for q in queries])
    utility = Utility(db.uni, categories, func, sensitivity=1)

    # sample and output db with the exponential mechanism
    y = exponential(db, utility, eps)
    return y