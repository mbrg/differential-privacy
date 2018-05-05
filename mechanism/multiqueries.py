import numpy as np
from itertools import combinations_with_replacement
from math import ceil

from data.database import Database
from data.query import Query, Utility
from data.curator import OnlineCurator
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


class AT(OnlineCurator):
    """
    Goal:
        The AboveTreshold mechanism is a DP method for answering multiple numerical queries which
        are above a certain threshold, while paying in accuracy abose-threshold queries
    Guaranties:
        1. (eps,delta) Differential Privacy
        2. Accuracy:
           Let f_1,..f_k be a series of queries s.t |{f_i(db) >= T-alpha}| <= num_at_queries
               c = num_at_queries
               k = num_queries
           Then for
              delta>0 =>
                 alpha=(1/eps)(log(k) + log(4*c/beta))sqrt(c*log(2/delta))(sqrt(512)+1)
              delta=0 =>
                 alpha=(1/eps)9c(log(k)+log(4c/beta))

           With probability 1-beta
                  AT(f_i(db)) in R    => |AT(f_i(db)) - f_i(db)| <= alpha
                  AT(f_i(db)) is None => f_i(db) <= thresh + alpha
    """

    def __init__(self, db, num_queries, num_at_queries, thresh, eps, delta):
        super().__init__(db, num_queries)
        self._thresh = thresh
        self._num_at_queries = num_at_queries
        # init private threshold and AT answered query counter
        if delta==0:
            self._eps1 = eps * (8/9)
            self._eps2 = eps * (2/9)
            self._sigma = lambda e: (2*num_at_queries)/e
        else:
            self._eps1 = eps * (np.sqrt(512) / (np.sqrt(512) + 1))
            self._eps2 = eps * (2 / (np.sqrt(512) + 1))
            self._sigma = lambda e: np.sqrt(32*num_at_queries*np.log(1/delta))/e
        self._pthresh = np.random.laplace(loc=thresh, scale=self._sigma(self._eps1))
        self._at_answered = 0
        self._answered = 0

    def value(self, query):

        assert (self._at_answered < self._num_at_queries), "Maximal number of AT queries exceeded"
        assert (self._answered < self._num_queries), "Maximal number of queries exceeded"
        assert (query.dim == 1)

        # calculate noise answer
        y = query.value(self._db)
        v = np.random.laplace(loc=y, scale=2*self._sigma(self._eps1))
        self._answered += 1

        # is above threshold
        if v >= self._pthresh:
            self._at_answered += 1
            self._pthresh = np.random.laplace(loc=self._thresh, scale=self._sigma(self._eps1))
            return np.random.laplace(loc=y, scale=self._sigma(self._eps2))
        else:
            return None

    def __str__(self):
        return '[AT (db=%s, num_at_queries=%s)]' % (self._db, self._num_queries)


