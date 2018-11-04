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


class PMW(OnlineCurator):
    """
    Goal:
        The Private Multiplicative Weights mechanism is a DP method for answering multiple numerical queries by
         holding and improving an estimation of the database
    Guaranties:
        1. (eps,delta) Differential Privacy
        2. Accuracy:
           Let f_1,..f_k be a series of queries s.t |{f_i(db) >= T-alpha}| <= num_at_queries
               k = num_queries
               |u| = db.uni.size
           Then for
              delta=0 =>
                 alpha=(db.norm^(2/3)/eps^(1/3)) * [(36 * log(|u|) * (log(k) + log((32(log(|u|))^(1/3) * db.norm^(2/3))/beta))]
              delta>0 =>
                 alpha=[db.norm*(1/eps)(2 + 32*sqrt(2))sqrt(log(|u|)log(1/delta))(log(k) + log((32(log(|u|))]^(1/2)
           With probability 1-beta
                  |f_i(db)-ans_i(db)| <= 3*alpha
    """
    def __init__(self, db, num_queries, eps, delta, alpha, beta):
        # verify representation
        assert(db.rep == 'histogram')
        db.change_representation('probability')

        super().__init__(db, num_queries)

        # init uniform hypothesis
        self._at_answered = 0
        self._h = Database(normalize(np.ones(db.data.shape), ord=1), 'probability', db.uni)

        # init AboveThreshold
        c = (4 * np.log(db.uni.size)) / np.power(alpha, 2)
        dbnrm = db.norm
        if delta==0:
            T = (1 / (eps * dbnrm)) * 18 * c * (np.log(2 * num_queries) + np.log((4 * c) / beta))
        else:
            T = (1 / (eps * dbnrm)) * (2 + 32 * np.sqrt(2)) * np.sqrt(c * np.log(2/delta)) * (np.log(num_queries) + np.log((4 * c) / beta))
        self._at = AT(db, 2*num_queries, c, T, eps, delta)

    def value(self, query):

        assert (self._answered < self._num_queries), "Maximal number of queries exceeded"
        assert (query.dim == 1)

        # noisy error
        q_h = query.value(self._h)
        f1 = lambda d: query.value(d) - q_h
        f2 = lambda d: q_h - query.value(d)
        q1 = Query(self._db.uni, f1, dim=1, sensitivity=1 / self._db.norm)
        q2 = Query(self._db.uni, f2, dim=1, sensitivity=1 / self._db.norm)
        e1 = self._at.value(q1)
        e2 = self._at.value(q2)

        # answer and update
        self._answered += 1
        if e1 is None and e2 is None:
            res = q_h
        else:
            if e2 is None:
                res = q_h + e1
            else: # e1 is None
                res = q_h - e2
            self._h = self._mw_update(self._h, query, res)
            self._at_answered += 1
        return res

    def _mw_update(self, db, query, vals, etha=1e-4):

        # directed query value
        y = query.value(db)
        r = (vals < y)*y + (vals >= y)*(1-y)

        # next Database
        new_xp = np.multiply(np.exp(-etha*r), db.data)
        new_x = normalize(new_xp, ord=1)
        new_db = Database(new_x , 'probability', db.uni)
        new_db._norm = db.norm

        return new_db

    def __str__(self):
        return '[PMW (db=%s, num_queries=%s)]' % (self._db, self._num_queries)
