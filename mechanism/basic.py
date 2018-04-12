import numpy as np

from data.database import Database
from data.query import Query, Utility
from misc.vectors import normalize


def laplace(db, query, eps=1e-5):
    """
    Goal:
       The Laplace mechanism is a DP method for answering numeric queries,
       by adding Laplace[s(func)/eps] distributed noise to the true answer.
    Guaranties:
    1. (eps,0) Differential Privacy
    2. Accuracy:
       Let func: N ^ |Universe| -> R^k
       Denote y = laplace(db, func, eps)
              s = func.sensitivity
       Then forall beta in (0,1]
              P[max_i |f(x)_i - y_i| >= log(k/beta)(s/eps)] <= beta
    """

    assert (db.rep == 'histogram')
    assert (isinstance(query, Query))
    assert (isinstance(db, Database))
    assert (eps > 0)

    lap_param = query.sensitivity / eps
    return np.random.laplace(loc=query.value(db), scale=lap_param)


def exponential(db, utility, eps=1e-5):
    """
    Goal:
       The Exponential mechanism is a DP method for answering categorical queries,
       by sampling from an exponential distribution over possible choices.
    Guaranties:
    1. (eps,0) Differential Privacy
    2. Accuracy:
       Let utility: N ^ |Universe| x categories -> R
       Denote c* = exponential(db, utility, eps)
              s = utility.sensitivity
              R = utility.categories
              Opt(u,x) = max_{c in R} u(x,r)
       Then forall t > 0
              P[u(x,c*) <= Opt(u,x) - (2s / epsilon) (ln(|R|) + t)] <= e^-t
    """

    assert (db.rep == 'histogram')
    assert (isinstance(utility, Utility))
    assert (isinstance(db, Database))
    assert (eps > 0)

    # evaluate utility for all categories for db
    evals = np.array([utility.value(db, cat) for cat in utility.categories])

    # evaluate constants needed for the exponential parameter
    consts = eps / (2 * utility.sensitivity)

    # create weights
    weights = np.exp(consts * evals)

    # normalize to distribution and sample
    res = np.random.choice(utility.categories, p=normalize(weights, norm_ord=1))

    return res