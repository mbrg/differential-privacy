import numpy as np

from data.database import Database
from data.query import Query


def laplace(db, query, eps=1e-5):
    """
    Goal:
       The Laplace mechanism is a DP method to answer numeric queries,
       by adding Laplace[s(func)/eps] distributed noise to the true answer
    Guaranties:
    1. (eps,0) Differential Privacy
    2. Accuracy:
       Let func: N ^ |Universe| -> R^k
       Denote y = laplace(db, func, eps)
              s = func.sensitivity
       Then forall d in (0,1]
              P[max_i |f(x)_i - y_i| >= log(k/d)(s/eps)] <= d
    """

    assert (db.rep == 'histogram')
    assert (isinstance(query, Query))
    assert (isinstance(db, Database))
    assert (eps > 0)

    lap_param = query.sensitivity / eps
    return np.random.laplace(loc=query.value, scale=lap_param)
