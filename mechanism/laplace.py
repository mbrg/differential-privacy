import numpy as np

from data.database import Database
from data.query import Query


def laplace(db, query, eps=1e-5):

    assert (db.rep == 'histogram')
    assert (isinstance(query, Query))
    assert (isinstance(db, Database))
    assert (eps > 0)

    lap_param = query.sensitivity / eps
    return np.random.laplace(loc=query.value, scale=lap_param)
