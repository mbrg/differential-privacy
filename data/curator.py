from data.database import Database


class OnlineCurator(object):

    def __init__(self, db, num_queries):

        assert (isinstance(db, Database))
        assert (num_queries > 0)

        self._db = db
        self._num_queries = num_queries
        self._answered = 0

    def eval(self, query):
        raise NotImplementedError

    def copy(self):
        """
        return an independent copy of self
        """
        return OnlineCurator(self._db.copy(), self._num_queries)

    def __str__(self):
        return '[OnlineCurator (db=%s, num_queries=%s)]' % (self._db, self._num_queries)