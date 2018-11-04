import numpy as np
from misc.vectors import normalize, norm


class Universe(object):
    def __init__(self, objects):

        self.objects = np.array(list(set(objects)))

        assert (self.size > 0), 'Universe cannot be empty.'

    def random_db(self, m):
        """
        Generate a Database of m records, over the Universe
        """

        assert (m >= 0)

        # randomly generate database
        choose_m_records = np.random.randint(low=0, high=self.size, size=m)

        # reorder to right format
        database = np.bincount(choose_m_records, minlength=self.size)

        # generate Database
        db = Database(database, 'histogram', self)

        return db

    @property
    def shape(self):
        return self.objects.shape

    @property
    def size(self):
        return self.objects.shape[0]

    def copy(self):
        """
        return an independent copy of self
        """
        return Universe(self.objects[:])

    def __str__(self):
        return '[Universe (size=%d)]' % self.size


class Database(object):
    REPS = ('histogram', 'probability')

    def __init__(self, database, rep, uni):

        assert (rep in Database.REPS)
        assert (isinstance(uni, Universe))
        assert (isinstance(database, np.ndarray))
        assert (database.shape == uni.shape)

        self.data = database
        self.rep = rep
        self.uni = uni

        # lazy calculated attributed
        self._norm = None

    def change_representation(self, rep):

        assert (rep in Database.REPS)

        if self.rep == 'histogram' and rep == 'probability':
            _ = self.norm  # trigger norm evaluation
            self.data = normalize(self.data)
        if self.rep == 'probability' and rep == 'histogram':
            if self._norm is None:
                raise Exception('Cannot change representation from probability to histogram without _norm.')
            else:
                self.data = self.norm * self.data

    def _eval_norm(self):
        return norm(self.data, ord=1) if self.rep == 'histogram' else None

    @property
    def norm(self):
        if self._norm is None:
            self._norm= self._eval_norm()
        return self._norm

    def copy(self):
        """
        return an independent copy of self
        """
        return Database(self.data[:], self.rep, self.uni.copy())

    def __str__(self):
        return '[Database (data=%s, rep=%s, uni=%s)]' % (self.data.shape, self.rep, self.uni)