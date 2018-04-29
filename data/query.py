import numpy as np
from data.database import Universe


class Query(object):
    def __init__(self, uni, func, dim=None, sensitivity=None):
        assert (isinstance(uni, Universe))
        assert (callable(func))

        self._uni = uni
        self._func = func

        # lazy calculated attributed
        self._dim = dim
        self._sensitivity = sensitivity

    def _eval_sensitivity(self, num_trials=1, random_db_size=None):
        """
        Evaluates query sensitivity by generating num_trials dbs and calculating their sensitivity.
        If random_db_size is provided, it is used to generate the dbs. Otherwise - Universe size is used.
        """

        max_sens = 0
        m = random_db_size if isinstance(random_db_size, int) else self._uni.size

        for j in range(num_trials):

            # create a new random database
            data = self._uni.random_db(m).data
            perturb = data[:]

            for i in range(self._uni.size):
                for p in [1, -1]:
                    perturb[i] += p

                    if perturb[i] >= 0:
                        perturb_val = np.abs(np.subtract(self.value(data), self.value(perturb)))
                        max_sens = max(max_sens, perturb_val)

                    perturb[i] -= p

        return max_sens

    def value(self, db):
        return self._func(db)

    @property
    def sensitivity(self):
        if self._sensitivity is None:
            self._sensitivity = self._eval_sensitivity()
        return self._sensitivity

    def _eval_dim(self):
        """
        Obtain query output dimension by triggering it on a random input.
        """

        # create a new random database
        db = self._uni.random_db(self._uni.size)

        # trigger query
        result = self.value(db)

        dim = 1 if isinstance(result, (int, float)) else result.shape[0]
        return dim

    @property
    def dim(self):
        if self._dim is None:
            self._dim= self._eval_dim()
        return self._dim

    def copy(self):
        """
        return an independent copy of self
        """
        return Query(self._uni.copy(), self._func, dim=self._dim, sensitivity=self._sensitivity)

    def __str__(self):
        return '[Query (uni=%s, dim=%d, sensitivity=%.2f)]' % (self._uni, self._dim, self._sensitivity)


class Utility(object):
    def __init__(self, uni, categories, func, sensitivity=None):
        assert (isinstance(uni, Universe))
        assert (callable(func))

        self.categories = categories
        self._uni = uni
        self._func = func

        # lazy calculated attributed
        self._sensitivity = sensitivity

    def _eval_sensitivity(self, num_trials=1, random_db_size=None):
        """
        Evaluates utility sensitivity by generating num_trials dbs and calculating their sensitivity.
        If random_db_size is provided, it is used to generate the dbs. Otherwise - Universe size is used.
        """

        max_sens = 0
        m = random_db_size if isinstance(random_db_size, int) else self._uni.size

        for j in range(num_trials):

            # create a new random
            data = self._uni.random_db(m).data
            perturb = data[:]

            # for each category
            for c in self.categories:

                for i in range(self._uni.size):
                    for p in [1, -1]:
                        perturb[i] += p

                        if perturb[i] >= 0:
                            perturb_val = np.abs(np.subtract(self.value(data, c), self.value(perturb, c)))
                            max_sens = max(max_sens, perturb_val)

                        perturb[i] -= p

        return max_sens

    def value(self, db, cat):

        assert (cat in self.categories)

        return self._func(db, cat)

    def optimal(self, db):
        """
        Optimal utility value for this db
        """
        return max([self.value(db, c) for c in self.categories])

    @property
    def sensitivity(self):
        if self._sensitivity is None:
            self._sensitivity = self._eval_sensitivity()
        return self._sensitivity

    def copy(self):
        """
        return an independent copy of self
        """
        return Utility(self._uni.copy(), self.categories, self._func, sensitivity=self._sensitivity)

    def __str__(self):
        return '[Utility (uni=%s, num_categories=%d, sensitivity=%.2f)]' % (self._uni, len(self.categories), self._sensitivity)