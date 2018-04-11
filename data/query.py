import numpy as np
from data.database import Universe


class Query(object):
    def __init__(self, uni, func, sensitivity=None):
        assert (isinstance(uni, Universe))
        assert (callable(func))

        self._uni = uni
        self._func = func

        # lazy calculated attributed
        self._value = None
        self._sensitivity = sensitivity

    def eval_sensitivity(self, num_trials=1, random_db_size=None):
        """
        Evaluates query sensitivity for generating num_trials dbs and calculating their sensitivity.
        If random_db_size is provided, it is used to generate the dbs. Otherwise - Universe size is used.
        """

        max_sens = 0
        m = random_db_size if isinstance(random_db_size, int) else self._uni.size

        for j in range(num_trials):

            # create a new random
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
            self._sensitivity = self.eval_sensitivity()
        return self._sensitivity