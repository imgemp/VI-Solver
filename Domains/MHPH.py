import numpy as np

from VISolver.Domains import Domain


class MHPH(Domain):

    def __init__(self, dim=1000):
        self.dim = dim
        self.min = 0.0
        self.l = (15.0 * np.double(dim)) ** 2
        M = np.random.uniform(low=-15, high=-12, size=(self.dim, self.dim))
        self.A = np.dot(M, M.T)
        self.b = np.random.uniform(low=-500, high=0, size=self.dim)

    def F(self, Data):
        return np.dot(self.A, Data) + self.b

    def gap_simplex(self, Data):
        gap = 0.0
        F = np.ravel(self.F(Data))
        count = 0
        z = 1.0
        for ind in abs(F).argsort()[::-1]:
            if (F[ind] < 0) or (count == len(F) - 1):
                diff = Data[ind] - z
                gap += F[ind] * diff
                count += 1
                z = 0.0
            else:
                diff = Data[ind] - 0.0
                gap += F[ind] * diff
                count += 1
        return gap
