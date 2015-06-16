import numpy as np

from Domain import Domain


class Sun(Domain):

    def __init__(self, Dim=8000):
        self.Dim = Dim
        self.Min = 0.0
        self.L = (2.0 * np.double(Dim)) ** 2
        L = np.zeros((self.Dim, self.Dim))
        U = np.triu(2 * np.ones((self.Dim, self.Dim)), 1)
        D = np.diag(np.ones(self.Dim), 0)
        self.A = L + U + D
        self.b = -1 * np.ones(self.Dim)

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
