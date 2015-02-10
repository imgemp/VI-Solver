import numpy as np

from Domain import Domain


class RG(Domain):

    def __init__(self, Dim=1000):
        self.Dim = Dim
        self.Min = 0.0
        self.L = (15.0 * np.double(Dim)) ** 2
        self.A = np.random.uniform(
            low=-50,
            high=150,
            size=(
                self.Dim,
                self.Dim))
        self.b = np.random.uniform(low=-200, high=300, size=self.Dim)

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
