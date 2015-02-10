import numpy as np

from Domain import Domain


class Watson(Domain):

    def __init__(self, Pos=0):
        self.Dim = 10
        self.Min = 0.0
        self.L = 10.0
        self.Pos = Pos
        self.A = np.double(np.array([[0, 0, -1, -1, -1, 1, 1, 0, 1, 1],
                                     [-2, -1, 0, 1, 1, 2, 2, 0, -1, 0],
                                     [1, 0, 1, -2, -1, -1, 0, 2, 0, 0],
                                     [2, 1, -1, 0, 1, 0, -1, -1, -1, 1],
                                     [-2, 0, 1, 1, 0, 2, 2, -1, 1, 0],
                                     [-1, 0, 1, 1, 1, 0, -1, 2, 0, 1],
                                     [0, -1, 1, 0, 2, -1, 0, 0, 1, -1],
                                     [0, -2, 2, 0, 0, 1, 2, 2, -1, 0],
                                     [0, -1, 0, 2, 2, 1, 1, 1, -1, 0],
                                     [2, -1, -1, 0, 1, 0, 0, -1, 2, 2]]))
        self.b = np.zeros(self.Dim)
        self.b[self.Pos] = 1.0

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
