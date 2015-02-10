import numpy as np

from Domain import Domain


class KojimaShindo(Domain):

    def __init__(self):
        self.Dim = 4
        self.Min = 0.0
        self.L = 10.0

    def F(self, Data):
        F = np.array(Data)
        x1 = Data[0]
        x2 = Data[1]
        x3 = Data[2]
        x4 = Data[3]
        F[0] = 3 * (x1 ** 2) + 2 * x1 * x2 + 2 * (x2 ** 2) + x3 + 3 * x4 - 6
        F[1] = 2 * (x1 ** 2) + x1 + (x2 ** 2) + 10 * x3 + 2 * x4 - 2
        F[2] = 3 * (x1 ** 2) + x1 * x2 + 2 * (x2 ** 2) + 2 * x3 + 9 * x4 - 9
        F[3] = (x1 ** 2) + 3 * (x2 ** 2) + 2 * x3 + 3 * x4 - 3
        return F

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
