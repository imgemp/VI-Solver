import numpy as np

from Domain import Domain


class Rosenbrock(Domain):

    def __init__(self, Dim=2):
        self.Dim = Dim + Dim % 2
        self.Min = 0.0

    def f(self, Data):
        return np.sum(
            100. * (Data[0::2] ** 2 - Data[1::2]) ** 2. + (Data[0::2] - 1) ** 2.)

    def F(self, Data):
        F = np.zeros(Data.shape)
        F[0::2] = 400. * Data[0::2] * \
            (Data[0::2] ** 2. - Data[1::2]) + 2. * (Data[0::2] - 1)
        F[1::2] = -200. * (Data[0::2] ** 2. - Data[1::2])
        return F

    def f_Error(self, Data):
        return self.f(Data) - self.Min
