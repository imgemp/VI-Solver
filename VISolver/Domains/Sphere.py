import numpy as np

from VISolver.Domain import Domain


class Sphere(Domain):

    def __init__(self,Dim=None):
        self.Dim = Dim
        self.Min = 0.0
        self.L = 2.0

    def f(self,Data):
        return np.sum(Data**2)

    def F(self,Data):
        return 2.0*Data

    def J(self,Data):
        return 2.0*np.ones((Data.shape[0],Data.shape[0]))

    def f_Error(self,Data):
        return self.f(Data) - self.Min
