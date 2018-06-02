import numpy as np

from VISolver.Domain import Domain


class MonotoneCycle(Domain):

    def __init__(self):
        self.Dim = 2
        self.Min = 0.0
        self.A = np.array([[0,-1],[1,0]])
        self.b = np.array([0,0])

    def f(self,Data):
        return 0

    def F(self,Data):
        return np.dot(self.A,Data)+self.b

    def J(self,Data):
        return self.A

    def f_Error(self,Data):
        return self.f(Data) - self.Min
