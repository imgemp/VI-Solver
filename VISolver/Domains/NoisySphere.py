import numpy as np

from VISolver.Domain import Domain


class NoisySphere(Domain):

    def __init__(self,Dim=None,Sigma=1.):
        self.Dim = Dim
        self.Min = 0.0
        self.L = 2.0
        self.Sigma = Sigma

    def f(self,Data):
        return np.sum(Data**2)

    def F(self,Data):
        noise = np.random.normal(0.,self.Sigma,Data.shape)
        return 2.0*Data*(1.+noise)

    def f_Error(self,Data):
        return self.f(Data) - self.Min
