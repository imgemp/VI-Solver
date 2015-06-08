import numpy as np
from scipy.optimize import rosen, rosen_der

from VISolver.Domain import Domain


class NoisyRosenbrock(Domain):

    def __init__(self,Dim=2,Sigma=1.):
        self.Dim = Dim + Dim % 2
        self.Sigma = Sigma
        self.Min = 0.0

    def f(self,Data):
        return rosen(Data)

    def F(self,Data):
        if self.Sigma > 0:
            noise = np.random.normal(0.,self.Sigma,Data.shape)
        else:
            noise = 0.
        return rosen_der(Data)*(1. + noise)

    def f_Error(self,Data):
        return self.f(Data) - self.Min
