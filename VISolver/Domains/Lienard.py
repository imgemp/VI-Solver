from __future__ import division
import numpy as np
from VISolver.Domain import Domain


class Lienard(Domain):

    def __init__(self):
        self.Dim = 2

    def F(self,Data):
        x,y = Data
        dxy = np.array([y-.32*(x**5.)+4./3.*(x**3.)-.8*x,-x])
        return dxy

    def Jac(self,Data):
        x,y = Data
        return np.array([[-5.*.32*(x**4.)+4.*(x**2.)-.8,1.],[-1.,0.]])
