from __future__ import division
import numpy as np
from VISolver.Domain import Domain


class Test(Domain):

    def __init__(self,gap_alpha=2):
        self.Dim = 2
        self.gap_alpha = gap_alpha

    def F(self,Data):
        x,y = Data
        dxy = np.array([-2*(x**2+y**2)*np.exp(-10*(x**2+y**2)**2),1/2*x])
        return dxy
