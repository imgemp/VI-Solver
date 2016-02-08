import numpy as np

from VISolver.Domain import Domain


class PowerIteration(Domain):

    def __init__(self,A):
        self.A = np.asarray(A)
        self.Dim = A.shape[-1]

    def F(self,Data):
        v = self.A.dot(Data)
        return v - Data

    def res_norm(self,Data):
        Av = np.dot(self.A,Data)
        Av_norm = np.linalg.norm(Av)
        v_norm = np.linalg.norm(Data)
        res = Av/Av_norm - Data/v_norm
        return np.linalg.norm(res)
