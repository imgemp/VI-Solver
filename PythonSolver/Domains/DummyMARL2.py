import numpy as np

from Domain import Domain

class MARL2(Domain):

    def __init__(self,Dim=2):
        self.Dim = Dim
        self.Min = 0.0

    def f(self,Data):
        return None

    def F(self,Data):
        F = np.zeros(Data.shape)
        F[0] = Data[1]
        F[1] = -Data[0]
        return F

    def Origin_Error(self,Data):
        return np.sum(Data**2)