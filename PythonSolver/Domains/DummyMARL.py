import numpy as np

from Domain import Domain

class MARL(Domain):

    def __init__(self,Dim=2):
        self.Dim = Dim
        self.Min = 0.0

    def f(self,Data):
        return None

    def F(self,Data):
        F = np.zeros(Data.shape)
        F[0] = Data[1]   -Data[0]
        F[1] = -Data[0]  -Data[1]
        return F

    def Origin_Error(self,Data):
        print(np.sum(Data**2))
        return np.sum(Data**2)