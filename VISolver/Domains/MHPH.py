import numpy as np

from VISolver.Domain import Domain


class MHPH(Domain):

    def __init__(self,Dim=1000):
        self.Dim = Dim
        self.Min = 0.0
        self.L = (15.0*np.double(Dim))**2
        M = np.random.uniform(low=-15,high=-12,size=(self.Dim,self.Dim))
        self.A = np.dot(M,M.T)
        self.b = np.random.uniform(low=-500,high=0,size=self.Dim)

    def F(self,Data):
        J_now = self.J(Data)
        F_reg = np.dot(self.A,Data)+self.b
        F_perp = np.dot(J_now-J_now.T,F_reg)
        return F_reg - F_perp

    def J(self,Data):
        return self.A

    def gap_simplex(self,Data):
        gap = 0.0
        F = np.ravel(self.F(Data))
        count = 0
        z = 1.0
        for ind in abs(F).argsort()[::-1]:
            if (F[ind] < 0) or (count == len(F)-1):
                diff = Data[ind]-z
                gap += F[ind]*diff
                count += 1
                z = 0.0
            else:
                diff = Data[ind]-0.0
                gap += F[ind]*diff
                count += 1
        return gap
