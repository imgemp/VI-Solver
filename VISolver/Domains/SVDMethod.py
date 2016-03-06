import numpy as np

from VISolver.Domain import Domain


class SVDMethod(Domain):

    def __init__(self,Data,keepData=False,tau=1.,Dim=None):
        self.Data = self.load_data(Data)
        self.keepData = keepData
        self.tau = tau
        self.Dim = Dim

    def load_data(self,Data):
        self.mask = (Data != 0).toarray()
        self.fro = np.linalg.norm(Data.toarray(),ord='fro')
        return Data

    def F(self,parameters):
        R = self.shrink(parameters,self.tau)
        return np.asarray(self.Data-R)*self.mask

    def shrink(self,x,tau):
        U, S, Vt = np.linalg.svd(x,full_matrices=False)
        s = np.clip(S-tau,0.,np.inf)
        R = U.dot(np.diag(s)).dot(Vt)
        return R

    def rel_error(self,parameters):
        return np.linalg.norm(self.F(parameters),ord='fro')/self.fro
