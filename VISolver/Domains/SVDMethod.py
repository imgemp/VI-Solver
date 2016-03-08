import numpy as np
from scipy.sparse.linalg import svds
# from scipy.linalg import diagsvd

from VISolver.Domain import Domain


class SVDMethod(Domain):

    def __init__(self,Data,keepData=False,tau=1.,Dim=None):
        self.Data = self.load_data(Data)
        self.keepData = keepData
        self.tau = tau
        self.Dim = Dim
        self.last_F = np.inf

    def load_data(self,Data):
        self.mask = (Data != 0).toarray()
        self.fro = np.linalg.norm(self.mask*Data.toarray(),ord='fro')
        return Data.toarray()

    def F(self,parameters):
        R = self.shrink(parameters,self.tau)
        # grad = np.asarray(self.Data-R)*self.mask
        grad = (self.Data-R)*self.mask
        self.last_F = grad
        return grad

    def shrink(self,x,tau,k=125):
        U, S, Vt = svds(x,k=k)
        # U, S, Vt = np.linalg.svd(x,full_matrices=False)
        # U, S, Vt = np.linalg.svd(x)
        s = np.clip(S-tau,0.,np.inf)
        R = U.dot(np.diag(s)).dot(Vt)
        # R = U.dot(diagsvd(s,U.shape[1],Vt.shape[0])).dot(Vt)
        return R

    def rel_error(self,parameters):
        err = np.linalg.norm(self.last_F,ord='fro')/self.fro
        print(err)
        return err
