import numpy as np

from VISolver.Domain import Domain


class MatrixFactorization(Domain):

    def __init__(self,Data,sh_P,sh_Q,keepData=False):
        self.Data = self.load_data(Data)
        self.keepData = keepData
        self.sh_P = sh_P
        self.sh_Q = sh_Q
        self.split = np.prod(sh_P)
        self.Dim = np.prod(sh_P) + np.prod(sh_Q)

    def load_data(self,Data):
        self.mask = (Data != 0).toarray()
        return Data

    def unpack(self,parameters):
        P = parameters[:self.split].reshape(self.sh_P)
        Q = parameters[self.split:].reshape(self.sh_Q)
        return P,Q

    def predict(self,parameters):
        P,Q = self.unpack(parameters)
        return P.dot(Q.T)

    def rmse(self,pred,test,mask):
        sqerr = mask*np.asarray(pred - test)**2.
        return np.sqrt(sqerr.sum()/test.nnz)

    def F(self,parameters):
        P,Q = self.unpack(parameters)
        err = self.mask*np.asarray(self.Data - P.dot(Q.T))
        dP = err.dot(Q)
        dQ = err.T.dot(P)
        grad = np.hstack((dP.flatten(),dQ.flatten()))
        return grad
