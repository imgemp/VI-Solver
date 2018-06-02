import numpy as np
from VISolver.Domain import Domain


class AverageDomains(Domain):
    def __init__(self,Domains,weights=None):
        self.Domains = Domains
        self.Dim = Domains[0].Dim
        assert np.all(Domain.Dim == self.Dim for Domain in Domains)
        if weights is None:
            self.weights = np.ones(len(Domains))/len(Domains)
        else:
            assert weights.shape == (len(Domains),)
            self.weights = weights
        # this part isn't general but works for SOI
        self.alpha = Domains[0].alpha

    def F(self,Data):
        Fs = [Domain.F(Data) for Domain in self.Domains]
        return np.sum([self.weights[idx]*F for idx,F in enumerate(Fs)],axis=0)

    # this part isn't general but works for SOI
    def gap_rplus(self, X):
        dFdX = self.F(X)

        Y = np.maximum(0,X - dFdX/self.alpha)
        Z = X - Y

        return np.dot(dFdX,Z) - self.alpha/2.*np.dot(Z,Z)
