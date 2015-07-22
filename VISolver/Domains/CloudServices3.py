from __future__ import division
import numpy as np
# from IPython import embed
from VISolver.Domain import Domain

# import warnings
# warnings.filterwarnings('error')


class CloudServices(Domain):

    def __init__(self,Network,gap_alpha=2):
        self.UnpackNetwork(*Network)
        self.Network = (Network[0],Network[1])
        self.Dim = self.CalculateNetworkSize()
        self.gap_alpha = gap_alpha

    def F(self,Data):
        return -self.dCloudProfits(Data)

    def gap_rplus(self,X):
        dFdX = self.F(X)

        Y = np.maximum(0,X - dFdX/self.gap_alpha)
        Z = X - Y

        return np.dot(dFdX,Z) - self.gap_alpha/2.*np.dot(Z,Z)

    # Functions used to Initialize the Cloud Network and Calculate F

    def UnpackNetwork(self,nClouds,nBiz,c_clouds,H,pref_bizes):
        self.nClouds = nClouds
        self.nBiz = nBiz
        self.c_clouds = c_clouds
        self.H = H
        self.pref_bizes = pref_bizes

    def CalculateNetworkSize(self):
        return 2*self.nClouds

    def Demand_IJ(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        return 1.10**(-market**2.)*self.H

    def Demand(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        return np.sum(1.10**(-market**2.)*self.H,axis=0)

    def CloudProfits(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        Q = self.Demand(Data)
        Revenue = p*Q
        Cost = self.c_clouds*Q*q**(-2)
        return Revenue - Cost

    def dCloudProfits(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        Qij = 1.10**(-market**2.)*self.H

        pfac = (2/p-1/np.sum(p))*2
        qfac = (2/q-1/np.sum(q))*2
        dfpj = -market**2*pfac*np.log(1.10)
        dfqj = -market**2*qfac*np.log(1.10)

        c = self.c_clouds

        dpj = np.sum(Qij*(1+dfpj*(p-c*q**(-2))),axis=0)
        dqj = np.sum(Qij*(2*c*q**(-3)+dfqj*(p-c*q**(-2))),axis=0)

        return np.hstack([dpj,dqj])


def CreateNetworkExample():

    # Cloud cost function coefficients
    c_clouds = np.array([1,.75])

    # Business preferences
    pref_bizes = np.array([[.7,1],
                           [.7,1]])

    # Business scale factors
    H = np.array([[10,10],
                  [10,10]])

    return (2,2,c_clouds,H,pref_bizes)


def CreateRandomNetwork(nClouds=2,nBiz=2,seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Cloud cost function coefficients
    c_clouds = np.random.rand(nClouds)+.5

    # Business preferences
    pref_bizes = np.random.rand(nBiz,nClouds)+.5

    # Business scale factors
    H = np.tile(np.random.rand(nBiz)*10+5,(nClouds,1)).T

    return (nClouds,nBiz,c_clouds,H,pref_bizes)
