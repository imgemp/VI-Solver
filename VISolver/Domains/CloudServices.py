import numpy as np
from scipy.special import erf

from VISolver.Domain import Domain


class CloudServices(Domain):

    def __init__(self,Network,alpha=2):
        raise NotImplementedError('Under construction.')
        self.UnpackNetwork(*Network)
        self.Network = (Network[0],Network[1])
        self.Dim = self.CalculateNetworkSize()
        self.alpha = alpha

    def F(self,Data,FDs=None):
        return -np.concatenate((self.dCloudProfit(Data),self.dBizProfits(Data)))

    def gap_rplus(self,Data):
        X = Data
        dFdX = self.F(Data)

        Y = np.maximum(0,X - dFdX/self.alpha)
        Z = X - Y

        return np.dot(dFdX,Z) - self.alpha/2.*np.dot(Z,Z)

    # Functions used to Initialize the Cloud Network and Calculate F

    def UnpackNetwork(self,nClouds,nBiz,c_clouds,c_bizes,dist_bizes,
                      lam_bizes,p_bizes):
        self.nClouds = nClouds
        self.nBiz = nBiz
        self.c_clouds = c_clouds
        self.c_bizes = c_bizes
        self.dist_bizes = dist_bizes
        self.lam_bizes = lam_bizes
        self.p_bizes = p_bizes

    def CalculateNetworkSize(self):
        return 2*self.nClouds*(self.nBiz + 1)

    def CloudProfit(self,Data):
        Q_L = self.QL_Cloud(Data)
        Q_S = self.QS_Cloud(Data)
        p_L = self.pL_Cloud(Data)
        p_S = self.pS_Cloud(Data)
        Revenue = p_L*Q_L + p_S*Q_S
        Cost = self.Quadratic(self.c_clouds,p_L+p_S)
        return Revenue - Cost

    def dCloudProfit(self,Data):
        dCP = np.empty((2*self.nClouds,))
        dCP[0::2] = self.dCloudProfitdPL(Data)
        dCP[1::2] = self.dCloudProfitdPS(Data)
        return dCP

    def dCloudProfitdPL(self,Data):
        Q_L = self.QL_Cloud(Data)
        p_L = Data[0:2*self.nClouds:2]
        p_S = Data[1:2*self.nClouds:2]
        dCost = self.dQuadratic(self.c_clouds,p_L+p_S)
        return Q_L + p_L*self.dQLdpL - dCost

    def dCloudProfitdPS(self,Data):
        Q_S = self.QS_Cloud(Data)
        p_L = Data[0:2*self.nClouds:2]
        p_S = Data[1:2*self.nClouds:2]
        dCost = self.dQuadratic(self.c_clouds,p_L+p_S)
        return Q_S + p_S*self.dQSdpS - dCost

    def BizProfits(self,Data):
        q_L = self.qL_Biz(Data)
        q_S = self.qS_Biz(Data)
        Revenue = self.p_bizes*(q_L+q_S)
        Cloud_Costs = np.sum( Data[:2*self.nClouds] * np.reshape( Data[2*self.nClouds:], (self.nBiz,2*self.nClouds) ), axis=1)
        Operating_Costs = self.Quadratic(self.c_bizes,q_L+q_S)
        Forecast_Costs = self.lam_bizes[:,0]*self.E_shortage(self.dist_bizes,q_L+q_S) \
                       + self.lam_bizes[:,1]*self.E_surplus(self.dist_bizes,q_L+q_S)
        return Revenue - Cloud_Costs - Operating_Costs - Forecast_Costs

    def dBizProfits(self,Data):
        dBP = np.empty((2*self.nBiz*self.nClouds,))
        dBP[0::2] = self.dBizProfitsdQL(Data)
        dBP[1::2] = self.dBizProfitsdQS(Data)
        return dBP

    def dBizProfitsdQL(self,Data):
        q_L = self.qL_Biz(Data)
        q_S = self.qS_Biz(Data)
        dRevenue = np.repeat( self.p_bizes, self.nClouds )
        dCloud_Costs = np.tile( Data[0:2*self.nClouds:2], self.nBiz )
        dOperating_Costs = np.repeat( self.dQuadratic(self.c_bizes,q_L+q_S), self.nClouds )
        dForecast_Costs = np.repeat(
                                self.lam_bizes[:,0]*self.dE_shortage(self.dist_bizes,q_L+q_S)
                              + self.lam_bizes[:,1]*self.dE_surplus(self.dist_bizes,q_L)
                        , self.nClouds )
        return dRevenue - dCloud_Costs - dOperating_Costs - dForecast_Costs

    def dBizProfitsdQS(self,Data):
        q_L = self.qL_Biz(Data)
        q_S = self.qS_Biz(Data)
        dRevenue = np.repeat( self.p_bizes, self.nClouds )
        dCloud_Costs = np.tile( Data[1:2*self.nClouds:2], self.nBiz )
        dOperating_Costs = np.repeat( self.dQuadratic(self.c_bizes,q_L+q_S), self.nClouds )
        dForecast_Costs = np.repeat( self.lam_bizes[:,0]*self.dE_shortage(self.dist_bizes,q_L+q_S), self.nClouds )
        return dRevenue - dCloud_Costs - dOperating_Costs

    def E_shortage(self,dists,q_L):
        mu = dists[:,0]
        sigma = dists[:,1]
        try:
            out = np.empty(q_L.shape)
            zero = (q_L<=0.)
            out[zero] = np.exp(mu[zero]+0.5*sigma[zero]**2)
            out[~zero] = np.exp(mu[~zero]+0.5*sigma[~zero]**2)*self.Normal_CDF( (mu[~zero] + sigma[~zero]**2 - np.log(q_L[~zero]))/sigma[~zero] ) \
                       - q_L[~zero]*(1 - self.LogNormal_CDF(mu[~zero],sigma[~zero],q_L[~zero]))
            return out
        except AttributeError:
            if q_L<=0.:
                return np.exp(mu+0.5*sigma**2)
            else:
                return np.exp(mu+0.5*sigma**2)*self.Normal_CDF( (mu + sigma**2 - np.log(q_L))/sigma ) \
                     - q_L*(1 - self.LogNormal_CDF(mu,sigma,q_L))

    def E_surplus(self,dists,q_L):
        mu = dists[:,0]
        sigma = dists[:,1]
        try:
            out = np.empty(q_L.shape)
            zero = (q_L<=0.)
            out[zero] = 0.
            out[~zero] = q_L[~zero]*self.LogNormal_CDF(mu[~zero],sigma[~zero],q_L[~zero]) \
                       + np.exp(mu[~zero]+0.5*sigma[~zero]**2)*self.Normal_CDF( (mu[~zero] + sigma[~zero]**2 - np.log(q_L[~zero]))/sigma[~zero] ) \
                       - np.exp(mu[~zero]+0.5*sigma[~zero]**2)
            return out
        except AttributeError:
            if q_L<=0.:
                return 0.
            else:
                return q_L*self.LogNormal_CDF(mu,sigma,q_L) \
                     + np.exp(mu+0.5*sigma**2)*self.Normal_CDF( (mu + sigma**2 - np.log(q_L))/sigma ) \
                     - np.exp(mu+0.5*sigma**2)

    def dE_shortage(self,dists,q_L):
        return self.LogNormal_CDF(dists[:,0],dists[:,1],q_L) - 1

    def dE_surplus(self,dists,q_L):
        return self.LogNormal_CDF(dists[:,0],dists[:,1],q_L)

    def Normal_CDF(self,x):
        return 0.5* ( 1 + erf( x/np.sqrt(2) ) )

    def LogNormal_CDF(self,mu,sigma,x):
        try:
            out = np.zeros(x.shape)
            pos = (x>0.)
            out[pos] = 0.5*( 1 + erf( (np.log(x[pos]) - mu[pos])/(sigma[pos]*np.sqrt(2)) ) )
            return out
        except AttributeError:
            if x<=0.:
                return 0.
            else:
                return 0.5*( 1 + erf( (np.log(x) - mu)/(sigma*np.sqrt(2)) ) )

    def Quadratic(self,c,x):
        if len(c.shape) == 1:
            return c[0] + x*( c[1] + x*c[2] )
        else:
            return c[:,0] + x*( c[:,1] + x*c[:,2] )

    def dQuadratic(self,c,x):
        if len(c.shape) == 1:
            return c[1] + 2*x*c[2]
        else:
            return c[:,1] + 2*x*c[:,2]

    def QL_Cloud(self,Data):
        return np.array( [ np.sum(Data[2*(c+self.nClouds)::2*self.nClouds]) for c in xrange(self.nClouds) ] )

    def QS_Cloud(self,Data):
        return np.array( [ np.sum(Data[2*(c+self.nClouds)+1::2*self.nClouds]) for c in xrange(self.nClouds) ] )

    def pL_Cloud(self,Data):
        return Data[0:2*self.nClouds:2]

    def pS_Cloud(self,Data):
        return Data[1:2*self.nClouds:2]

    def qL_Biz(self,Data):
        return np.array( [ np.sum(Data[2*self.nClouds*(b+1):2*self.nClouds*(b+2):2]) for b in xrange(self.nBiz) ] )

    def qS_Biz(self,Data):
        return np.array( [ np.sum(Data[2*self.nClouds*(b+1)+1:2*self.nClouds*(b+2):2]) for b in xrange(self.nBiz) ] )

def CreateRandomNetwork(nClouds,nBiz,seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Cloud quadratic cost functions = c_cloud[0] + c_cloud[1]*Q + c_cloud[2]*Q**2
    c_clouds = 0*( np.random.rand(nClouds,3)*[0,.5,.2]+[1,1,1] )

    # Business quadratic cost functions = c_biz[0] + c_biz[1]*Q_biz + c_biz[2]*Q_biz**2
    c_bizes = np.random.rand(nBiz,3)*[0,0.2,0.1]+[0,0,0]

    # Business demand distribution function means, mu_biz, and standard deviations, sigma_biz
    dist_bizes = np.random.rand(nBiz,2)*[0.2,0.02]+[2,.8]
    # mu = dist_bizes[:,0]
    # sigma = dist_bizes[:,1]
    # means = np.exp(mu+0.5*sigma**2)
    # variances = (np.exp(sigma**2)-1)*(means**2)

    # Business forecasting cost functions = lam_biz[0]*E_shortage + lam_biz[1]*E_surplus
    lam_bizes = np.random.rand(nBiz,2)*[0,0.01]+[0,0.005]

    # Business sale prices, p_biz
    p_bizes = np.random.rand(nBiz)*1+5

    return (nClouds,nBiz,c_clouds,c_bizes,dist_bizes,lam_bizes,p_bizes)
