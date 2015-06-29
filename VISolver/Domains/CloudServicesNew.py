from __future__ import division
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from IPython import embed
from VISolver.Domain import Domain

# import warnings
# warnings.filterwarnings('error')


class CloudServices(Domain):

    def __init__(self,Network,alpha=2):
        # raise NotImplementedError('Under construction.')
        self.UnpackNetwork(*Network)
        self.Network = (Network[0],Network[1])
        self.Dim = self.CalculateNetworkSize()
        self.alpha = alpha

    def F(self,Data):
        return -self.dCloudProfits(Data)

    def gap_rplus(self,Data):
        X = Data
        # print('-------------error here--------------------------')
        dFdX = self.F(Data)
        # print('-'*100)

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
        self.q = np.zeros((self.nBiz,2*self.nClouds))

    def CalculateNetworkSize(self):
        return 2*self.nClouds

    def CloudProfits(self,Data):
        half = len(Data)//2

        p_L = Data[:half]
        p_S = Data[half:]

        q = self.argmax_firm_profits(Data)
        self.q = q

        Q_L = np.sum(q[:,:half],axis=0)
        Q_S = np.sum(q[:,half:],axis=0)
        Q = Q_L + Q_S

        Revenue = p_L*Q_L + p_S*Q_S

        Cost = np.zeros(self.nClouds)
        for i in xrange(self.nClouds):
            # Cost[i] = self.exp2lin(Q[i],*self.c_clouds[i])
            # Cost[i] = self.poly(Q[i],*self.c_clouds[i])
            a = 0.3
            QL_cost = a*self.poly(Q_L[i],*self.c_clouds[i])
            c_s = 1.5*self.c_clouds[i][0]
            d_s = self.c_clouds[i][1]
            QS_cost = a*self.poly(Q_S[i],c_s,d_s)
            Cost[i] = QL_cost + QS_cost
        # embed()
        # assert False
        return Revenue - Cost

    def CloudProfit(self,Data,i):
        p_L = Data[i]
        p_S = Data[i+self.nClouds]

        q = self.argmax_firm_profits(Data)
        if (q < 0.).any():
            embed()
            assert False
        self.q = q

        Q_L = np.sum(q[:,i])
        Q_S = np.sum(q[:,i+self.nClouds])
        Q = Q_L + Q_S

        Revenue = p_L*Q_L + p_S*Q_S

        # Cost = self.exp2lin(Q,*self.c_clouds[i])
        # Cost = self.poly(Q,*self.c_clouds[i])
        # print('in cloudprofit')
        QL_cost = self.poly(Q_L,*self.c_clouds[i])
        if Q_L < 0.:
            embed()
            assert False
        # print('error above?')
        c_s = 1.5*self.c_clouds[i][0]
        d_s = self.c_clouds[i][1]
        QS_cost = self.poly(Q_S,c_s,d_s)
        # print('error above now?')
        Cost = QL_cost + QS_cost

        return Revenue - Cost

    def dCloudProfits(self,Data):
        delta = 1e-5
        findiff = np.zeros_like(Data)
        pert = np.zeros_like(Data)
        pert[-1] = delta
        for i in xrange(Data.shape[0]):
            # print(i)
            pert = np.roll(pert,1)
            f = lambda x: self.CloudProfit(x,i % self.nClouds)
            findiff[i] = self.forwdiff(f,Data,Data+pert,delta)
        return findiff

    def nngauss_pdf(self,x,mu,sigma):
        if x >= 0.:
            N = 2./(1.-erf(-mu/(sigma*np.sqrt(2.)))) * \
                1./(sigma*np.sqrt(2.*np.pi))
            gauss = np.exp(-(x-mu)**2./(2.*sigma**2.))
            return N*gauss
        else:
            return 0.

    def nngauss_cdf(self,x,mu,sigma):
        if x >= 0.:
            a = erf((x-mu)/(sigma*np.sqrt(2.)))
            b = erf(-mu/(sigma*np.sqrt(2.)))
            return 1./2.*(a-b)
        else:
            return 0.

    def nngauss_intcdf(self,x,mu,sigma):
        if x >= 0.:
            xi = -mu/(sigma*np.sqrt(2.))
            eta = xi*erf(xi)
            kappa = 1./np.sqrt(np.pi)*np.exp(-xi**2.)
            _x = (x-mu)/(sigma*np.sqrt(2.))
            return 1./2.*(_x*erf(_x) +
                          1./np.sqrt(np.pi)*np.exp(-_x**2.) -
                          eta - kappa - eta/xi*x)
        else:
            return 0.

    def nngauss_mean(self,mu,sigma):
        skew = 2./(1.-erf(-mu/(sigma*np.sqrt(2.)))) * sigma/np.sqrt(2.*np.pi)
        return mu + skew

    def exp2lin(self,x,a,b,c):
        if x <= 0.:
            return 0.
        else:
        # elif x <= b:
            return 1./c*(np.exp(x/a)-1.)
        # else:
        #     return 1./c*(np.exp(b/a)*(x/a+1-b/a)-1)

    def dexp2lin(self,x,a,b,c):
        if x <= 0.:
            return 0.
        elif x <= b:
            return 1./(a*c)*np.exp(x/a)
        else:
            return 1./(a*c)*np.exp(b/a)

    def poly(self,x,c,d):
        # c and d should also be 1-d, else should match x shape maybe?
        # embed()
        assert c.shape[-1] == d.shape[-1]  # == x.shape[-1]
        # try:
        #     np.sum(c*x**d,axis=-1)
        # except Warning:
        #     embed()
        return np.sum(c*x**d,axis=-1)

    def dpoly(self,x,c,d):
        # c and d should also be 1-d, else should match x shape maybe?
        assert c.shape[-1] == d.shape[-1]  # == x.shape[-1]
        _c = c[d != 0.]
        _d = d[d != 0.]
        return np.sum(_d*_c*x**(_d-1),axis=-1)

    def firm_profit(self,q,dp,lam,mu,intcdf,c):
        half = len(q)//2
        Ql = np.sum(q[:half])
        cst = c(Ql)
        return np.sum(q*dp) + lam[1]*(Ql-mu) - (lam[0]+lam[1])*intcdf(Ql) - cst

    def dfirm_profit(self,q,dp,lam,cdf,dc):
        half = len(q)//2
        Ql = np.sum(q[:half])
        res = np.empty_like(dp)
        res[:half] = dp[:half] + lam[1] - (lam[0]+lam[1])*cdf(Ql) - dc(Ql)
        res[half:] = dp[half:]
        return res

    def forwdiff(self,f,x,_x,h):
        return (f(_x)-f(x))/h

    def maxfirm_profits(self,Data):
        res = np.empty(self.nBiz)
        for j in xrange(self.nBiz):
            res[j] = self.maxfirm_profit(Data,j)
        return res

    def maxfirm_profit(self,Data,j):
        x0 = self.q[j]
        mu, sigma = self.dist_bizes[j]
        lam = self.lam_bizes[j]
        dp = self.p_bizes[j] - Data

        intcdf = lambda x: self.nngauss_intcdf(x,mu,sigma)
        cost = lambda x: self.poly(x,*self.c_bizes[j])
        fun = lambda q: -self.firm_profit(q,dp,lam,mu,intcdf,cost)

        cdf = lambda x: self.nngauss_cdf(x,mu,sigma)
        dcost = lambda x: self.dpoly(x,*self.c_bizes[j])
        dfun = lambda q: -self.dfirm_profit(q,dp,lam,cdf,dcost)

        bnds = tuple([(0,None)]*len(x0))

        res = minimize(fun,x0,jac=dfun,method='SLSQP',bounds=bnds)

        if -res.fun < 0.:
            return 0.
        else:
            return -res.fun

    def argmax_firm_profits(self,Data):
        q = np.zeros_like(self.q)
        for j in xrange(self.nBiz):
            # print(j)
            q[j] = self.argmax_firm_profit(Data,j)
            if (q[j] < 0.).any():
                embed()
                assert False
        if (q < 0.).any():
            embed()
            assert False
        return q

    def argmax_firm_profit(self,Data,j):
        x0 = self.q[j]
        mu, sigma = self.dist_bizes[j]
        lam = self.lam_bizes[j]
        dp = self.p_bizes[j] - Data

        intcdf = lambda x: self.nngauss_intcdf(x,mu,sigma)
        cost = lambda x: self.poly(x,*self.c_bizes[j])
        fun = lambda q: -self.firm_profit(q,dp,lam,mu,intcdf,cost)

        cdf = lambda x: self.nngauss_cdf(x,mu,sigma)
        dcost = lambda x: self.dpoly(x,*self.c_bizes[j])
        dfun = lambda q: -self.dfirm_profit(q,dp,lam,cdf,dcost)
        # embed()
        # assert False
        bnds = tuple([(0,None)]*len(x0))
        # print('start')
        # res = minimize(fun,x0,jac=dfun,method='SLSQP',bounds=bnds)
        res = minimize(fun,x0,jac=dfun,method='L-BFGS-B',bounds=bnds)
        # print('finish')
        if (res.x < 0.).any():
            embed()
            assert False
        if -res.fun < 0.:
            return 0.*res.x
        else:
            return res.x


def CreateRandomNetwork(nClouds=2,nBiz=10,seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Cloud cost function parameters
    # a: small --> large means large --> small linear slope
    # b: transition point from exp to lin
    # c: scaling factor for costs
    # c_clouds = 0*(np.random.rand(nClouds,3)*[0,.5,.2]+[1,1,1])
    # c_clouds = np.array([[21.,23.,1.],
    #                      [27.,43.,1.],
    #                      [32.,68.,1.]])
    # c_clouds = np.array([[[0.,4.5],[0.,2.]],
    #                      [[0.,2.0],[0.,2.2]],
    #                      [[0.,3.0],[0.,2.1]]])
    c_clouds = np.array([[[0.,4.5],[0.,2.]],
                         [[0.,2.0],[0.,2.2]]])

    # Business cost functions
    # Need to add a business cost function
    # c_bizes = np.random.rand(nBiz,3)*[0,0.2,0.1]+[0,0,0]
    # c_bizes = np.zeros((nBiz,3))
    c_bizes = np.array([[[0.,4.5],[0.,2.]],
                        [[0.,2.0],[0.,2.2]],
                        [[0.,3.0],[0.,2.1]],
                        [[0.,4.5],[0.,2.]],
                        [[0.,2.0],[0.,2.2]],
                        [[0.,3.0],[0.,2.1]],
                        [[0.,4.5],[0.,2.]],
                        [[0.,2.0],[0.,2.2]],
                        [[0.,3.0],[0.,2.1]],
                        [[0.,4.5],[0.,2.]]])

    # Business demand distribution function means, mu_biz, and
    # standard deviations, sigma_biz
    # dist_bizes = np.random.rand(nBiz,2)*[0.2,0.02]+[2,.8]
    dist_bizes = np.array([[10.,2.],
                           [15.,3.],
                           [18.,5.],
                           [20.,4.],
                           [23.,7.],
                           [26.,6.],
                           [30.,4.],
                           [13.,5.],
                           [17.,2.],
                           [24.,5.]])
    # mu = dist_bizes[:,0]
    # sigma = dist_bizes[:,1]
    # means = np.exp(mu+0.5*sigma**2)
    # variances = (np.exp(sigma**2)-1)*(means**2)

    # Business forecasting cost functions
    # = lam_biz[0]*E_surplus + lam_biz[1]*E_shortage
    # lam_bizes = np.random.rand(nBiz,2)*[0,0.01]+[0,0.005]
    lam_bizes = 1e2*np.array([[.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1],
                              [.1,.1]])

    # Business sale prices, p_biz
    # p_bizes = np.random.rand(nBiz)*1+5
    p_bizes = np.array([.3,.4,.2,.5,.6,.4,.3,.2,.5,.3])*10.

    return (nClouds,nBiz,c_clouds,c_bizes,dist_bizes,lam_bizes,p_bizes)
