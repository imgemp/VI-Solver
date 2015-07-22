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
        return np.exp(-market**2.)*self.H
        # return 1.10**(-market**2.)*self.H

    def Demand(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        return np.sum(np.exp(-market**2.)*self.H,axis=0)
        # return np.sum(1.10**(-market**2.)*self.H,axis=0)

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
        Qij = np.exp(-market**2.)*self.H
        # Qij = 1.10**(-market**2.)*self.H

        pfac = (2/p-1/np.sum(p))*2
        qfac = (2/q-1/np.sum(q))*2
        dfpj = -market**2*pfac
        dfqj = -market**2*qfac
        # dfpj = -market**2*pfac*np.log(1.10)
        # dfqj = -market**2*qfac*np.log(1.10)

        c = self.c_clouds

        dpj = np.sum(Qij*(1+dfpj*(p-c*q**(-2))),axis=0)
        dqj = np.sum(Qij*(2*c*q**(-3)+dfqj*(p-c*q**(-2))),axis=0)

        return np.hstack([dpj,dqj])

    def Hess(self,Data):

        # only for base e right now

        p = Data[:self.nClouds]
        q = Data[self.nClouds:]

        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        fij = -market**2

        Qij = np.exp(fij)*self.H

        ps = np.sum(p)
        qs = np.sum(q)

        dfij_dpj = 2*fij*(2/p-1/ps)
        dfij_dqj = 2*fij*(2/q-1/qs)
        dfij_dpk = 2*fij*(-1/ps)  # same for every k
        dfij_dqk = 2*fij*(-1/qs)  # same for every k

        # maybe the problem is in these next 3
        d2fij_dpj2 = 2*fij*(6/(p**2)-8/(p*ps)+3/(ps**2))
        d2fij_dqj2 = 2*fij*(6/(q**2)-8/(q*qs)+3/(qs**2))

        d2fij_dpjdqj = 2*fij*2*(2/q-1/qs)*(2/p-1/ps)

        d2fij_dpjdpk = 2*fij*(-4/(p*ps)+3/(ps**2))
        d2fij_dpjdqk = 2*fij*2*(-1/qs)*(2/p-1/ps)

        d2fij_dqjdqk = 2*fij*(-4/(q*qs)+3/(qs**2))
        d2fij_dqjdpk = 2*fij*2*(-1/ps)*(2/q-1/qs)

        c = self.c_clouds
        x = (p-c/(q**2))
        a = 2*c/(q**3)

        dpjdpk = Qij*(dfij_dpk+(d2fij_dpjdpk+dfij_dpj*dfij_dpk)*x)
        dpjdqk = Qij*(dfij_dqk+(d2fij_dpjdqk+dfij_dpj*dfij_dqk)*x)

        dqjdpk = Qij*(a*dfij_dpk+(d2fij_dqjdpk+dfij_dqj*dfij_dpk)*x)
        dqjdqk = Qij*(a*dfij_dqk+(d2fij_dqjdqk+dfij_dqj*dfij_dqk)*x)

        # the next 3 have errors somewhere
        dpj2 = Qij*(2*dfij_dpj+(d2fij_dpj2+dfij_dpj**2)*x)
        dqj2 = Qij*(2*a*dfij_dqj-3*a/q+(d2fij_dqj2+dfij_dqj**2)*x)

        dpjdqj = Qij*(a*dfij_dpj+dfij_dqj+(d2fij_dpjdqj+dfij_dpj*dfij_dqj)*x)

        # sum over i (rows) to get derivatives

        nc = self.nClouds
        Hessian = np.zeros((2*nc,2*nc))
        Hessian[:nc,:nc] = np.sum(dpjdpk,axis=0)[:,None]
        Hessian[:nc,nc:] = np.sum(dpjdqk,axis=0)[:,None]

        Hessian[nc:,:nc] = np.sum(dqjdpk,axis=0)[:,None]
        Hessian[nc:,nc:] = np.sum(dqjdqk,axis=0)[:,None]

        np.fill_diagonal(Hessian[:nc,:nc],np.sum(dpj2,axis=0))
        np.fill_diagonal(Hessian[:nc,nc:],np.sum(dpjdqj,axis=0))

        np.fill_diagonal(Hessian[nc:,:nc],np.sum(dpjdqj,axis=0))
        np.fill_diagonal(Hessian[nc:,nc:],np.sum(dqj2,axis=0))

        return Hessian

    def approx_jacobian(self,x,epsilon=np.sqrt(np.finfo(float).eps),*args):
        """Approximate the Jacobian matrix of callable function func
           * Rob Falck's implementation as a part of scipy.optimize.fmin_slsqp
           * Parameters
             x       - The state vector at which the Jacobian matrix is
             desired
             func    - A vector-valued function of the form f(x,*args)
             epsilon - The peturbation used to determine the partial derivatives
             *args   - Additional arguments passed to func

           * Returns
             An array of dimensions (lenf, lenx) where lenf is the length
             of the outputs of func, and lenx is the number of

           * Notes
             The approximation is done using forward differences

        """
        func = self.F
        x0 = np.asfarray(x)
        f0 = func(*((x0,)+args))
        jac = np.zeros([len(x0),len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()

    def eigstats(self,Data):

        hess = self.Hess(Data)
        eigs_r = np.real(np.linalg.eigvals(hess))
        print('min: %g, max: %g' % (min(eigs_r),max(eigs_r)))


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
