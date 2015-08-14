from __future__ import division
import numpy as np
from VISolver.Domain import Domain


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

    def valid(self,X):
        return not any(np.isnan(X) or np.isinf(X))

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
        return self.H*np.exp(-market**2)

    def Demand(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        return np.sum(self.H*np.exp(-market**2),axis=0)

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
        Qij = self.H*np.exp(-market**2)

        pfac = (2/p-1/np.sum(p))*2
        qfac = (2/q-1/np.sum(q))*2
        dfpj = -market**2*pfac
        dfqj = -market**2*qfac

        c = self.c_clouds

        dpj = np.sum(Qij*(1+dfpj*(p-c*q**(-2))),axis=0)
        dqj = np.sum(Qij*(2*c*q**(-3)+dfqj*(p-c*q**(-2))),axis=0)

        return np.hstack([dpj,dqj])

    def Jac(self,Data):

        p = Data[:self.nClouds]
        q = Data[self.nClouds:]

        relprice = p/np.mean(p)
        relquali = q/np.mean(q)
        supply = p*q*relprice*relquali
        market = self.pref_bizes*supply
        fij = -market**2

        Qij = self.H*np.exp(fij)

        ps = np.sum(p)
        qs = np.sum(q)

        fij = fij

        dfij_dpj = 2*fij*(2/p-1/ps)
        dfij_dqj = 2*fij*(2/q-1/qs)
        dfij_dpk = 2*fij*(-1/ps)  # same for every k
        dfij_dqk = 2*fij*(-1/qs)  # same for every k

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

        dpj2 = Qij*(2*dfij_dpj+(d2fij_dpj2+dfij_dpj**2)*x)
        dqj2 = Qij*(2*a*dfij_dqj-3*a/q+(d2fij_dqj2+dfij_dqj**2)*x)

        dpjdqj = Qij*(a*dfij_dpj+dfij_dqj+(d2fij_dpjdqj+dfij_dpj*dfij_dqj)*x)

        nc = self.nClouds
        Jacobian = np.zeros((2*nc,2*nc))
        Jacobian[:nc,:nc] = np.sum(dpjdpk,axis=0)[:,None]
        Jacobian[:nc,nc:] = np.sum(dpjdqk,axis=0)[:,None]

        Jacobian[nc:,:nc] = np.sum(dqjdpk,axis=0)[:,None]
        Jacobian[nc:,nc:] = np.sum(dqjdqk,axis=0)[:,None]

        np.fill_diagonal(Jacobian[:nc,:nc],np.sum(dpj2,axis=0))
        np.fill_diagonal(Jacobian[:nc,nc:],np.sum(dpjdqj,axis=0))

        np.fill_diagonal(Jacobian[nc:,:nc],np.sum(dpjdqj,axis=0))
        np.fill_diagonal(Jacobian[nc:,nc:],np.sum(dqj2,axis=0))

        return -Jacobian

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

    def eig_stats(self,Data):

        jac = self.Jac(Data)
        eigs = np.real_if_close(np.linalg.eigvals(jac))
        eigs_r = np.real(eigs)
        eigs_i = np.imag(eigs)

        eig_min = min(eigs_r)
        eig_max = max(eigs_r)
        N_real = sum(np.abs(eigs_i) == 0.)
        N_imag = len(eigs) - N_real
        N_neg = sum(eigs_r < 0.)
        N_zer = sum(eigs_r == 0.)
        N_pos = len(eigs) - N_neg - N_zer
        div_trace = sum(eigs_r)

        return np.array([eig_min,eig_max,
                         N_real,N_imag,
                         N_neg,N_zer,N_pos,
                         div_trace])


def CreateNetworkExample(ex=1):
    '''
    Ex1: There are two clouds, one with higher costs than the other. The
    businesses prefer the cloud with the higher costs, maybe because it's
    greener (which is why it has higher costs).
    Ex2: There are 5 clouds - 3 of which are large providers with highly
    optimized servicing abilities, while the other 2 are newcomers to the
    market trying to fill a niche with higher cost green-tech.
    There are four businesses in the market:
        Biz 1: Big buyer loyal to clouds with 3 lowest cost functions
        Biz 2: Medium buyer with slight preference towards green-tech
        Biz 3: Small buyer prefers green-tech, not opposed to large corp though
        Biz 4: Big buyer loyal to single cloud, green is lesser of 2 evils
    Ex3: Same as Ex3 except Biz 1 has flipped its affinity so that it's now
        only loyal to cloud 5 (the cloud with the highest cost)
    *** Want to explore Ex2 in a bounded region, p in (eps,10) / q in (eps,2),
        to find BofA(s)
    '''

    if ex == 1:

        # Cloud cost function coefficients
        c_clouds = np.array([1,.75])

        # Business preferences
        pref_bizes = np.array([[.7,1],
                               [.7,1]])

        # Business scale factors
        H = np.array([[10,10],
                      [10,10]])

    elif ex == 2:

        # Cloud cost function coefficients
        c_clouds = np.array([1.05,1.1,.95,1.15,1.2])

        # Business preferences
        pref_bizes = np.array([[.27,.27,.27,.38,.38],
                               [.34,.34,.34,.31,.31],
                               [.33,.33,.33,.26,.26],
                               [.25,.40,.40,.34,.34]])

        # Business scale factors
        H = np.array([[11,11,11,11,11],
                      [9,9,9,9,9],
                      [6,6,6,6,6],
                      [12,12,12,12,12]])

    elif ex == 3:

        # Cloud cost function coefficients
        c_clouds = np.array([1.05,1.1,.95,1.15,1.2])

        # Business preferences
        pref_bizes = np.array([[.38,.38,.38,.38,.27],
                               [.34,.34,.34,.31,.31],
                               [.33,.33,.33,.26,.26],
                               [.25,.40,.40,.34,.34]])

        # Business scale factors
        H = np.array([[11,11,11,11,11],
                      [9,9,9,9,9],
                      [6,6,6,6,6],
                      [12,12,12,12,12]])

    else:
        raise NotImplementedError('There are only 3 predefined examples (1-3)')

    nClouds = pref_bizes.shape[1]
    nBiz = pref_bizes.shape[0]

    return (nClouds,nBiz,c_clouds,H,pref_bizes)


def CreateRandomNetwork(nClouds=2,nBiz=2,seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Cloud cost function coefficients
    c_clouds = np.random.rand(nClouds)+.5

    # Business preferences
    pref_bizes = np.random.rand(nBiz,nClouds)*.15+.2

    # Business scale factors
    H = np.tile(np.random.rand(nBiz)*10+5,(nClouds,1)).T

    return (nClouds,nBiz,c_clouds,H,pref_bizes)
