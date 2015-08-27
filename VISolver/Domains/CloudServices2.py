from __future__ import division
import numpy as np
from VISolver.Domain import Domain
from numpy.polynomial.polynomial import polyval


class CloudServices(Domain):

    def __init__(self,Network,poly_splice=True,gap_alpha=2):
        self.UnpackNetwork(*Network)
        self.Network = (Network[0],Network[1])
        self.Dim = self.CalculateNetworkSize()
        self.poly_splice = poly_splice
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
        self.coeff = np.array([3520,-4752,2564,-691,93,-5])
        self.t0 = 3
        self.tf = 4

    def CalculateNetworkSize(self):
        return 2*self.nClouds

    def Demand_IJ(self,Data):
        p = Data[:self.nClouds]
        q = Data[self.nClouds:]
        relprice = p/np.mean(p)
        relquali = q/np.mean(q)

        t = self.pref_bizes*p*q*relprice*relquali

        exp = self.H*np.exp(-t**2)

        if self.poly_splice:
            poly = self.H*np.exp(-9)*polyval(t,self.coeff)

            texp = (t <= self.t0)
            tpoly = np.logical_and(t > self.t0,t < self.tf)

            Qij = exp*texp + poly*tpoly
        else:
            Qij = exp

        return Qij, p, q, t

    def Demand(self,Data):
        Qij, p, q = self.Demand_IJ(Data)[:3]
        return np.sum(Qij,axis=0), p, q

    def CloudProfits(self,Data):
        Q, p, q = self.Demand(Data)[:3]
        Revenue = p*Q
        Cost = self.c_clouds*Q*q**(-2)
        return Revenue - Cost

    def dCloudProfits(self,Data):
        Qij, p, q, t = self.Demand_IJ(Data)

        ps = np.sum(p)
        qs = np.sum(q)

        c = self.c_clouds
        x = (p-c/(q**2))
        a = 2*c/(q**3)

        dQ_dt_exp = -2*t*Qij

        if self.poly_splice:
            coeff_d1 = self.coeff[1:]*np.arange(1,len(self.coeff))
            dQ_dt_poly = self.H*np.exp(-9)*polyval(t,coeff_d1)

            texp = (t <= self.t0)
            tpoly = np.logical_and(t > self.t0,t < self.tf)

            dQ_dt = dQ_dt_exp*texp + dQ_dt_poly*tpoly
        else:
            dQ_dt = dQ_dt_exp

        dt_dpj = t*(2/p-1/ps)
        dt_dqj = t*(2/q-1/qs)

        dpj = np.sum(Qij+x*(dQ_dt*dt_dpj),axis=0)
        dqj = np.sum(a*Qij+x*(dQ_dt*dt_dqj),axis=0)

        return np.hstack([dpj,dqj])

    def Jac(self,Data):
        Qij, p, q, t = self.Demand_IJ(Data)

        ps = np.sum(p)
        qs = np.sum(q)

        c = self.c_clouds
        x = (p-c/(q**2))
        a = 2*c/(q**3)

        dQ_dt_exp = -2*t*Qij
        d2Q_dt2_exp = -2*(1-2*t**2)*Qij

        if self.poly_splice:
            coeff_d1 = self.coeff[1:]*np.arange(1,len(self.coeff))
            dQ_dt_poly = self.H*np.exp(-9)*polyval(t,coeff_d1)
            coeff_d2 = coeff_d1[1:]*np.arange(1,len(coeff_d1))
            d2Q_dt2_poly = self.H*np.exp(-9)*polyval(t,coeff_d2)

            texp = (t <= self.t0)
            tpoly = np.logical_and(t > self.t0,t < self.tf)

            dQ_dt = dQ_dt_exp*texp + dQ_dt_poly*tpoly
            d2Q_dt2 = d2Q_dt2_exp*texp + d2Q_dt2_poly*tpoly
        else:
            dQ_dt = dQ_dt_exp
            d2Q_dt2 = d2Q_dt2_exp

        dt_dpj = t*(2/p-1/ps)
        dt_dqj = t*(2/q-1/qs)
        dt_dpk = t*(-1/ps)
        dt_dqk = t*(-1/qs)

        d2t_dpj2 = 2*t*(1/(p**2)+1/(ps**2)-2/(p*ps))
        d2t_dqj2 = 2*t*(1/(q**2)+1/(qs**2)-2/(q*qs))

        d2t_dpjdqj = t*(2/p-1/ps)*(2/q-1/qs)

        d2t_dpjdpk = 2*t*(1/(ps**2)-1/(p*ps))
        d2t_dpjdqk = t*(2/p-1/ps)*(-1/qs)

        d2t_dqjdqk = 2*t*(1/(qs**2)-1/(q*qs))
        d2t_dqjdpk = t*(2/q-1/qs)*(-1/ps)

        dpj2 = 2*dQ_dt*dt_dpj+x*(dQ_dt*d2t_dpj2+d2Q_dt2*dt_dpj**2)
        dqj2 = a*(2*dQ_dt*dt_dqj-3/q*Qij)+x*(dQ_dt*d2t_dqj2+d2Q_dt2*dt_dqj**2)

        dpjdqj = dQ_dt*(dt_dqj+a*dt_dpj) +\
            x*(dQ_dt*d2t_dpjdqj+d2Q_dt2*dt_dpj*dt_dqj)

        dpjdpk = dQ_dt*dt_dpk+x*(dQ_dt*d2t_dpjdpk+d2Q_dt2*dt_dpj*dt_dpk)
        dpjdqk = dQ_dt*dt_dqk+x*(dQ_dt*d2t_dpjdqk+d2Q_dt2*dt_dpj*dt_dqk)

        dqjdqk = a*dQ_dt*dt_dqk+x*(dQ_dt*d2t_dqjdqk+d2Q_dt2*dt_dqj*dt_dqk)
        dqjdpk = a*dQ_dt*dt_dpk+x*(dQ_dt*d2t_dqjdpk+d2Q_dt2*dt_dqj*dt_dpk)

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
