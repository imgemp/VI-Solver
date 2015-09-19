import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm

from VISolver.Domain import Domain


class SOI(Domain):

    def __init__(self,Network,alpha=2):
        self.UnpackNetwork(Network)
        self.Network = (self.m,self.n,self.o)
        self.Dim = self.CalculateNetworkSize()
        self.alpha = alpha

        self.cmap = cm.rainbow
        norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
        self.to_rgba = [cm.ScalarMappable(norm=norm, cmap=self.cmap).to_rgba]*3

    def F(self,Data):
        return self.F_P2UP(Data)

    def gap_rplus(self, X):
        dFdX = self.F(X)

        Y = np.maximum(0,X - dFdX/self.alpha)
        Z = X - Y

        return np.dot(dFdX,Z) - self.alpha/2.*np.dot(Z,Z)

    # Functions Used to Animate and Save Network Run to Movie File

    def FlowNormalizeColormap(self,Data,cmap):
        self.cmap = cmap
        maxFlows = [0]*2
        for data in Data:
            x = self.UnpackData(data)
            xflows = list(self.PathFlow2LinkFlow_x2f(*x)[0])
            newFlows = [np.max(flow) for flow in xflows]
            maxFlows = np.max([maxFlows,newFlows],axis=0)
        norm = [mpl.colors.Normalize(vmin=0.,vmax=mxf) for mxf in maxFlows]
        self.to_rgba = \
            [cm.ScalarMappable(norm=n, cmap=self.cmap).to_rgba for n in norm]

    def InitVisual(self):

        ax = plt.gca()
        fig = plt.gcf()

        # Add Colorbar
        cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
        plt.axis('off')
        cb = mpl.colorbar.ColorbarBase(cax, cmap=self.cmap,
                                       spacing='proportional')
        cb.set_label('Internet Traffic (Q)')

        plt.sca(ax)

        # Create Network Skeleton
        mid = (max([self.m,self.n,self.o]) - 1.)/2.
        Ix = np.linspace(mid-(self.m-1.)/2.,
                         mid+(self.m-1.)/2.,self.m)
        Iy = 2.
        Jx = np.linspace(mid-(self.n-1.)/2.,
                         mid+(self.n-1.)/2.,self.n)
        Jy = 1.
        Kx = np.linspace(mid-(self.o-1.)/2.,
                         mid+(self.o-1.)/2.,self.o)
        Ky = 0.
        od = []
        for i in xrange(self.m):
            for j in xrange(self.n):
                od.append([(Ix[i],Iy),(Jx[j],Jy)])
        for j in xrange(self.n):
            for k in xrange(self.o):
                od.append([(Jx[j],Jy),(Kx[k],Ky)])
        lc = mc.LineCollection(od, colors=(0,0,0,0), linewidths=10)
        ax.add_collection(lc)
        ax.set_xlim((0,2*mid))
        ax.set_ylim((-.5,2.5))
        ax.add_collection(lc)

        # Annotate Plot
        plt.box('off')
        plt.yticks([0,1,2],['Demand\nMarkets', 'Network\nProviders',
                            'Service\nProviders'],
                   rotation=45)
        plt.xticks(Kx,['Market\n'+str(k+1) for k in xrange(self.o)])
        plt.tick_params(axis='y',right='off')
        plt.tick_params(axis='x',top='off')

        return ax.collections

    def UpdateVisual(self,num,ax,Frames,annotations):

        Data = Frames[num]

        # Check for Next Annotation
        if len(annotations) > 0:
            ann = annotations[-1]
            if num >= ann[0]:
                ann[1](ann[2])
                annotations.pop()

        # Unpack Data
        f_Q, f_q, f_Pi = self.PathFlow2LinkFlow_x2f(*self.UnpackData(Data))
        f_Q_top = f_Q[0]
        f_Q_bot = f_Q[1]

        colors = []
        for i in xrange(self.m):
            for j in xrange(self.n):
                colors.append(self.to_rgba[0](f_Q_top[i,j]))
        for j in xrange(self.n):
            for k in xrange(self.o):
                colors.append(self.to_rgba[1](f_Q_bot[j,k]))

        ax.collections[0].set_color(colors)

        return ax.collections

    # Functions used to Initialize the BloodBank Network and Calculate F

    def UnpackNetwork(self,Network):

        self.m,self.n,self.o,\
            self.coeff_f_Q,self.coeff_rho_Q,\
            self.coeff_rho_q,self.coeff_rho_const,\
            self.coeff_c_q_pow1,self.coeff_c_q_pow2,self.coeff_oc_Pi,\
            self.drhodQ_ind_I,self.dcdq_ind,\
            self.ind_IJ_I,self.ind_IJ_J,self.ind_JK_J,self.ind_JK_K = Network

    def UnpackData(self,Data):

        shp = (self.m,self.n,self.o)
        rng = xrange(0,self.Dim,self.Dim//3)
        return [np.reshape(Data[s:s+self.Dim//3],shp) for s in rng]

    def CalculateNetworkSize(self):

        return 3*self.m*self.n*self.o

    def PathFlow2LinkFlow_x2f(self,Q,q,Pi):

        slice_Q_top = Q[self.ind_IJ_I,self.ind_IJ_J,:]
        f_Q_top = np.reshape(np.sum(slice_Q_top,axis=1),(self.m,self.n))
        slice_Q_bot = Q[:,self.ind_JK_J,self.ind_JK_K]
        f_Q_bot = np.reshape(np.sum(slice_Q_bot,axis=0),(self.n,self.o))
        f_Q = [f_Q_top,f_Q_bot]

        slice_q_top = q[self.ind_IJ_I,self.ind_IJ_J,:]
        f_q_top = np.reshape(np.sum(slice_q_top,axis=1),(self.m,self.n))
        slice_q_bot = q[:,self.ind_JK_J,self.ind_JK_K]
        f_q_bot = np.reshape(np.sum(slice_q_bot,axis=0),(self.n,self.o))
        f_q = [f_q_top,f_q_bot]

        slice_Pi_top = Pi[self.ind_IJ_I,self.ind_IJ_J,:]
        f_Pi_top = np.reshape(np.sum(slice_Pi_top,axis=1),(self.m,self.n))
        slice_Pi_bot = Pi[:,self.ind_JK_J,self.ind_JK_K]
        f_Pi_bot = np.reshape(np.sum(slice_Pi_bot,axis=0),(self.n,self.o))
        f_Pi = [f_Pi_top,f_Pi_bot]

        return f_Q, f_q, f_Pi

    def F_P2UP(self,Data):

        # Unpack Data
        Q,q,Pi = self.UnpackData(Data)

        F_unpacked = self.FX_dX(Q,q,Pi)

        # Pack Data
        F_packed = np.array([])
        for Fx in F_unpacked:
            F_packed = np.append(F_packed,Fx.flatten())

        return F_packed

    def ProductionCost_f(self,Q):

        return self.coeff_f_Q*np.sum(Q,axis=(1,2))**2+np.sum(Q,axis=(1,2))

    def dProductionCostdQuantity_dfdQ(self,Q):

        shp = Q.shape[1:][::-1]+(1,)
        return np.tile(2.*self.coeff_f_Q*np.sum(Q,axis=(1,2))+1.,shp).T

    def DemandPrice_rho(self,Q,q):

        rho_Q_Q = self.coeff_rho_Q*np.resize(Q,self.coeff_rho_Q.shape)
        rho_Q_q = self.coeff_rho_q*q
        return np.sum(rho_Q_Q,axis=(3,4,5))+rho_Q_q+self.coeff_rho_const

    def dDemandPricedQuantity_drhodQ(self,Q,q):

        toorder = self.coeff_rho_Q[self.drhodQ_ind_I,:,:,self.drhodQ_ind_I,:,:]
        return np.swapaxes(np.swapaxes(toorder,1,3),2,4)

    def TransportationCost_c(self,Q,q):

        return self.coeff_c_q_pow2*(q**2) + self.coeff_c_q_pow1*q

    def dTransportationCostdQuality_dcdq(self,Q,q):

        dcdq = np.zeros(q.shape+(q.shape[0],q.shape[2]))
        dquad = 2.*self.coeff_c_q_pow2*q + self.coeff_c_q_pow1
        dcdq[self.dcdq_ind] = dquad.flatten()

        return dcdq

    def OpportunityCost_oc(self,Pi):

        return self.coeff_oc_Pi*(Pi**2)

    def dOpportunityCostdPrice_docdPi(self,Pi):

        return 2.*self.coeff_oc_Pi*Pi

    def ServiceProviderProfit(self,Q,q,Pi):

        f = self.ProductionCost_f(Q)
        rho = self.DemandPrice_rho(Q,q)

        return np.sum(rho*Q,axis=(1,2))-f-np.sum(Pi*Q,axis=(1,2))

    def dServiceProviderProfitdQuantity_dU1dQ(self,Q,q,Pi):

        rho = self.DemandPrice_rho(Q,q)
        drhodQ = self.dDemandPricedQuantity_drhodQ(Q,q)
        dfdQ = self.dProductionCostdQuantity_dfdQ(Q)

        Qrep = np.rollaxis(np.tile(Q,(self.n,self.o,1,1,1)),2,0)

        shp = (0+len(Q.shape),1+len(Q.shape))

        return rho+np.sum(drhodQ*Qrep,axis=shp)-Pi-dfdQ

    def NetworkProviderProfit(self,Q,q,Pi):

        c = self.TransportationCost_c(Q,q)
        oc = self.OpportunityCost_oc(Pi)

        return np.sum(Pi*Q,axis=(0,2))-np.sum(c+oc,axis=(0,2))

    def dNetworkProvderProfitdQuality_dU2dq(self,Q,q,Pi):

        dcdq = self.dTransportationCostdQuality_dcdq(Q,q)

        return -np.sum(dcdq,axis=(0+len(Q.shape),1+len(Q.shape)))

    def dNetworkProvderProfitdDemandPrice_dU2dPi(self,Q,q,Pi):

        docdPi = self.dOpportunityCostdPrice_docdPi(Pi)

        return Q-docdPi

    def FX_dX(self,Q,q,Pi):

        return [-self.dServiceProviderProfitdQuantity_dU1dQ(Q,q,Pi),
                -self.dNetworkProvderProfitdQuality_dU2dq(Q,q,Pi),
                -self.dNetworkProvderProfitdDemandPrice_dU2dPi(Q,q,Pi)]


def CreateNetworkExample(ex=1):

    m = 3
    n = 2
    o = 2

    coeff_f_Q = np.array([2.,1.,3.])

    coeff_rho_Q = np.zeros((3,2,2,3,2,2))
    coeff_rho_Q[0,0,0,0,0,0] = -1.
    coeff_rho_Q[0,0,0,0,0,1] = -.5
    coeff_rho_Q[0,0,1,0,0,1] = -2.
    coeff_rho_Q[0,0,1,0,0,0] = -1.
    coeff_rho_Q[0,1,0,0,1,0] = -2.
    coeff_rho_Q[0,1,0,0,0,0] = -.5
    coeff_rho_Q[0,1,1,0,1,1] = -3.
    coeff_rho_Q[0,1,1,0,0,1] = -1.
    coeff_rho_Q[1,0,0,1,0,0] = -1.
    coeff_rho_Q[1,0,0,1,0,1] = -.5
    coeff_rho_Q[1,0,1,1,0,1] = -3.
    coeff_rho_Q[1,1,0,1,1,0] = -2.
    coeff_rho_Q[1,1,0,1,1,1] = -1.
    coeff_rho_Q[1,1,1,1,1,1] = -3.
    coeff_rho_Q[1,1,1,0,1,0] = -1.
    coeff_rho_Q[2,0,0,2,0,0] = -4.
    coeff_rho_Q[2,0,1,2,0,1] = -2.
    coeff_rho_Q[2,0,1,2,1,0] = -1.
    coeff_rho_Q[2,1,0,2,1,0] = -3.
    coeff_rho_Q[2,1,0,2,0,0] = -1.
    coeff_rho_Q[2,1,1,2,1,1] = -4.

    coeff_rho_q = np.zeros((3,2,2))
    coeff_rho_q[0,0,0] = 1.
    coeff_rho_q[0,0,1] = 1.
    coeff_rho_q[0,1,0] = .5
    coeff_rho_q[0,1,1] = .5
    coeff_rho_q[1,0,0] = .3
    coeff_rho_q[1,0,1] = .8
    coeff_rho_q[1,1,0] = 1.
    coeff_rho_q[1,1,1] = 1.
    coeff_rho_q[2,0,0] = .5
    coeff_rho_q[2,0,1] = .3
    coeff_rho_q[2,1,0] = .2
    coeff_rho_q[2,1,1] = .7

    coeff_rho_const = np.zeros((3,2,2))
    coeff_rho_const[0,0,0] = 100.
    if ex == 2:
        coeff_rho_const[0,0,0] = 200.
    coeff_rho_const[0,0,1] = 200.
    coeff_rho_const[0,1,0] = 100.
    coeff_rho_const[0,1,1] = 150.
    coeff_rho_const[1,0,0] = 100.
    coeff_rho_const[1,0,1] = 200.
    coeff_rho_const[1,1,0] = 140.
    coeff_rho_const[1,1,1] = 300.
    coeff_rho_const[2,0,0] = 230.
    coeff_rho_const[2,0,1] = 150.
    coeff_rho_const[2,1,0] = 200.
    coeff_rho_const[2,1,1] = 300.

    coeff_c_q_pow2 = np.zeros((3,2,2))
    coeff_c_q_pow2[0,0,0] = 1.
    coeff_c_q_pow2[0,0,1] = .5
    coeff_c_q_pow2[0,1,0] = .1
    coeff_c_q_pow2[0,1,1] = 1.
    coeff_c_q_pow2[1,0,0] = .1
    coeff_c_q_pow2[1,0,1] = 1.
    coeff_c_q_pow2[1,1,0] = 2.
    coeff_c_q_pow2[1,1,1] = .5
    coeff_c_q_pow2[2,0,0] = 1.
    coeff_c_q_pow2[2,0,1] = .5
    coeff_c_q_pow2[2,1,0] = 1.
    coeff_c_q_pow2[2,1,1] = 2.

    coeff_c_q_pow1 = np.zeros((3,2,2))
    coeff_c_q_pow1[0,0,0] = -.5
    coeff_c_q_pow1[0,0,1] = -1.
    coeff_c_q_pow1[0,1,0] = -1.
    coeff_c_q_pow1[0,1,1] = 0.
    coeff_c_q_pow1[1,0,0] = -1.
    coeff_c_q_pow1[1,0,1] = -.5
    coeff_c_q_pow1[1,1,0] = 0.
    coeff_c_q_pow1[1,1,1] = -1.
    coeff_c_q_pow1[2,0,0] = -1.
    coeff_c_q_pow1[2,0,1] = -1.
    coeff_c_q_pow1[2,1,0] = -1.
    coeff_c_q_pow1[2,1,1] = -2.

    coeff_oc_Pi = np.zeros((3,2,2))
    coeff_oc_Pi[0,0,0] = 2.
    coeff_oc_Pi[0,0,1] = 2.
    coeff_oc_Pi[0,1,0] = 1.
    coeff_oc_Pi[0,1,1] = .5
    coeff_oc_Pi[1,0,0] = 1.
    coeff_oc_Pi[1,0,1] = .5
    coeff_oc_Pi[1,1,0] = 2.
    coeff_oc_Pi[1,1,1] = 1.5
    coeff_oc_Pi[2,0,0] = 1.
    coeff_oc_Pi[2,0,1] = 2.5
    coeff_oc_Pi[2,1,0] = 1.5
    coeff_oc_Pi[2,1,1] = 1.

    #Helper Arguments

    drhodQ_ind_I = np.arange(m)

    I = np.tile(np.arange(m),(n*o,1)).T.flatten()
    J = np.tile(np.tile(np.arange(n),(o,1)).T.flatten(),m)
    K = np.tile(np.arange(o),m*n)
    dcdq_ind = (I,J,K,I,K)

    ind_IJ_I = np.tile(np.arange(m),(n,1)).T.flatten()
    ind_IJ_J = np.tile(np.arange(n),m)
    ind_JK_J = np.tile(np.arange(n),(o,1)).T.flatten()
    ind_JK_K = np.tile(np.arange(o),n)

    return [m,n,o,
            coeff_f_Q,coeff_rho_Q,coeff_rho_q,coeff_rho_const,
            coeff_c_q_pow1,coeff_c_q_pow2,coeff_oc_Pi,
            drhodQ_ind_I,dcdq_ind,
            ind_IJ_I,ind_IJ_J,ind_JK_J,ind_JK_K]


def CreateRandomNetwork(m,n,o,seed):

    np.random.seed(seed)

    coeff_f_Q = 3.*np.random.rand(m)

    coeff_rho_Q = np.zeros((m,n,o,m,n,o))
    for i in xrange(m):
        for j in xrange(n):
            for k in xrange(o):
                #self dependence
                coeff_rho_Q[i,j,k,i,j,k] = -5.*np.random.rand()
                #external dependence
                ext = np.random.randint(0,m*n*o-1)
                if ext >= (i*n*o + j*o + k):
                    ext += 1
                r = divmod(ext,n*o)
                ext_i = r[0]
                r = divmod(r[1],o)
                ext_j = r[0]
                ext_k = r[1]
                coeff_rho_Q[i,j,k,ext_i,ext_j,ext_k] = -2.5*np.random.rand()
    coeff_rho_q = np.random.rand(m,n,o)
    coeff_rho_const = 300.*np.random.rand(m,n,o)

    coeff_c_q_pow1 = -2.*np.random.rand(m,n,o)
    coeff_c_q_pow2 = 2.*np.random.rand(m,n,o)

    coeff_oc_Pi = 3.*np.random.rand(m,n,o)

    #Helper Arguments

    drhodQ_ind_I = np.arange(m)

    I = np.tile(np.arange(m),(n*o,1)).T.flatten()
    J = np.tile(np.tile(np.arange(n),(o,1)).T.flatten(),m)
    K = np.tile(np.arange(o),m*n)
    dcdq_ind = (I,J,K,I,K)

    ind_IJ_I = np.tile(np.arange(m),(n,1)).T.flatten()
    ind_IJ_J = np.tile(np.arange(n),m)
    ind_JK_J = np.tile(np.arange(n),(o,1)).T.flatten()
    ind_JK_K = np.tile(np.arange(o),n)

    return [m,n,o,
            coeff_f_Q,coeff_rho_Q,coeff_rho_q,coeff_rho_const,
            coeff_c_q_pow1,coeff_c_q_pow2,coeff_oc_Pi,
            drhodQ_ind_I,dcdq_ind,
            ind_IJ_I,ind_IJ_J,ind_JK_J,ind_JK_K]
