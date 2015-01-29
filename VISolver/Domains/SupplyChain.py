import numpy as np

import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import matplotlib.cm as cm

from VISolver.Domain import Domain


class SupplyChain(Domain):

    def __init__(self,Network,alpha=2):
        raise NotImplementedError(
            'SupplyChain domain is under construction. Coming soon!')
        self.UnpackNetwork(Network)
        self.Network = (self.I,self.Nm,self.Nd,self.Nr)
        self.Dim = self.CalculateNetworkSize()
        self.alpha = alpha

        self.cmap = cm.rainbow
        norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
        self.to_rgba = [cm.ScalarMappable(norm=norm, cmap=self.cmap).to_rgba]*5

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
        maxFlows = [0]*5
        for data in Data:
            newFlows = [np.max(flow) for flow in list(self.PathFlow2LinkFlow_x2f(*self.UnpackData(data))[0])]
            maxFlows = np.max([maxFlows,newFlows],axis=0)
        norm = [mpl.colors.Normalize(vmin=0.,vmax=mxf) for mxf in maxFlows]
        self.to_rgba = [cm.ScalarMappable(norm=n, cmap=self.cmap).to_rgba for n in norm]

    def InitVisual(self):

        ax = plt.gca()
        fig = plt.gcf()

        # Add Colorbar
        cax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
        plt.axis('off')
        cb = mpl.colorbar.ColorbarBase(cax, cmap=self.cmap, spacing='proportional')
        cb.set_label('Suppy Chain Product Flow (Q)')

        plt.sca(ax)

        # Create Network Skeleton
        mid = (max([self.I,self.I*self.Nm,self.I*self.Nd,self.Nr]) - 1.)/2.
        Ix = np.linspace(mid-(self.I-1.)/2.,mid+(self.I-1.)/2.,self.I); Iy = 4.
        Mx = np.linspace(mid-(self.I*self.Nm-1.)/2.,mid+(self.I*self.Nm-1.)/2.,self.I*self.Nm); My = 3.
        D1x = np.linspace(mid-(self.I*self.Nd-1.)/2.,mid+(self.I*self.Nd-1.)/2.,self.I*self.Nd); D1y = 2.
        D2x = D1x; D2y = 1.
        Rx = np.linspace(mid-(self.Nr-1.)/2.,mid+(self.Nr-1.)/2.,self.Nr); Ry = 0.
        od = []
        for i in xrange(self.I):
            for m in xrange(self.I*self.Nm,(self.I+1)*self.Nm):
                od.append([(Ix[i],Iy),(Mx[m],My)])
        for i in xrange(self.I):
            for m in xrange(self.Nm):
                for d1 in xrange(self.Nd):
                    od.append([(Mx[(i+1)*m],My),(D1x[(i+1)*d1],D1y)])

        lc = mc.LineCollection(od, colors=(0,0,0,0), linewidths=10)
        ax.add_collection(lc)
        ax.set_xlim((0,2*mid))
        ax.set_ylim((-.5,4.5))
        ax.add_collection(lc)

        # Annotate Plot
        plt.box('off')
        plt.yticks([0,1,2,3,4],['Demand\nMarkets', 'Warehouses', 'Transportation', 'Manufacturing', 'Firms'], rotation=45)
        plt.xticks(Kx,['Market\n'+str(r+1) for r in xrange(self.Nr)])
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

        self.I,self.Nm,self.Nd,self.Nr,\
        self.coeff_rho_d,self.coeff_rho_const,\
        self.coeff_c_pow1_IM,self.coeff_c_pow2_IM,self.coeff_c_pow1_MD1,self.coeff_c_pow2_MD1,\
        self.coeff_c_pow1_D1D2,self.coeff_c_pow2_D1D2,self.coeff_c_pow1_D2R,self.coeff_c_pow2_D2R,\
        self.coeff_g_pow1_IM,self.coeff_g_pow2_IM,self.coeff_g_pow1_MD1,self.coeff_g_pow2_MD1,\
        self.coeff_g_pow1_D1D2,self.coeff_g_pow2_D1D2,self.coeff_g_pow1_D2R,self.coeff_g_pow2_D2R,\
        self.u_IM,self.u_MD1,self.u_D1D2,self.u_D2R,\
        self.coeff_e_f_pow1_IM,self.coeff_e_f_pow2_IM,self.coeff_e_g_pow1_IM,self.coeff_e_g_pow2_IM,\
        self.coeff_e_f_pow1_MD1,self.coeff_e_f_pow2_MD1,self.coeff_e_g_pow1_MD1,self.coeff_e_g_pow2_MD1,\
        self.coeff_e_f_pow1_D1D2,self.coeff_e_f_pow2_D1D2,self.coeff_e_g_pow1_D1D2,self.coeff_e_g_pow2_D1D2,\
        self.coeff_e_f_pow1_D2R,self.coeff_e_f_pow2_D2R,self.coeff_e_g_pow1_D2R,self.coeff_e_g_pow2_D2R,\
        self.w,\
        self.ind_IM_I,self.ind_IM_M,self.ind_MD1_I,self.ind_MD1_M,self.ind_MD1_D,\
        self.ind_D1D2_I,self.ind_D1D2_D,self.ind_D2R_I,self.ind_D2R_D,self.ind_D2R_R,\
        self.ind_I,self.ind_M,self.ind_D,self.ind_R = Network

    def UnpackData(self,Data):
        # not implemented yet
        return [np.reshape(Data[s:s+self.Dim//3],(self.m,self.n,self.o)) for s in xrange(0,self.Dim,self.Dim//3)]

    def CalculateNetworkSize(self):

        return self.I*self.Nm*self.Nd*self.Nr*(1.+2./(self.Nd*self.Nr)+2./self.Nr+2./(self.Nm*self.Nr)+2./self.Nm)

    def PathFlow2LinkFlow_x2f(x):

        f_IM = np.reshape(np.sum(x[self.ind_IM_I,self.ind_IM_M,:,:],axis=(1,2)),(self.I,self.Nm));
        f_MD1 = np.reshape(np.sum(x[self.ind_MD1_I,self.ind_MD1_M,self.ind_MD1_D,:],axis=(1,)),(self.I,self.Nm,self.Nd));
        f_D1D2 = np.reshape(np.sum(x[self.ind_D1D2_I,:,self.ind_D1D2_D,:],axis=(1,2)),(self.I,self.Nd));
        f_D2R = np.reshape(np.sum(x[self.ind_D2R_I,:,self.ind_D2R_D,self.ind_D2R_R],axis=(1,)),(self.I,self.Nd,self.Nr));

        return f_IM, f_MD1, f_D1D2, f_D2R;

    def F_P2UP(self,Data):

        # Unpack Data
        Q,q,Pi = self.UnpackData(Data)

        F_unpacked =  self.FX_dX(Q,q,Pi)

        # Pack Data
        F_packed = np.array([])
        for Fx in F_unpacked:
            F_packed = np.append(F_packed,Fx.flatten())

        return F_packed

    def SumLinksOnPath_Edap(f_IM,f_MD1,f_D1D2,f_D2R):

        return np.reshape(f_IM[self.ind_I,self.ind_M] + f_MD1[self.ind_I,self.ind_M,self.ind_D] +
            f_D1D2[self.ind_I,self.ind_D] + f_D2R[self.ind_I,self.ind_D,self.ind_R],(self.I,self.Nm,self.Nd,self.Nr))

    def dOperationalCostdLinkFlow_dcdf(f_IM,f_MD1,f_D1D2,f_D2R):

        dcdf_IM   = self.coeff_c_pow1_IM   + 2.*self.coeff_c_pow2_IM*f_IM
        dcdf_MD1  = self.coeff_c_pow1_MD1  + 2.*self.coeff_c_pow2_MD1*f_MD1
        dcdf_D1D2 = self.coeff_c_pow1_D1D2 + 2.*self.coeff_c_pow2_D1D2*f_D1D2
        dcdf_D2R  = self.coeff_c_pow1_D2R  + 2.*self.coeff_c_pow2_D2R*f_D2R

        return dcdf_IM, dcdf_MD1, dcdf_D1D2, dcdf_D2R

    def dOperationalCostdPathFlow_dCdx(f_IM,f_MD1,f_D1D2,f_D2R):

        dcdf_IM, dcdf_MD1, dcdf_D1D2, dcdf_D2R = self.dOperationalCostdLinkFlow_dcdf(f_IM,f_MD1,f_D1D2,f_D2R)

        return self.SumLinksOnPath_Edap(dcdf_IM,dcdf_MD1,dcdf_D1D2,dcdf_D2R)

    def dEmissionCostdLinkFlow_dedf(f_IM,f_MD1,f_D1D2,f_D2R):

        dedf_IM   = self.coeff_e_f_pow1_IM   + 2.*self.coeff_e_f_pow2_IM*f_IM
        dedf_MD1  = self.coeff_e_f_pow1_MD1  + 2.*self.coeff_e_f_pow2_MD1*f_MD1
        dedf_D1D2 = self.coeff_e_f_pow1_D1D2 + 2.*self.coeff_e_f_pow2_D1D2*f_D1D2
        dedf_D2R  = self.coeff_e_f_pow1_D2R  + 2.*self.coeff_e_f_pow2_D2R*f_D2R

        return dedf_IM, dedf_MD1, dedf_D1D2, dedf_D2R

    def dEmissionCostdPathFlow_dEdx(f_IM,f_MD1,f_D1D2,f_D2R):

        dedf_IM, dedf_MD1, dedf_D1D2, dedf_D2R = self.dEmissionCostdLinkFlow_dedf(f_IM,f_MD1,f_D1D2,f_D2R)

        return self.SumLinksOnPath_Edap(dedf_IM,dedf_MD1,dedf_D1D2,dedf_D2R)

    def dLinkFlow_df(f_IM,f_MD1,f_D1D2,f_D2R,\
        lam_IM,lam_MD1,lam_D1D2,lam_D2R):

        dcdf_IM, dcdf_MD1, dcdf_D1D2, dcdf_D2R = self.dOperationalCostdLinkFlow_dcdf(f_IM,f_MD1,f_D1D2,f_D2R)
        dedf_IM, dedf_MD1, dedf_D1D2, dedf_D2R = self.dEmissionCostdLinkFlow_dedf(f_IM,f_MD1,f_D1D2,f_D2R)

        w_IM   = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedf_IM.shape[1:])).flatten(),dedf_IM.shape)
        w_MD1  = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedf_MD1.shape[1:])).flatten(),dedf_MD1.shape)
        w_D1D2 = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedf_D1D2.shape[1:])).flatten(),dedf_D1D2.shape)
        w_D2R  = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedf_D2R.shape[1:])).flatten(),dedf_D2R.shape)

        df_IM   = dcdf_IM   + w_IM*dedf_IM     + lam_IM
        df_MD1  = dcdf_MD1  + w_MD1*dedf_MD1   + lam_MD1
        df_D1D2 = dcdf_D1D2 + w_D1D2*dedf_D1D2 + lam_D1D2
        df_D2R  = dcdf_D2R  + w_D2R*dedf_D2R   + lam_D2R

        return df_IM, df_MD1, df_D1D2, df_D2R

    def DemandPrice_rho(d):

        return -np.sum(self.coeff_rho_d*np.swapaxes(np.rollaxis(np.tile(d,(self.coeff_rho_d.shape[2],1,1)),0,len(self.coeff_rho_d.shape)),0,2),axis=2) + self.coeff_rho_const

    def dDemandPricedDemand_drhodd():

        return -self.coeff_rho_d[self.drhodd_ind_I,self.drhodd_ind_R,self.drhodd_ind_I]

    def dDemand_dd(d):

        rho    = self.DemandPrice_rho(d)
        drhodd = self.dDemandPricedDemand_drhodd()

        return -rho - drhodd*d

    def dPathFlow_dx(x,\
        lam_IM,lam_MD1,lam_D1D2,lam_D2R):

        #reconstruct f from x
        f_IM, f_MD1, f_D1D2, f_D2R = self.PathFlow2LinkFlow_x2f(x)
        #reconstruct d from f
        d = np.sum(f_D2R,axis=(1,))

        dCdx = self.dOperationalCostdPathFlow_dCdx(f_IM,f_MD1,f_D1D2,f_D2R)

        w_x = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(x.shape[1:])).flatten(),x.shape)

        dEdx = self.dEmissionCostdPathFlow_dEdx(f_IM,f_MD1,f_D1D2,f_D2R)

        lam_Path = self.SumLinksOnPath_Edap(lam_IM,lam_MD1,lam_D1D2,lam_D2R)

        dRhodx_d = self.dDemand_dd(d)
        dRhodx_d = np.tile(np.swapaxes(dRhodx_d[None][None],0,2),(1,self.Nm,self.Nd,1))

        return dCdx + w_x*dEdx + lam_Path + dRhodx_d

    def dFrequencyCostdGamma_dgdgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R):

        dgdgam_IM   = self.coeff_g_pow1_IM   + 2.*self.coeff_g_pow2_IM*gam_IM
        dgdgam_MD1  = self.coeff_g_pow1_MD1  + 2.*self.coeff_g_pow2_MD1*gam_MD1
        dgdgam_D1D2 = self.coeff_g_pow1_D1D2 + 2.*self.coeff_g_pow2_D1D2*gam_D1D2
        dgdgam_D2R  = self.coeff_g_pow1_D2R  + 2.*self.coeff_g_pow2_D2R*gam_D2R

        return dgdgam_IM, dgdgam_MD1, dgdgam_D1D2, dgdgam_D2R    

    def dEmissionCostdGamma_dedgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R):

        dedgam_IM   = self.coeff_e_g_pow1_IM   + 2.*self.coeff_e_g_pow2_IM*gam_IM
        dedgam_MD1  = self.coeff_e_g_pow1_MD1  + 2.*self.coeff_e_g_pow2_MD1*gam_MD1
        dedgam_D1D2 = self.coeff_e_g_pow1_D1D2 + 2.*self.coeff_e_g_pow2_D1D2*gam_D1D2
        dedgam_D2R  = self.coeff_e_g_pow1_D2R  + 2.*self.coeff_e_g_pow2_D2R*gam_D2R

        return dedgam_IM, dedgam_MD1, dedgam_D1D2, dedgam_D2R

    def dGamma_dgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R,\
        lam_IM,lam_MD1,lam_D1D2,lam_D2R):

        dgdgam_IM, dgdgam_MD1, dgdgam_D1D2, dgdgam_D2R = self.dFrequencyCostdGamma_dgdgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R)
        dedgam_IM, dedgam_MD1, dedgam_D1D2, dedgam_D2R = self.dEmissionCostdGamma_dedgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R)

        w_IM   = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedgam_IM.shape[1:])).flatten(),dedgam_IM.shape)
        w_MD1  = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedgam_MD1.shape[1:])).flatten(),dedgam_MD1.shape)
        w_D1D2 = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedgam_D1D2.shape[1:])).flatten(),dedgam_D1D2.shape)
        w_D2R  = np.reshape(np.tile(np.swapaxes(w[None],0,1),np.prod(dedgam_D2R.shape[1:])).flatten(),dedgam_D2R.shape)

        dgam_IM   = dgdgam_IM   + w_IM*dedgam_IM     - u_IM*lam_IM
        dgam_MD1  = dgdgam_MD1  + w_MD1*dedgam_MD1   - u_MD1*lam_MD1
        dgam_D1D2 = dgdgam_D1D2 + w_D1D2*dedgam_D1D2 - u_D1D2*lam_D1D2
        dgam_D2R  = dgdgam_D2R  + w_D2R*dedgam_D2R   - u_D2R*lam_D2R

        return dgam_IM, dgam_MD1, dgam_D1D2, dgam_D2R

    def dLambda_dlam(f_IM,f_MD1,f_D1D2,f_D2R,\
        gam_IM,gam_MD1,gam_D1D2,gam_D2R):

        dlam_IM   = self.u_IM*gam_IM     - f_IM
        dlam_MD1  = self.u_MD1*gam_MD1   - f_MD1
        dlam_D1D2 = self.u_D1D2*gam_D1D2 - f_D1D2
        dlam_D2R  = self.u_D2R*gam_D2R   - f_D2R

        return dlam_IM, dlam_MD1, dlam_D1D2, dlam_D2R

    def dLambda_dlam(x,\
        gam_IM,gam_MD1,gam_D1D2,gam_D2R):

        f_IM, f_MD1, f_D1D2, f_D2R = self.PathFlow2LinkFlow_x2f(x)

        dlam_IM   = self.u_IM*gam_IM     - f_IM
        dlam_MD1  = self.u_MD1*gam_MD1   - f_MD1
        dlam_D1D2 = self.u_D1D2*gam_D1D2 - f_D1D2
        dlam_D2R  = self.u_D2R*gam_D2R   - f_D2R

        return dlam_IM, dlam_MD1, dlam_D1D2, dlam_D2R

    def FX_dX(self,Q,q,Pi):
        # dgamma returns list - might have issue here
        return [self.dPathFlow_dx(x,lam_IM,lam_MD1,lam_D1D2,lam_D2R),
                self.dGamma_dgam(gam_IM,gam_MD1,gam_D1D2,gam_D2R,lam_IM,lam_MD1,lam_D1D2,lam_D2R),
                self.dLambda_dlam(x,gam_IM,gam_MD1,gam_D1D2,gam_D2R)]

def CreateNetworkExample(ex=1):

    m = 3
    n = 2
    o = 2

    coeff_f_Q = np.array([2.,1.,3.])

    coeff_rho_Q = np.zeros((3,2,2,3,2,2))
    coeff_rho_Q[0,0,0,0,0,0] = -1.; coeff_rho_Q[0,0,0,0,0,1] = -.5
    coeff_rho_Q[0,0,1,0,0,1] = -2.; coeff_rho_Q[0,0,1,0,0,0] = -1.
    coeff_rho_Q[0,1,0,0,1,0] = -2.; coeff_rho_Q[0,1,0,0,0,0] = -.5
    coeff_rho_Q[0,1,1,0,1,1] = -3.; coeff_rho_Q[0,1,1,0,0,1] = -1.
    coeff_rho_Q[1,0,0,1,0,0] = -1.; coeff_rho_Q[1,0,0,1,0,1] = -.5
    coeff_rho_Q[1,0,1,1,0,1] = -3.
    coeff_rho_Q[1,1,0,1,1,0] = -2.; coeff_rho_Q[1,1,0,1,1,1] = -1.
    coeff_rho_Q[1,1,1,1,1,1] = -3.; coeff_rho_Q[1,1,1,0,1,0] = -1.
    coeff_rho_Q[2,0,0,2,0,0] = -4.
    coeff_rho_Q[2,0,1,2,0,1] = -2.; coeff_rho_Q[2,0,1,2,1,0] = -1.
    coeff_rho_Q[2,1,0,2,1,0] = -3.; coeff_rho_Q[2,1,0,2,0,0] = -1.
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
    if ex==2:
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

def CreateRandomNetwork(I,Nm,Nd,Nr,seed):

    np.random.seed(seed)

    #Rho's - Demand Functions
    coeff_rho_d = np.zeros((I,Nr,I))
    for i in xrange(I):
        for j in xrange(Nr):
            #self dependence
            coeff_rho_d[i,j,i] = 5.*np.random.rand()
            #external dependence
            ext = np.random.rand(I-1)
            amp = 0.4*np.random.rand()+0.1
            ext = amp*coeff_rho_d[i,j,i]*ext/np.sum(ext)
            count = 0
            for k in xrange(I):
                if k!=i:
                    coeff_rho_d[i,j,k] = ext[count]
                    count+=1
    coeff_rho_const = 100.*np.random.rand(I,Nr)+300.

    #C's - Operational Costs
    coeff_c_pow1_IM   = np.zeros((I,Nm))
    coeff_c_pow2_IM   = np.zeros((I,Nm))
    coeff_c_pow1_MD1  = np.zeros((I,Nm,Nd))
    coeff_c_pow2_MD1  = np.zeros((I,Nm,Nd))
    coeff_c_pow1_D1D2 = np.zeros((I,Nd))
    coeff_c_pow2_D1D2 = np.zeros((I,Nd))
    coeff_c_pow1_D2R  = np.zeros((I,Nd,Nr))
    coeff_c_pow2_D2R  = np.zeros((I,Nd,Nr))
    for i in xrange(I):
        for j in xrange(Nm):
            coeff_c_pow1_IM[i,j] = 5.*np.random.rand()+1.
            coeff_c_pow2_IM[i,j] = 6.*np.random.rand()+0.
    for i in xrange(I):
        for j in xrange(Nm):
            for k in xrange(Nd):
                coeff_c_pow1_MD1[i,j,k] = 10.-8.*(coeff_c_pow2_IM[i,j]/6.) #deterministic
                coeff_c_pow2_MD1[i,j,k] = 1.-0.5*(coeff_c_pow2_IM[i,j]/6.) #deterministic
    for i in xrange(I):
        for j in xrange(Nd):
            coeff_c_pow1_D1D2[i,j] = np.random.rand()+.5
            coeff_c_pow2_D1D2[i,j] = 0.2*np.random.rand()+.4
    for i in xrange(I):
        for j in xrange(Nd):
            for k in xrange(Nr):
                coeff_c_pow1_D2R[i,j,k] = np.random.rand()+1.
                coeff_c_pow2_D2R[i,j,k] = .4*np.random.rand()+.8

    #G's - Frequency Costs
    coeff_g_pow1_IM   = np.zeros((I,Nm))
    coeff_g_pow2_IM   = np.zeros((I,Nm))
    coeff_g_pow1_MD1  = np.zeros((I,Nm,Nd))
    coeff_g_pow2_MD1  = np.zeros((I,Nm,Nd))
    coeff_g_pow1_D1D2 = np.zeros((I,Nd))
    coeff_g_pow2_D1D2 = np.zeros((I,Nd))
    coeff_g_pow1_D2R  = np.zeros((I,Nd,Nr))
    coeff_g_pow2_D2R  = np.zeros((I,Nd,Nr))
    for i in xrange(I):
        for j in xrange(Nm):
            coeff_g_pow1_IM[i,j] = 2.*(coeff_c_pow2_IM[i,j]/6.)+0.5 #deterministic
            coeff_g_pow2_IM[i,j] = coeff_c_pow2_IM[i,j]/6.+0.3      #deterministic
    for i in xrange(I):
        for j in xrange(Nm):
            for k in xrange(Nd):
                coeff_g_pow1_MD1[i,j,k] = coeff_g_pow2_IM[i,j]      #deterministic
                coeff_g_pow2_MD1[i,j,k] = 2.-(coeff_g_pow2_IM[i,j]) #deterministic
    for i in xrange(I):
        for j in xrange(Nd):
            coeff_g_pow1_D1D2[i,j] = np.random.rand()+.25
            coeff_g_pow2_D1D2[i,j] = 0.5*np.random.rand()+.75
    for i in xrange(I):
        for j in xrange(Nd):
            for k in xrange(Nr):
                coeff_g_pow1_D2R[i,j,k] = 0.5*np.random.rand()+.75
                coeff_g_pow2_D2R[i,j,k] = .1*np.random.rand()+.95

    #U's - Capacities
    u_IM = np.zeros((I,Nm))
    u_MD1 = np.zeros((I,Nm,Nd))
    u_D1D2 = np.zeros((I,Nd))
    u_D2R = np.zeros((I,Nd,Nr))
    for i in xrange(I):
        u_IM[i,:] = 20.*np.random.rand()+90.
    for i in xrange(I):
        for j in xrange(Nm):
            u_MD1[i,j,:] = 60.-50.*(coeff_c_pow2_IM[i,j]/6.) #deterministic
    for i in xrange(I):
        u_D1D2[i,:] = u_IM[i,0] #deterministic
    for i in xrange(I):
        for j in xrange(Nr):
            u_D2R[i,:,j] = 10.*np.random.rand()+15.

    #E's - Emission Costs
    coeff_e_f_pow1_IM   = np.zeros((I,Nm))
    coeff_e_f_pow2_IM   = np.zeros((I,Nm))
    coeff_e_g_pow1_IM   = np.zeros((I,Nm))
    coeff_e_g_pow2_IM   = np.zeros((I,Nm))
    coeff_e_f_pow1_MD1  = np.zeros((I,Nm,Nd))
    coeff_e_f_pow2_MD1  = np.zeros((I,Nm,Nd))
    coeff_e_g_pow1_MD1  = np.zeros((I,Nm,Nd))
    coeff_e_g_pow2_MD1  = np.zeros((I,Nm,Nd))
    coeff_e_f_pow1_D1D2 = np.zeros((I,Nd))
    coeff_e_f_pow2_D1D2 = np.zeros((I,Nd))
    coeff_e_g_pow1_D1D2 = np.zeros((I,Nd))
    coeff_e_g_pow2_D1D2 = np.zeros((I,Nd))
    coeff_e_f_pow1_D2R  = np.zeros((I,Nd,Nr))
    coeff_e_f_pow2_D2R  = np.zeros((I,Nd,Nr))
    coeff_e_g_pow1_D2R  = np.zeros((I,Nd,Nr))
    coeff_e_g_pow2_D2R  = np.zeros((I,Nd,Nr))
    for i in xrange(I):
        for j in xrange(Nm):
            coeff_e_f_pow1_IM[i,j] = 0.5*np.random.rand()+0.5
            coeff_e_f_pow2_IM[i,j] = 0.1*np.random.rand()+0.05
            coeff_e_g_pow1_IM[i,j] = 1.0*np.random.rand()+1.0
            coeff_e_g_pow2_IM[i,j] = 2.0*np.random.rand()+0.5
    for i in xrange(I):
        for j in xrange(Nm):
            for k in xrange(Nd):
                coeff_e_f_pow1_MD1[i,j,k] = 0.5*np.random.rand()+0.5
                coeff_e_f_pow2_MD1[i,j,k] = 0.05*np.random.rand()+0.05
                coeff_e_g_pow1_MD1[i,j,k] = 1.0*np.random.rand()+1.0
                coeff_e_g_pow2_MD1[i,j,k] = 1.0*np.random.rand()+1.0
    for i in xrange(I):
        for j in xrange(Nd):
            coeff_e_f_pow1_D1D2[i,j] = 0.02*np.random.rand()+0.09
            coeff_e_f_pow2_D1D2[i,j] = 0.04*np.random.rand()+0.01
            coeff_e_g_pow1_D1D2[i,j] = 0.1*np.random.rand()+0.1
            coeff_e_g_pow2_D1D2[i,j] = 0.02*np.random.rand()+0.09
    for i in xrange(I):
        for j in xrange(Nd):
            for k in xrange(Nr):
                coeff_e_f_pow1_D2R[i,j,k] = 1.0*np.random.rand()+1.0
                coeff_e_f_pow2_D2R[i,j,k] = 0.1*np.random.rand()+0.1
                coeff_e_g_pow1_D2R[i,j,k] = 1.0*np.random.rand()+1.0
                coeff_e_g_pow2_D2R[i,j,k] = 3.0*np.random.rand()+1.0

    w = np.ones((I,))

    #Helper Arguments

    #dd
    drhodd_ind_I = np.arange(coeff_rho_d.shape[0])[:,None]
    drhodd_ind_R = np.arange(coeff_rho_d.shape[1])

    #x2f
    ind_IM_I = np.tile(np.arange(I),(Nm,1)).T.flatten()
    ind_IM_M = np.tile(np.arange(Nm),I)

    ind_MD1_I = np.tile(np.arange(I),(Nm*Nd,1)).T.flatten()
    ind_MD1_M = np.tile(np.tile(np.arange(Nm),(Nd,1)).T.flatten(),I)
    ind_MD1_D = np.tile(np.arange(Nd),I*Nm)

    ind_D1D2_I = np.tile(np.arange(I),(Nd,1)).T.flatten()
    ind_D1D2_D = np.tile(np.arange(Nd),I)

    ind_D2R_I = np.tile(np.arange(I),(Nd*Nr,1)).T.flatten()
    ind_D2R_D = np.tile(np.tile(np.arange(Nd),(Nr,1)).T.flatten(),I)
    ind_D2R_R = np.tile(np.arange(Nr),I*Nd)

    #Edap
    ind_I = np.tile(np.arange(I),(Nm*Nd*Nr,1)).T.flatten()
    ind_M = np.tile(np.tile(np.arange(Nm),(Nd*Nr,1)).T.flatten(),I)
    ind_D = np.tile(np.tile(np.arange(Nd),(Nr,1)).T.flatten(),I*Nm)
    ind_R = np.tile(np.arange(Nr),I*Nm*Nd)

    return [I,Nm,Nd,Nr,
        coeff_rho_d,coeff_rho_const,
        coeff_c_pow1_IM,coeff_c_pow2_IM,coeff_c_pow1_MD1,coeff_c_pow2_MD1,
        coeff_c_pow1_D1D2,coeff_c_pow2_D1D2,coeff_c_pow1_D2R,coeff_c_pow2_D2R,
        coeff_g_pow1_IM,coeff_g_pow2_IM,coeff_g_pow1_MD1,coeff_g_pow2_MD1,
        coeff_g_pow1_D1D2,coeff_g_pow2_D1D2,coeff_g_pow1_D2R,coeff_g_pow2_D2R,
        u_IM,u_MD1,u_D1D2,u_D2R,
        coeff_e_f_pow1_IM,coeff_e_f_pow2_IM,coeff_e_g_pow1_IM,coeff_e_g_pow2_IM,
        coeff_e_f_pow1_MD1,coeff_e_f_pow2_MD1,coeff_e_g_pow1_MD1,coeff_e_g_pow2_MD1,
        coeff_e_f_pow1_D1D2,coeff_e_f_pow2_D1D2,coeff_e_g_pow1_D1D2,coeff_e_g_pow2_D1D2,
        coeff_e_f_pow1_D2R,coeff_e_f_pow2_D2R,coeff_e_g_pow1_D2R,coeff_e_g_pow2_D2R,
        w,
        ind_IM_I,ind_IM_M,ind_MD1_I,ind_MD1_M,ind_MD1_D,
        ind_D1D2_I,ind_D1D2_D,ind_D2R_I,ind_D2R_D,ind_D2R_R,
        ind_I,ind_M,ind_D,ind_R]