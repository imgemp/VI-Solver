import numpy as np
from scipy.spatial.distance import cdist

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm

from VISolver.Domain import Domain

from IPython import embed
class MLN(Domain):

    def __init__(self,Network):
        self.Network = Network
        self.UnpackNetwork(Network)
        self.dim = self.CalculateNetworkSize()

    def F(self,Data):
        return self._F(Data)

    # Functions Used to Animate and Save Network Run to Movie File

    def plotboundary(self,ax,B,b,c='w',boundary=None,mask=False):
        if B[0] != 0 and B[1] != 0:
            start = [self.xlims[0],-(B[0]*self.xlims[0]+b)/B[1]]
            if start[1] < self.ylims[0]:
                start = [-(B[1]*self.ylims[0]+b)/B[0],self.ylims[0]]
            elif start[1] > self.ylims[1]:
                start = [-(B[1]*self.ylims[1]+b)/B[0],self.ylims[1]]

            end = [self.xlims[1],-(B[0]*self.xlims[1]+b)/B[1]]
            if end[1] < self.ylims[0]:
                end = [-(B[1]*self.ylims[0]+b)/B[0],self.ylims[0]]
            elif end[1] > self.ylims[1]:
                end = [-(B[1]*self.ylims[1]+b)/B[0],self.ylims[1]]
        elif B[0] == 0 and B[1] != 0:
            start = [self.xlims[0],-b/B[1]]
            end = [self.xlims[1],-b/B[1]]
        elif B[1] != 0 and B[1] == 0:
            start = [-b/B[0],self.ylims[0]]
            end = [-b/B[0],self.ylims[1]]
        else:
            start = [self.xlims[0]-1,self.ylims[0]-1]
            end = [self.xlims[0]-1,self.ylims[0]-1]

        xs = [start[0],end[0]]
        ys = [start[1],end[1]]

        if mask:
            xs = np.ma.array(xs,mask=True)
            ys = np.ma.array(ys,mask=True)

        if boundary is None:
            boundary, = ax.plot(xs,ys,c=c)
        else:
            boundary.set_xdata(xs)
            boundary.set_ydata(ys)

        return boundary

    def InitVisual(self):

        assert self.D == 2

        assert self.I == 4
        colors = ['c','m','b','y']

        ax = plt.gca()
        fig = plt.gcf()

        ax.set_xlim(self.xlims)
        ax.set_ylim(self.ylims)

        pos = ax.plot(self.pos[0],self.pos[1],'g+',mew=5,ms=10)
        neg = ax.plot(self.neg[0],self.neg[1],'rx',mew=5,ms=10)

        datacenters = ax.scatter(self.xjd[:,0],self.xjd[:,1],c='k',s=50)
        boundary = self.plotboundary(ax,self.B,self.b,c='k')

        maskedI = np.ma.array(np.zeros(self.I),mask=True)
        self.datastreams = ax.scatter(maskedI,maskedI,c='w',s=50)
        self.datastreams.set_facecolors(colors)

        self.boundaries = [self.plotboundary(ax,np.zeros(self.D),0,c=colors[i],mask=True) for i in range(self.I)]

        return datacenters,boundary,pos,neg

    def UpdateVisual(self,num,ax,Frames):

        Data = Frames[num]

        qij, xid, Bid, bi = self.UnpackData(Data)

        one = np.ones(xid.shape[0])
        pyx = self.Py_x(one,xid,Bid,bi)
        pred_one = pyx > 0.5
        colors = ['g' if po else 'r' for po in pred_one]

        self.datastreams.set_offsets(xid)
        self.datastreams.set_edgecolors(colors)
        boundaries = [self.plotboundary(ax,B,b,boundary=self.boundaries[i]) for i,B,b in zip(range(self.I),Bid,bi)]

        return self.datastreams,boundaries

    # Functions used to Initialize the Machine Learning Network and Calculate F

    def UnpackNetwork(self,Network):
        I,J,D,pAj,pbj,ri,Ci,ci,lamj,xjd,Gj,gami,duration,Ki,alphai,B,b,los,his,pos,neg = Network
        self.I = I
        self.J = J
        self.D = D
        self.pAj = pAj
        self.pbj = pbj
        self.ri = ri
        self.Ci = Ci
        self.ci = ci
        self.lamj = lamj
        self.clam = np.outer(self.ci,self.lamj)
        self.xjd = xjd
        self.Gj = Gj
        self.gami = gami
        self.duration = duration
        self.counter = 0
        self.nt = None
        self.Ki = Ki
        self.alphai = alphai
        self.B = B
        self.b = b
        self.los = los
        self.his = his
        self.xlims = [self.los[I*J],self.his[I*J]]
        self.ylims = [self.los[I*J+1],self.his[I*J+1]]
        self.pos = pos
        self.neg = neg

    def CalculateNetworkSize(self):
        I,J,D = self.I, self.J, self.D
        return I*J + 2*I*D + I

    def UnpackData(self,Data):
        I,J,D = self.I, self.J, self.D
        ptr = 0
        qij = Data[ptr:ptr+I*J].reshape((I,J))
        ptr += I*J
        xid = Data[ptr:ptr+I*D].reshape((I,D))
        ptr += I*D
        Bid = Data[ptr:ptr+I*D].reshape((I,D))
        ptr += I*D
        bi = Data[ptr:ptr+I]
        return qij,xid,Bid,bi

    def Price(self,qij):
        return self.pAj - self.pbj*np.sum(qij,axis=0)

    def Distances(self,xid):
        return np.hstack([cdist(xid,self.xjd[j,None],'mahalanobis',VI=self.Gj[j]) for j in range(self.J)])

    def NewsTip(self,xid):
        if self.nt is None or self.counter % self.duration == 0:
            los = self.los[self.I*self.J:self.I*(self.J+self.D)]
            his = self.his[self.I*self.J:self.I*(self.J+self.D)]
            closest = np.round((xid.flatten() - los)/(his-los))
            farthest = closest*los + (1-closest)*his
            self.nt = farthest.reshape(xid.shape)
            self.counter = 1
        else:
            self.counter += 1
        return self.nt

    def y(self,xid):
        one = np.ones(xid.shape[0])
        return (self.Py_x(one,xid,self.B,self.b) > 0.5).astype(float)*2-1

    def Py_x(self,yi,xid,Bid,bi):
        return 1/(1+np.exp(-yi*(np.sum(Bid*xid,axis=1)+bi)))

    def _F(self,Data):
        qij, xid, Bid, bi = self.UnpackData(Data)
        pj = self.Price(qij)
        dij = self.Distances(xid)
        xtd = self.NewsTip(xid)
        yi = self.y(xid)
        pyx = self.Py_x(yi,xid,Bid,bi)

        dfi_dqij = pj - self.pbj*qij + 2*(self.ri*qij.T).T - self.clam + (self.ci*dij.T).T
        temp_dij = np.swapaxes([np.dot((xid-self.xjd[j]),self.Gj[j]) for j in range(self.J)],0,2)
        dfi_dxid = 2*(self.gami*(xid-xtd).T).T + 2*(self.ci*np.sum(qij*temp_dij,axis=-1)).T + 2*(self.alphai*(((Bid*xid).T-bi).T*Bid).T).T
        dLRi_dBid = (self.Ki*Bid.T).T - (yi*(1-pyx)*xid.T).T
        dLRi_dbi = self.Ki*bi - yi*(1-pyx)

        return np.hstack([delta.flatten() for delta in [dfi_dqij,dfi_dxid,dLRi_dBid,dLRi_dbi]])

def CreateRandomNetwork(I=4,J=3,D=2,seed=None):
    if seed is not None:
        np.random.seed(seed)

    pAj = np.random.rand(J)*.005 + 0.02
    pbj = np.random.rand(J)*0.2
    ri = np.ones(I)*.02
    Ci = np.random.rand(I)
    ci = np.random.rand(I)*.02
    lamj = np.random.rand(J)*5
    xjd = np.random.rand(J,D)
    Gj = np.asarray([np.eye(D) for j in range(J)])
    gami = np.ones(I)
    duration = 100
    Ki = np.ones(I)*0.
    alphai = np.random.rand(I)*.02
    B = np.array([1,-1])*0.5
    b = 0

    los = np.array([0]*I*J + [0]*I*D + [-1]*I*D + [-1]*I)
    his = np.array([1]*I*J + [1]*I*D + [1]*I*D + [1]*I)

    pos = np.array([0.75,0.25])
    neg = np.array([0.25,0.75])

    return [I,J,D,pAj,pbj,ri,Ci,ci,lamj,xjd,Gj,gami,duration,Ki,alphai,B,b,los,his,pos,neg]



