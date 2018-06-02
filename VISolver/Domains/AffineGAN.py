import numpy as np

from VISolver.Domain import Domain

# expansion on z won't do anything
# D(x) needs to be nonlinear to capture statistics of distribution beyond just mean (mu)
class AffineGAN(Domain):

    def __init__(self, u=np.zeros(2), S=np.eye(2), zdim=2, batch_size=100, alpha=1., expansion=False):
        self.u = u
        self.S = S
        self.xdim = np.size(u)
        self.zdim = zdim
        if expansion:
            zdim = zdim + zdim*(zdim+1)//2
        self.idim = zdim
        self.Dim = self.xdim + self.xdim*(self.idim+1)
        self.batch_size = batch_size
        self.alpha = alpha
        self.expansion = expansion

    def F(self,Data):
        x, z = self.get_data()
        d, G = self.get_args(Data)
        dD = x.mean(axis=0) - np.dot(z,G.T).mean(axis=0) - self.alpha*d
        dG = -np.repeat(d,self.idim+1)*np.tile(z.mean(axis=0),self.xdim) + self.alpha*G.flatten()
        return np.concatenate((-dD,dG))

    def V(self,Data):
        x, z = self.get_data()
        d, G = self.get_args(Data)
        gen = np.dot(z,G.T)
        return np.dot(x-gen,d).mean()

    def get_args(self,Data):
        d = Data[:self.xdim]  # xdim
        G = Data[self.xdim:].reshape(self.xdim,self.idim+1)  # xdim x zdim+1
        return d, G

    def get_data(self,size=None):
        if size is None:
            size = self.batch_size
        x = np.random.multivariate_normal(mean=self.u,cov=self.S,size=size)
        z = np.random.normal(size=(size,self.zdim))
        if self.expansion:
            z = self.expand(z)
        ones = np.ones((size,1))
        z = np.hstack((z,ones))  # N x idim+1
        return x, z

    def expand(self,z):
        xtra = []
        for i in range(self.zdim):
            for j in range(i,self.zdim):
                xtra += [z[:,i]*z[:,j]]
        xtra = np.array(xtra).T
        return np.hstack((z,xtra))

    def generate(self,Data,size=100):
        x, z = self.get_data(size)
        d, G = self.get_args(Data)
        return x, np.dot(z,G.T)
