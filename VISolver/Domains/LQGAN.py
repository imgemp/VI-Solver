import numpy as np

from VISolver.Domain import Domain
from VISolver.Utilities import approx_jacobian

from IPython import embed
class LQGAN(Domain):

    def __init__(self,dim=None,mu=None,sig=None,preconditioner='Fsim',var_only=False):
        if dim is not None:
            self.Dim = 3*dim + dim**2
        else:
            dim = mu.size
            self.Dim = 3*mu.size + mu.size**2
        self.dim = int((np.sqrt(4*self.Dim+9)-3)/2)
        assert self.dim == dim
        self.s = (self.dim**2+self.dim)//2
        self.preconditioner = preconditioner
        self.var_only = var_only
        if mu is not None and sig is not None:
            self.set_mu_sigma(mu=mu,sigma=sig)

    def set_mu_sigma(self,mu,sigma):
        self.mu = mu
        self.sig = sigma
        sigsqrt = np.linalg.cholesky(self.sig)
        self.xstar = np.hstack([np.zeros(self.s),np.zeros(self.dim),sigsqrt[np.tril_indices(self.dim)],self.mu])
        # equi_error = np.linalg.norm(self.F(self.xstar))
        # if equi_error > 1e-20:
        #     print('Warning')
        # self.logsigdet = np.log(np.linalg.det(self.sig))
        self.logsigdet = np.linalg.slogdet(self.sig)[1]
        # print(self.logsigdet)

    def F(self,Data):
        if self.preconditioner == 'Fcc':
            return self.Fcc(Data)
        if self.preconditioner == 'Fccprime':
            return self.Fccprime(Data)
        elif self.preconditioner == 'Fsim':
            return self._F(Data)
        elif self.preconditioner == 'Feg':
            return self.Feg(Data)
        elif self.preconditioner == 'Fegprime':
            return self.Fegprime(Data)
        elif self.preconditioner == 'Fcon':
            return self.Fcon(Data)
        elif self.preconditioner == 'Freg':
            return self.Freg(Data)
        elif self.preconditioner == 'Falt':
            return self.Falt(Data)
        elif self.preconditioner == 'Funr':
            return self.Funr(Data)
        else:
            assert False, self.preconditioner+' is not a valid option'

    def upper2sym(self,fill,d):
        W2 = np.zeros((d,d))
        W2[np.triu_indices(d)] = fill
        W2 = W2+W2.T
        W2[range(d),range(d)] /= 2.
        return W2

    def _F(self,Data):
        dim = Data.size
        # dim = 2(s+d) = 2(1/2*d**2+d/2+d) = d**2+3d = dim
        d = int((np.sqrt(4*dim+9)-3)/2)
        s = (d**2+d)//2
        W2 = self.upper2sym(Data[:s],d)
        w1 = Data[s:s+d]
        A = np.zeros((d,d))
        A[np.tril_indices(d)] = Data[-s-d:-d]
        b = Data[-d:]
        # embed()
        W2dot = self.sig - np.dot(A,A.T) + np.outer(self.mu,self.mu) - np.outer(b,b)
        W2dot = W2dot + W2dot.T
        W2dot[range(d),range(d)] /= 2.
        Adot = np.dot(W2+W2.T,A)
        w1dot = self.mu - b
        bdot = np.dot(W2+W2.T,b) + w1
        if self.var_only:
            w1dot *= 0.
            bdot *= 0.

        return -np.hstack([W2dot[np.triu_indices(d)],w1dot,Adot[np.tril_indices(d)],bdot])

    def divAAT(self,Data,F):
        d = self.dim
        # if d == 1:
        #     F[0] /= 2*Data[2]**2
        #     F[2] /= 2*Data[2]**2
        # else:
        s = self.s
        indices = np.tril_indices(d)
        A = np.zeros((d,d))
        A[indices] = Data[-s-d:-d]
        Asigmainv = 0.5*np.linalg.pinv(np.dot(A,A.T))
        FA = np.zeros_like(A)
        FA[indices] = F[-s-d:-d]
        F[-s-d:-d] = Asigmainv.dot(FA)[indices]
        FW2 = np.zeros_like(A)
        FW2[indices] = F[:s]
        FW2 = FW2 + FW2.T
        FW2[range(d),range(d)] /= 2.
        F[:s] = Asigmainv.dot(FW2)[indices]
        return F

    def Fcc(self,Data):
        j = self.J(Data)
        jasy = 0.5*(j-j.T)
        fcc = -jasy.dot(self._F(Data))
        return fcc

    def Fccprime(self,Data):
        return self.divAAT(Data,self.Fcc(Data))

    def Feg(self,Data):
        j = self.J(Data)
        feg = -j.dot(self._F(Data))
        return feg

    def Fegprime(self,Data):
        return self.divAAT(Data,self.Feg(Data))

    def Fcon(self,Data,gamma=.1):
        # gamma needs to be less than .1 --> try to find the stable range
        j = self.J(Data)
        return (np.eye(j.shape[0])+gamma*j.T).dot(self._F(Data))

    def Freg(self,Data,eta=0.1):
        # eta range is pretty stable .05 to 10 works, .01 is unstable
        dim = Data.size
        d = int((np.sqrt(4*dim+9)-3)/2)
        s = (d**2+d)//2
        _Fnow = self._F(Data)
        j = self.J(Data)
        jdg = j[:s+d,s+d:]  # upper right corner
        reg = 2*eta*np.dot(jdg.T,_Fnow[:s+d])
        _Fnow[s+d:] += reg
        return _Fnow

    def Falt(self,Data,alpha=5e-3):
        assert Data.size == 4
        w2,w1,a,b = Data.flatten()
        Fw2 = a**2-self.sig
        Fa = 2*alpha*a**3-2*a*(alpha*self.sig+w2)
        return np.hstack([Fw2.flatten(),np.array(0.),Fa.flatten(),np.array(0.)])

    def Funr(self,Data,alpha=5e-3):
        assert Data.size == 4
        w2,w1,a,b = Data.flatten()
        Fw2 = a**2-self.sig
        Fa = 4*alpha*a**3-2*a*(2*alpha*self.sig+w2)
        return np.hstack([Fw2.flatten(),np.array(0.),Fa.flatten(),np.array(0.)])

    def dist_Euclidean(self,Data):
        return np.linalg.norm(Data-self.xstar)

    def dist_KL(self,Data):
        W2 = self.upper2sym(Data[:self.s],self.dim)
        w1 = Data[self.s:self.s+self.dim]
        A = np.zeros((self.dim,self.dim))
        A[np.tril_indices(self.dim)] = Data[-self.s-self.dim:-self.dim]
        b = Data[-self.dim:]
        Asigma = np.dot(A,A.T)
        Asigmainv = np.linalg.pinv(Asigma)
        # print(Asigma[0,0])
        # KL = 0.5*(np.log(np.linalg.det(Asigma))-self.logsigdet - self.dim + np.trace(Asigmainv.dot(self.sig)) + np.dot((b-self.mu).T,Asigmainv.dot(b-self.mu)))
        KL = 0.5*(np.linalg.slogdet(Asigma)[1]-self.logsigdet - self.dim + np.trace(Asigmainv.dot(self.sig)) + np.dot((b-self.mu).T,Asigmainv.dot(b-self.mu)))
        return KL

    def dist(self,Data):
        return max(self.dist_KL(Data),self.dist_Euclidean(Data))

    def norm_F(self,Data):
        return np.linalg.norm(self._F(Data))/np.sqrt(self.Dim)

    def isNotNaNInf(self,Data):
        return ~np.any(np.logical_or(np.isnan(Data),np.isinf(Data)))

    def sym(self,mat):
        return 0.5*(mat+mat.T)

    def J(self,Data):
        dim = Data.size
        # dim = 2(s+d) = 2(1/2*d**2+d/2+d) = d**2+3d = dim
        d = int((np.sqrt(4*dim+9)-3)/2)
        s = (d**2+d)//2
        W2 = self.upper2sym(Data[:s],d)
        w1 = Data[s:s+d]
        A = np.zeros((d,d))
        A[np.tril_indices(d)] = Data[-s-d:-d]
        b = Data[-d:]

        jac = np.zeros((2*(s+d),2*(s+d)))

        jW2A = []
        for i in range(d):
            for j in range(i,d):
                for p in range(d):
                    for q in range(0,p+1):
                        if i==j==p:
                            jW2A += [-2*A[i,q]]
                        elif i==p:
                            jW2A += [-2*A[j,q]]
                        elif j==p:
                            jW2A += [-2*A[i,q]]
                        else:
                            jW2A += [0]
        jW2A = np.array(jW2A).reshape(s,s)
        jac[:s,s+d:s+d+s] = jW2A

        jW2b = []
        for i in range(d):
            for j in range(i,d):
                for p in range(d):
                    if i==j==p:
                        jW2b += [-2*b[i]]
                    elif i==p:
                        jW2b += [-2*b[j]]
                    elif j==p:
                        jW2b += [-2*b[i]]
                    else:
                        jW2b += [0]
        jW2b = np.array(jW2b).reshape(s,d)
        jac[:s,-d:] = jW2b

        jw1b = -np.eye(d)
        jac[s:s+d,-d:] = jw1b

        jAW2 = []
        for i in range(d):
            for j in range(0,i+1):
                for p in range(d):
                    for q in range(p,d):
                        if i==p:
                            jAW2 += [2*A[q,j]]
                        elif i==q:
                            jAW2 += [2*A[p,j]]
                        else:
                            jAW2 += [0]
        jAW2 = np.array(jAW2).reshape(s,s)
        jac[s+d:s+d+s,:s] = jAW2

        jAA = []
        for i in range(d):
            for j in range(0,i+1):
                for p in range(d):
                    for q in range(0,p+1):
                        if j==q:
                            jAA += [2*W2[i,p]]
                        else:
                            jAA += [0]
        jAA = np.array(jAA).reshape(s,s)
        jac[s+d:s+d+s,s+d:s+d+s] = jAA

        jbW2 = []
        for i in range(d):
            for p in range(d):
                for q in range(p,d):
                    if i==p:
                        jbW2 += [2*b[q]]
                    elif i==q:
                        jbW2 += [2*b[p]]
                    else:
                        jbW2 += [0]
        jbW2 = np.array(jbW2).reshape(d,s)
        jac[-d:,:s] = jbW2

        jbw1 = np.eye(d)
        jac[-d:,s:s+d] = jbw1

        jbb = []
        for i in range(d):
            for p in range(d):
                jbb += [2*W2[i,p]]
        jbb = np.array(jbb).reshape(d,d)
        jac[-d:,-d:] = jbb

        return -jac


def rand_mu_Sigma(dim=1):
    s = (dim**2+dim)//2
    mu = 10*np.random.rand(dim)
    L = 10*np.random.rand(dim,dim)-5
    L[range(dim),range(dim)] = np.clip(L[range(dim),range(dim)]+5.,1e-1,np.inf)
    L = np.tril(L)  # make diagonal dim times off diagonal on average
    # print(L)
    Sigma = np.dot(L,L.T)
    assert np.allclose(Sigma,Sigma.T)
    # try:
    #     print(np.linalg.cholesky(Sigma))
    # except:
    #     print(np.linalg.eigvals(Sigma))
    # embed()
    return mu, np.atleast_2d(Sigma)


def pmon(F,xf,x0,verbose=False):
    dx = xf-x0
    Fx0fx = np.dot(F(x0),dx)
    dxstar = xf-Domain.xstar
    if np.dot(F(xf),dxstar) < 0 and verbose:
        print('<F(x^*),x-x^*> < 0')
    if Fx0fx >= 0:
        Fxfdx = np.dot(F(xf),dx)
        if Fxfdx >= 0:
            return True
        else:
            if verbose:
                print(Fx0fx,Fxfdx)
            return False
    else:
        return True

def qmon(F,xf,x0,verbose=False):
    dx = xf-x0
    Fx0fx = np.dot(F(x0),dx)
    dxstar = xf-Domain.xstar
    if np.dot(F(xf),dxstar) < 0 and verbose:
        print('Not simple')
    if Fx0fx > 0:
        Fxfdx = np.dot(F(xf),dx)
        if Fxfdx >= 0:
            return True
        else:
            if verbose:
                print(Fx0fx,Fxfdx)
            return False
    else:
        return True


if __name__ == '__main__':
    # Creating a random LQGAN
    dim = 2
    s = (dim**2+dim)//2
    mu = np.zeros(dim)
    L = 10*np.random.rand(dim,dim)-5 + np.diag(5*np.ones(dim))
    L[range(dim),range(dim)] = np.clip(L[range(dim),range(dim)],1e-8,np.inf)
    L = np.tril(L)
    sig = np.dot(L,L.T)
    # sig = np.diag(np.random.rand(dim)/np.sqrt(2.))
    Domain = LQGAN(mu=mu,sig=sig)

    from VISolver.Projection import BoxProjection
    # Set Constraints
    loA = -np.inf*np.ones((dim,dim))
    loA[range(dim),range(dim)] = 1e-2
    lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
    P = BoxProjection(lo=lo)

    mx = -1
    for i in range(10000):
        Start = P.P(100*np.random.rand(Domain.Dim)-50.)
        jexact = Domain.J(Start)
        japprox = approx_jacobian(Domain._F,Start)
        newmx = np.max(np.abs(jexact-japprox))
        mx = max(mx,newmx)

    print(mx)


