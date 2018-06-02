import numpy as np

# from VISolver.Domain import Domain

from IPython import embed
class LQ(object):

    def __init__(self,mu=0,sig=1,method='ccGD'):
        self.Dim = 3*mu.size + mu.size**2
        self.mu = mu
        self.sig = sig
        self.d = int((np.sqrt(4*self.Dim+9)-3)/2)
        self.s = (self.d**2+self.d)//2
        sigsqrt = np.linalg.cholesky(sig)
        self.xstar = np.hstack([np.zeros(self.s),np.zeros(self.d),sigsqrt[np.tril_indices(self.d)],self.mu])
        self.method = method

    def F(self,Data):
        if self.method == 'ccGD':
            return self.RipCurl(Data)
        elif self.method == 'simGD':
            return self._F(Data)
        elif self.method == 'preEG':
            return self.EG(Data)
        elif self.method == 'conGD':
            return self.conGD(Data)
        elif self.method == 'regGD':
            return self.regGD(Data)
        else:
            assert False, self.method+' is not a valid option'

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
        w1dot = self.mu - b
        Adot = np.dot(W2+W2.T,A)
        bdot = np.dot(W2+W2.T,b) + w1

        return -np.hstack([W2dot[np.triu_indices(d)],w1dot,Adot[np.tril_indices(d)],bdot])

    def JReg(self,Data,gam=0.):
        j = self.J(Data)
        jreg = (gam*np.eye(2)+(1-gam)*(j.T-j)).dot(j)
        return jreg

    def JRegEV(self,Data,gam=0.):
        return np.linalg.eigvals(self.sym(self.JReg(Data,gam)))

    def RipCurl(self,Data):
        j = self.J(Data)
        # jsym = 0.5*(j+j.T)
        jasy = 0.5*(j-j.T)*2.
        # jsym_norm = np.linalg.norm(jsym)
        # # jasy_norm = np.linalg.norm(jasy)
        # jasy_norm = np.linalg.norm(np.dot(jasy,jasy)*2.)
        # norm = jsym_norm+jasy_norm
        # if norm > 0:
        #     # b = jsym_norm/norm
        #     # b = 0.5
        #     # if jasy_norm > jsym_norm:
        #     #     b = 0.
        #     # else:
        #     #     b = 1.
        #     # symlo = np.linalg.eigvals(jsym).min()
        #     # asylo = np.linalg.eigvals(np.dot(jasy,jasy)*2.).min()
        #     # print(symlo,asylo)
        #     # if symlo + asylo > 0:
        #     #     b = 1.
        #     #     print('got here')
        #     # else:
        #     #     b = 0.
        #     # jtemp = jsym + 2*np.dot(jasy,jasy)
        #     # if np.linalg.eigvals(jtemp).min() > 0:
        #     #     b = 1.
        #     # else:
        #     #     b = 0.
        #     # f = self._F(Data)
        #     # fnorm = np.linalg.norm(f)
        #     # jsym_norm2 = np.linalg.norm(np.dot(jsym,f))/fnorm
        #     # jasyf = np.dot(jasy,f)
        #     # jasy_norm2 = np.linalg.norm(np.dot(jasyf,jasyf)*2.)/(fnorm**2.)
        #     # # if jasy_norm2 > jsym_norm2:
        #     # if jasy_norm2 > 0:
        #     #     b = 0.
        #     # else:
        #     #     b = 1.
        #     b = 0.
        #     # out = (b*np.eye(j.shape[0])-(1-b)*jasy).dot(self._F(Data))
        #     # return out/np.linalg.norm(out)
        #     return (b*np.eye(j.shape[0])-(1-b)*jasy).dot(self._F(Data))
        # else:
        #     return self._F(Data)
        # b = 1e-1
        # b = 0.
        # return (b*np.eye(j.shape[0])-(1-b)*jasy).dot(self._F(Data))/(4*Data[2]**2.)
        return -jasy.dot(self._F(Data))


# 2976.29572593 -3126.33854465
# Not Quasi-Monotone! (2-d) case with w1,b pre-learned
# [ 2.33710534 -4.94321874  4.94915082]
# [ 1.29955671  5.70229707  4.76890201]
# [ 2.16408901 -4.24149937  3.05794442]
# [ 5.87582115  1.46982638  6.77473149]
# 250.654647152 -1093.05816353
# Not Quasi-Monotone! (2-d) case with w1,b pre-learned
# [-0.94770152 -1.81990963  3.13808917]
# [ 6.19679669  1.30257241  3.75441103]
# [-1.50147502 -3.80603767  4.82562736]
# [ 1.48182478  6.6293112   1.98738971]
    def EG(self,Data):
        j = self.J(Data)
        # jsym = 0.5*(j+j.T)
        # jasy = 0.5*(j-j.T)
        # jsym_norm = np.linalg.norm(jsym)
        # jasy_norm = np.linalg.norm(jasy)
        # norm = jsym_norm+jasy_norm
        # if norm > 0:
        #     b = jsym_norm/norm
        #     jeg = 0.5*j
        #     return (b*np.eye(j.shape[0])-(1-b)*jeg).dot(self._F(Data))
        # else:
        #     return self._F(Data)
        # return -j.dot(self._F(Data))/(4*Data[2]**2.)
        return -j.dot(self._F(Data))

    def conGD(self,Data,gamma=.1):
        # gamma needs to be less than .1 --> try to find the stable range
        j = self.J(Data)
        # jsym = 0.5*(j+j.T)
        # jasy = 0.5*(j-j.T)
        # jsym_norm = np.linalg.norm(jsym)
        # jasy_norm = np.linalg.norm(jasy)
        # norm = jsym_norm+jasy_norm
        # if norm > 0:
        #     b = jsym_norm/norm
        #     # b = 1.
        #     jgreg = -0.5*j.T
        #     return (b*np.eye(j.shape[0])-(1-b)*jgreg).dot(self._F(Data))
        # else:
        #     return self._F(Data)
        # print(Data)
        # print((np.eye(j.shape[0])+gamma*j.T).dot(self._F(Data)))
        return (np.eye(j.shape[0])+gamma*j.T).dot(self._F(Data))

    def regGD(self,Data,eta=1.):
        # eta range is pretty stable .05 to 10 works, .01 is unstable
        # assert Data.size == 4
        # w2,w1,a,b = Data
        # Fw2 = -self.sig**2 + a**2 + b**2
        # Fw1 = b
        # Fa = -2*w2*a + 4*eta*a*(-self.sig**2 + a**2 + b**2)
        # Fb = -2*w2*b - w1 + 4*eta*b*(-self.sig**2 + a**2 + b**2) + 2*eta*b
        # return np.array([Fw2,Fw1,Fa,Fb])
        dim = Data.size
        d = int((np.sqrt(4*dim+9)-3)/2)
        s = (d**2+d)//2
        _Fnow = self._F(Data)
        j = self.J(Data)
        jdg = j[:s+d,s+d:]  # upper right corner
        reg = 2*eta*np.dot(jdg.T,_Fnow[:s+d])
        _Fnow[s+d:] += reg
        return _Fnow

    def dist(self,Data):
        return np.linalg.norm(Data-self.xstar)

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
        # embed()
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


def pmon(F,xf,x0):
    dx = xf-x0
    Fx0fx = np.dot(F(x0),dx)
    dxstar = xf-Domain.xstar
    if np.dot(F(xf),dxstar) < 0:
        print('Not simple')
    if Fx0fx >= 0:
        Fxfdx = np.dot(F(xf),dx)
        if Fxfdx >= 0:
            return True
        else:
            print(Fx0fx,Fxfdx)
            return False
    else:
        return True

def qmon(F,xf,x0):
    dx = xf-x0
    Fx0fx = np.dot(F(x0),dx)
    dxstar = xf-Domain.xstar
    if np.dot(F(xf),dxstar) < 0:
        print('Not simple')
    if Fx0fx > 0:
        Fxfdx = np.dot(F(xf),dx)
        if Fxfdx >= 0:
            return True
        else:
            print(Fx0fx,Fxfdx)
            return False
    else:
        return True


def terms(w2,w1,a,b,dw2,dw1,da,db):       
    ab2 = a**2 + b**2
    alpha_terms = ab2*dw2+b*dw1-2*w2*a*da-(2*w2*b+w1)*db
    gamma_terms = -8*w2**2*a*da-4*w2*(2*w2*b+w1)*db
    sigalpha_term = -dw2
    sig_terms = -4*a*da-4*b*db                                                     
    others = 8*w2*ab2*dw2+4*w1*b*dw2+4*w2*b*dw1+2*w1*dw1+4*a*ab2*da+8*w2**2*a*da+4*b*ab2*db+2*b*db+4*w2*(2*w2*b+w1)*db
    return alpha_terms,gamma_terms,sigalpha_term,sig_terms,others

def getterms(w2y,w1y,ay,by,w2x,w1x,ax,bx):
    dw2 = w2x-w2y    
    dw1 = w1x-w1y                                       
    da = ax-ay                                     
    db = bx-by          
    return terms(w2y,w1y,ay,by,dw2,dw1,da,db), terms(w2x,w1x,ax,bx,dw2,dw1,da,db)

# x0 = np.array([ 4.05257091,  3.36347181,  1.20818115, -2.2366073 ])
# xf = np.array([ 4.75055746, -0.50926053,  0.42587375,  1.18763973])
# getterms(*x0,*xf)
#getterms(3.9,-1.99982,.01,.25,1.,2.,1.,1.8)
def search(T=100):
    y = np.array([3.9,-2.,.01,.25])
    x = np.array([1.,2.,1.,1.8])
    for t in range(T):
        delta1 = np.random.rand(4)*.3-.15
        delta2 = np.random.rand(4)*.3-.15
        yterms, xterms = getterms(*(y+delta1),*(x+delta2))
        if yterms[0]>0 and yterms[1]>0 and yterms[0]+0.5*yterms[2]>0 and yterms[4]>0.5*yterms[3]:
            if xterms[0]<0 and xterms[1]<0 and yterms[0]+0.5*yterms[2]<0:
                return y+delta1,x+delta2,xterms,yterms


if __name__ == '__main__':
    dim = 2
    s = (dim**2+dim)//2
    mu = np.zeros(dim)
    # mu = np.array([1])
    L = 10*np.random.rand(dim,dim)-5 + np.diag(5*np.ones(dim))
    L[range(dim),range(dim)] = np.clip(L[range(dim),range(dim)],1e-8,np.inf)
    L = np.tril(L)
    sig = np.dot(L,L.T)
    sig = np.diag(np.random.rand(dim)/np.sqrt(2.))
    # sig = np.array([[1e-4]])
    # sig = np.array([[1]])
    # sig = np.array([[9,2],[2,4]])
    # w2 = np.array([1,2,-1])
    # w1 = np.zeros(dim)
    # a = np.array([1,2,1])
    # b = np.zeros(dim)
    # x = np.hstack([w2,w1,a,b])

    # print('mu, sig, sig eigs')
    # print(mu)
    # print(sig)
    # print(np.linalg.eigvals(sig))

    Domain = LQ(mu=mu,sig=sig)

    scale = 1000

    for t in range(10000):
        # x0w2 = np.diag(np.random.rand(Domain.d)*10-5)[np.triu_indices(Domain.d)]  # random diagonal w2
        # x0w2 = np.triu(np.random.rand(Domain.d,Domain.d)*100-50,k=1)[np.triu_indices(Domain.d)]  # random w2 with zero diagonal
        x0w2 = np.random.rand(Domain.s)*scale-0.5*scale  # random w2
        x0w1 = np.zeros(Domain.d)
        # x0w1 = np.random.rand(Domain.d)*10-5
        # x0a = np.diag(np.random.rand(Domain.d)*10)[np.tril_indices(Domain.d)]  # random diagonal a
        # x0a = (np.tril(np.random.rand(Domain.d,Domain.d)*100-50,k=-1) + np.diag(np.diag(L)))[np.tril_indices(Domain.d)]  # random a with correct diagonal
        x0a = np.random.rand(Domain.s)*scale
        x0b = mu
        # x0b = np.random.rand(Domain.d)*10-5
        x0 = np.hstack([x0w2,x0w1,x0a,x0b])
        # xfw2 = np.diag(np.random.rand(Domain.d))[np.triu_indices(Domain.d)]  # random diagonal w2
        # xfw2 = np.triu(np.random.rand(Domain.d,Domain.d)*100-50,k=1)[np.triu_indices(Domain.d)]  # random w2 with zero diagonal
        xfw2 = np.random.rand(Domain.s)*scale-0.5*scale  # random w2
        xfw1 = np.zeros(Domain.d)
        # xfw1 = np.random.rand(Domain.d)*10-5
        # xfa = np.diag(np.random.rand(Domain.d)*10)[np.tril_indices(Domain.d)]  # random diagonal a
        # xfa = (np.tril(np.random.rand(Domain.d,Domain.d)*100-50,k=-1) + np.diag(np.diag(L)))[np.tril_indices(Domain.d)]  # random a with correct diagonal
        xfa = np.random.rand(Domain.s)*scale
        xfb = mu
        # xfb = np.random.rand(Domain.d)*10-5
        xf = np.hstack([xfw2,xfw1,xfa,xfb])
        if not qmon(Domain.F,xf,x0):
            print('Not Quasi-Monotone!')
            print(x0w2)
            print(x0a)
            print(xfw2)
            print(xfa)
            embed()
            break
        if not pmon(Domain.F,xf,x0):
            print('Not Pseudo-Monotone!')
            print(x0w2)
            print(x0a)
            print(xfw2)
            print(xfa)
            embed()
            break

    # print(Domain._F(x))
    embed()


