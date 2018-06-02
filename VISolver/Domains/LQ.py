import numpy as np

from VISolver.Domain import Domain

from IPython import embed
class LQ(Domain):

    def __init__(self,sig=1):
        self.Dim = 2
        self.sig = sig

    def F(self,Data):
        # return -np.array([1-Data[1]**2,2*Data[0]*Data[1]])
        # return self._F(Data)
        return self.RipCurl(Data)
        # return self.EG(Data)
        # return self.GReg(Data)

    def _F(self,Data):
        return -np.array([self.sig**2.-Data[1]**2,2*Data[0]*Data[1]])
        # return np.array([-Data[0]+Data[1],-Data[1]-Data[0]])

    def J(self,Data):
        return -np.array([[0,-2*Data[1]],[2*Data[1],2*Data[0]]])
        # return np.array([[-1,1],[-1,-1]])

    def T(self,Data):
        # 0, 2y
        # -2y, -2x
        t = np.zeros((2,2,2))
        t[0,1,1] = 2
        t[1,0,1] = -2
        t[1,1,0] = -2
        return t

    def TF(self,Data):
        t = self.T(Data)
        f = self._F(Data)
        tf = np.zeros((2,2))
        for i in range(2):
            for k in range(2):
                tf[i,k] = np.dot(t[:,i,k] - t[i,:,k],f)
        return tf

    # TF_00 = 0
    # TF_01 = 4*(2xy)
    # TF_10 = 0
    # TF_11 = -4*(sig^2-y^2)

    def JRipCurl(self,Data,gam=0.):
        j = self.J(Data)
        tf = self.TF(Data)
        jrc = (gam*np.eye(2)+(1-gam)*(j.T-j)).dot(j) + (1-gam)*tf
        return jrc

    def JRCEV(self,Data,gam=0.):
        return np.linalg.eigvals(self.sym(self.JRipCurl(Data,gam)))

    def JReg(self,Data,gam=0.):
        j = self.J(Data)
        jreg = (gam*np.eye(2)+(1-gam)*(j.T-j)).dot(j)
        return jreg

    def JRegEV(self,Data,gam=0.):
        return np.linalg.eigvals(self.sym(self.JReg(Data,gam)))

    def RipCurl(self,Data):
        j = self.J(Data)
        jsym = 0.5*(j+j.T)
        jasy = 0.5*(j-j.T)*2.
        jsym_norm = np.linalg.norm(jsym)
        # jasy_norm = np.linalg.norm(jasy)
        jasy_norm = np.linalg.norm(np.dot(jasy,jasy)*2.)
        norm = jsym_norm+jasy_norm
        if norm > 0:
            # b = jsym_norm/norm
            # b = 0.5
            # if jasy_norm > jsym_norm:
            #     b = 0.
            # else:
            #     b = 1.
            # symlo = np.linalg.eigvals(jsym).min()
            # asylo = np.linalg.eigvals(np.dot(jasy,jasy)*2.).min()
            # print(symlo,asylo)
            # if symlo + asylo > 0:
            #     b = 1.
            #     print('got here')
            # else:
            #     b = 0.
            # jtemp = jsym + 2*np.dot(jasy,jasy)
            # if np.linalg.eigvals(jtemp).min() > 0:
            #     b = 1.
            # else:
            #     b = 0.
            # f = self._F(Data)
            # fnorm = np.linalg.norm(f)
            # jsym_norm2 = np.linalg.norm(np.dot(jsym,f))/fnorm
            # jasyf = np.dot(jasy,f)
            # jasy_norm2 = np.linalg.norm(np.dot(jasyf,jasyf)*2.)/(fnorm**2.)
            # # if jasy_norm2 > jsym_norm2:
            # if jasy_norm2 > 0:
            #     b = 0.
            # else:
            #     b = 1.
            b = 1e-1
            # out = (b*np.eye(j.shape[0])-(1-b)*jasy).dot(self._F(Data))
            # return out/np.linalg.norm(out)
            return (b*np.eye(j.shape[0])-(1-b)*jasy).dot(self._F(Data))
        else:
            return self._F(Data)

    def EG(self,Data):
        j = self.J(Data)
        jsym = 0.5*(j+j.T)
        jasy = 0.5*(j-j.T)
        jsym_norm = np.linalg.norm(jsym)
        jasy_norm = np.linalg.norm(jasy)
        norm = jsym_norm+jasy_norm
        if norm > 0:
            b = jsym_norm/norm
            jeg = 0.5*j
            return (b*np.eye(j.shape[0])-(1-b)*jeg).dot(self._F(Data))
        else:
            return self._F(Data)

    def GReg(self,Data):
        j = self.J(Data)
        jsym = 0.5*(j+j.T)
        jasy = 0.5*(j-j.T)
        jsym_norm = np.linalg.norm(jsym)
        jasy_norm = np.linalg.norm(jasy)
        norm = jsym_norm+jasy_norm
        if norm > 0:
            b = jsym_norm/norm
            # b = 1.
            jgreg = -0.5*j.T
            return (b*np.eye(j.shape[0])-(1-b)*jgreg).dot(self._F(Data))
        else:
            return self._F(Data)

    def dist(self,Data):
        return np.linalg.norm(Data-np.array([0,self.sig]))

    def sym(self,mat):
        return 0.5*(mat+mat.T)

    def Jmult(self,Data):
        dim = Data.size
        # 2*d*(d+1) = dim
        d = int((np.sqrt(2*dim+1)-1)/2)
        d2 = d**2
        s = (d2+d)//2
        W2 = Data[:d**2].reshape(d,d)
        w1 = Data[d**2:d**2+d]
        A = Data[-d*(d+1):-d].reshape(d,d)
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
                            jW2A += [-A[j,q]]
                        elif j==p:
                            jW2A += [-A[i,q]]
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
                        jW2b += [-b[j]]
                    elif j==p:
                        jW2b += [-b[i]]
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

        return jac







