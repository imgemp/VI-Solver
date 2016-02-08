import numpy as np

from VISolver.Utilities import MachineLimit_Exp


class Projection(object):

    def P(self):
        '''This function projects the data.'''
        raise NotImplementedError(
            'Base classes of Projection must override the P method')


class IdentityProjection(Projection):

    def P(self,Data,Step=0.,Direc=0.):
        return Data+Step*Direc


class EntropicProjection(Projection):

    def P(self,Data,Step=0.,Direc=0.):
        ProjectedData = Data*np.exp(MachineLimit_Exp(Step,Direc)*Direc)
        return ProjectedData/np.sum(ProjectedData)


class EuclideanSimplexProjection(Projection):

    # Taken from: https://gist.github.com/daien/1272551
    def P(self,Data,Step=0.,Direc=0.,s=1):
        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        Data = Data + Step*Direc
        n, = Data.shape  # will raise ValueError if Data is not 1-D
        # check if we are already on the simplex
        if Data.sum() == s and np.alltrue(Data >= 0):
            # best projection: itself!
            return Data
        # get the array of cumulative sums of sorted (decreasing) copy of Data
        u = np.sort(Data)[::-1]
        cssd = np.cumsum(u)
        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n+1) > (cssd - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssd[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding Data using theta
        w = (Data - theta).clip(min=0)
        return w


class NormBallProjection(Projection):

    def __init__(self,p=2,axis=None):
        self.p = p
        self.axis = axis

    def P(self,Data,Step=0.,Direc=0.):
        un_norm = Data+Step*Direc
        return un_norm/np.linalg.norm(un_norm,ord=self.p,axis=self.axis)


class BoxProjection(Projection):

    def __init__(self,lo=-np.inf,hi=np.inf):
        self.min = lo
        self.max = hi

    def P(self,Data,Step=0.,Direc=0.):
        return np.clip(Data+Step*Direc,self.min,self.max)


class PolytopeProjection(Projection):

    # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # This function projects onto the polytope using L2 distance
    # Let x_k = Data + Step * Direc
    # min_x ||x-x_k||^2 s.t. Gx<=h, Ax=b
    def __init__(self,G=None,h=None,A=None,b=None):
        try:
            from cvxopt import solvers
            solvers.options['show_progress'] = False
            from cvxopt import matrix
        except ImportError:
            self.qp = None
            self.matrix = None
            raise ImportError('CVXOPT is required for '+self.__class__.__name__)
        else:
            self.qp = solvers.qp
            self.matrix = matrix
        self.G = self.h = self.A = self.b = None
        if (G is not None) and (h is not None):
            self.G = self.matrix(G,tc='d')
            self.h = self.matrix(h,tc='d')
            self.case = 1
        if (A is not None) and (b is not None):
            self.A = self.matrix(A,tc='d')
            self.b = self.matrix(b,tc='d')
            self.case = 2
        if (self.G is not None) and (self.A is not None):
            self.case = 3
        if (self.G is None) and (self.A is None):
            err = 'G & h and/or A & b should be specified as numpy arrays'
            raise TypeError(err)

    def P(self,Data,Step=0.,Direc=0.):
        P = self.matrix(2.*np.identity(len(Data)),tc='d')
        q = self.matrix(-2.*(Data+Step*Direc),tc='d')
        # cvx quad prog solver returns a matrix object, not numpy array
        if self.case == 1:
            NewData = self.qp(P,q,G=self.G,h=self.h)['x']
        elif self.case == 2:
            NewData = self.qp(P,q,A=self.A,b=self.b)['x']
        elif self.case == 3:
            NewData = self.qp(P,q,self.G,self.h,self.A,self.b)['x']
        else:
            raise NotImplementedError('Unexpected error: no match for case!')
        return np.reshape(NewData,Data.shape)
