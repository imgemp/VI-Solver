import numpy as np

from VISolver.Domain import Domain


class Rosenbrock(Domain):
    '''
    Multidimensional generalization of Rosenbrock function to Dim/2 uncoupled
    2D Rosenbrock problems: https://en.wikipedia.org/wiki/Rosenbrock_function
    '''
    def __init__(self,Dim=2,Newton=False):
        if Dim % 2:
            print('Dim will be rounded up to even number.')
        self.Dim = Dim + Dim % 2
        self.Min = 0.0
        self.ArgMin = np.ones(self.Dim)
        if Newton:
            self.F = self.N
        else:
            self.F = self.Grad
        self.Newton = Newton

    def f(self,Data):
        return np.sum(100.*(Data[0::2]**2-Data[1::2])**2.+(Data[0::2]-1)**2.)

    def Grad(self,Data):
        g = np.zeros(Data.shape)
        g[0::2] = 400.*Data[0::2]*(Data[0::2]**2.-Data[1::2])+2.*(Data[0::2]-1)
        g[1::2] = -200.*(Data[0::2]**2.-Data[1::2])
        return g

    def H(self,Data):
        d2even = 1200.*Data[0::2]**2.-400.*Data[1::2]+2
        d2odd = 200.*np.ones(self.Dim//2)
        diag_0 = np.vstack((d2even,d2odd)).ravel(order='F')
        d20 = np.zeros(self.Dim//2)
        d2evod = -400.*Data[0::2]
        diag_1 = np.vstack((d2evod,d20)).ravel(order='F')[:-1]
        return np.diag(diag_0) + np.diag(diag_1,k=1) + np.diag(diag_1,k=-1)

    def N(self,Data):
        return np.linalg.pinv(self.H(Data)).dot(self.Grad(Data))

    def f_Error(self,Data):
        return self.f(Data) - self.Min
