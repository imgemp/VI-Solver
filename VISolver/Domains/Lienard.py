from __future__ import division
import numpy as np
from VISolver.Domain import Domain

# from IPython import embed


class Lienard(Domain):

    def __init__(self):
        self.Dim = 2
        # self.Psi = np.eye(self.Dim)
        # self.Lyapunov = np.zeros(self.Dim)
        # self.iter = 0
        # self.lim = 10

    def F(self,Data):
        x,y = Data
        # x,y = Data[0:2]
        # psi = Data[2:].reshape(2,2)

        # self.Lyapunov += np.log(np.linalg.norm(psi,axis=0))

        # psi, R = np.linalg.qr(psi)

        dxy = np.array([y-.32*(x**5.)+4./3.*(x**3.)-.8*x,-x])
        # dpsi = np.dot(self.Jac(Data[0:2]),psi)

        # return np.concatenate((dxy,dpsi.flatten()))
        return dxy

    def Jac(self,Data):
        # x,y = Data[0:2]
        x,y = Data
        return np.array([[-5.*.32*(x**4.)+4.*(x**2.)-.8,1.],[-1.,0.]])

    def UpdateEllipsoid(self,Data):
        self.Psi += 0.01*np.dot(self.Jac(Data),self.Psi)
        # if self.Psi.shape != (2,1):
        #     embed()
        #     assert False
        self.Lyapunov += np.log(np.linalg.norm(self.Psi,axis=0))
        self.Psi, R = np.linalg.qr(self.Psi)
        # if self.iter % self.lim == 0:
            # self.Psi, R = np.linalg.qr(self.Psi)
        #     self.iter = 0
        # self.iter += 1
