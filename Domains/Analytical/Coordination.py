import numpy as np

from Domain import Domain


class Coordination(Domain):

    def __init__(self):
        self.Dim = 2
        self.r = np.array([[2., 0.], [0., 1.]])
        self.c = np.array([[1., 0.], [0., 2.]])
        self.u = self.u()
        self.uprime = self.uprime()
        self.A = np.array([[0., self.u], [self.uprime, 0.]])
        self.b = np.array(
            [-(self.r[1, 1] - self.r[0, 1]), -(self.c[1, 1] - self.c[1, 0])])
        self.A_curl = np.array(
            [[2. * self.uprime ** 2., 0], [0, 2. * self.u ** 2.]])
        self.b_curl = np.array([-
                                2. *
                                self.uprime *
                                (self.c[1, 1] -
                                 self.c[1, 0]), -
                                2. *
                                self.u *
                                (self.r[1, 1] -
                                    self.r[0, 1])])
        self.NE = np.array([[0., 0.], [1., 1.]])  # 2 pure NE

    def u(self):
        return (self.r[0, 0] + self.r[1, 1]) - (self.r[1, 0] + self.r[0, 1])

    def uprime(self):
        return (self.c[0, 0] + self.c[1, 1]) - (self.c[1, 0] + self.c[0, 1])

    def F(self, Data):
        return self.A.dot(Data) + self.b

    def F_curl(self, Data):
        return 0.5 * self.A_curl.dot(Data) + self.b_curl

    def NE_L2Error(self, Data):
        return np.min(
            np.array([np.linalg.norm(Data - self.NE[0]), np.linalg.norm(Data - self.NE[1])]))
