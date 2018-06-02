import numpy as np

from VISolver.Domain import Domain


class Affine(Domain):

    def __init__(self, A=np.eye(2), b=np.zeros(2)):
        self.Dim = b.shape[0]
        self.A = A
        self.b = b

    def F(self,Data):
        return np.dot(self.A,Data)+self.b

    def J(self,Data):
        return self.A
