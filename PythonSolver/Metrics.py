import numpy as np

class Metric:

    def __init__(self):
        print('This is a generic metric object. You need to pick a specific metric to use.')

    def w(self,Data):
        return None

    def grad_w(self,Data):
        return None

class Entropy(Metric):

    def __init__(self,alpha=1.0,d=1e-16):
        self.alpha = alpha
        self.d = d

    def w(self,Data):
        n = Data.size
        return (Data+self.d/n)*np.log(Data+self.d/n)

    def grad_w(self,Data):
        n = Data.size
        return np.log(Data+self.d/n)+Data/(Data+self.d/n)