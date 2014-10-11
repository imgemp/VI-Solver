import numpy as np

class Domain:

    def __init__(self):
        print('This is a generic domain object.  You need to pick a specific domain to use.')

    def CheckRequests(self,Requests):
        Requests = list(set(Requests))
        for req in Requests:
            if not (req in self.Fun.keys()):
                Requests.remove(req)
                print(req+' is not reported by this domain.')
        return Requests

class Sphere(Domain):

    def __init__(self,Dim=None):
        self.Dim = Dim;
        self.Min = 0.0
        self.L = 2.0

    def f(self,Data):
        return np.sum(Data**2)

    def F(self,Data):
        return 2.0*Data

    def f_Error(self,Data):
        return self.f(Data)-self.Min

class KojimaShindo(Domain):

    def __init__(self):
        self.Dim = 4;
        self.Min = 0.0
        self.L = 10.0
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Gradient':self.Gradient})

    def Gradient(self,Data):
        Grad = np.array(Data)
        x1 = Data[0]
        x2 = Data[1]
        x3 = Data[2]
        x4 = Data[3]
        Grad[0] = 3*(x1**2)+2*x1*x2+2*(x2**2)+x3+3*x4-6
        Grad[1] = 2*(x1**2)+x1+(x2**2)+10*x3+2*x4-2
        Grad[2] = 3*(x1**2)+x1*x2+2*(x2**2)+2*x3+9*x4-9
        Grad[3] = (x1**2)+3*(x2**2)+2*x3+3*x4-3
        return Grad

class Watson(Domain):

    def __init__(self,Pos=0):
        self.Dim = 10;
        self.Min = 0.0
        self.L = 10.0
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Gradient':self.Gradient})
        self.Pos = Pos
        self.A = np.double(np.array([[0,0,-1,-1,-1,1,1,0,1,1],
                                [-2,-1,0,1,1,2,2,0,-1,0],
                                [1,0,1,-2,-1,-1,0,2,0,0],
                                [2,1,-1,0,1,0,-1,-1,-1,1],
                                [-2,0,1,1,0,2,2,-1,1,0],
                                [-1,0,1,1,1,0,-1,2,0,1],
                                [0,-1,1,0,2,-1,0,0,1,-1],
                                [0,-2,2,0,0,1,2,2,-1,0],
                                [0,-1,0,2,2,1,1,1,-1,0],
                                [2,-1,-1,0,1,0,0,-1,2,2]]))
        self.b = np.zeros(self.Dim)
        self.b[self.Pos] = 1.0    

    def Gradient(self,Data):
        return np.dot(self.A,Data)+self.b

class Sun(Domain):

    def __init__(self,Dim=8000):
        self.Dim = Dim;
        self.Min = 0.0
        self.L = (2.0*np.double(Dim))**2
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Gradient':self.Gradient})
        L = np.zeros((self.Dim,self.Dim))
        U = np.triu(2*np.ones((self.Dim,self.Dim)),1)
        D = np.diag(np.ones(self.Dim),0)
        self.A = L+U+D
        self.b = -1*np.ones(self.Dim)

    def Gradient(self,Data):
        return np.dot(self.A,Data)+self.b

class MHPH(Domain):

    def __init__(self,Dim=1000):
        self.Dim = Dim;
        self.Min = 0.0
        self.L = (15.0*np.double(Dim))**2
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Gradient':self.Gradient})
        M = np.random.uniform(low=-15,high=-12,size=(self.Dim,self.Dim))
        self.A = np.dot(M,M.T)
        self.b = np.random.uniform(low=-500,high=0,size=self.Dim)

    def Gradient(self,Data):
        return np.dot(self.A,Data)+self.b

class RG(Domain):

    def __init__(self,Dim=1000):
        self.Dim = Dim;
        self.Min = 0.0
        self.L = (15.0*np.double(Dim))**2
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Gradient':self.Gradient})
        self.A = np.random.uniform(low=-50,high=150,size=(self.Dim,self.Dim))
        self.b = np.random.uniform(low=-200,high=300,size=self.Dim)

    def Gradient(self,Data):
        return np.dot(self.A,Data)+self.b






