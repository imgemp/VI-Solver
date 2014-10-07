import numpy as np

#Domains
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

    def Report(self,Data,Requests):
        Rep = dict()
        for req in Requests:
            Rep[req] = self.Fun[req](Data)
        return Rep

    def AddDefaultReports(self,CustomReports):
        CustomReports['GenError'] = self.GenError
        return CustomReports

    def GenError(self,Data):
        #this is specific to simplex constraint - need to generalize this with max and min of feasible set
        error = 0.0
        F = np.ravel(self.F(Data))
        count = 0
        z = 1.0
        for ind in abs(F).argsort()[::-1]:
            if (F[ind] < 0) or (count == len(F)-1):
                diff = Data[ind]-z
                error += F[ind]*diff
                count += 1
                z = 0.0
            else:
                diff = Data[ind]-0.0
                error += F[ind]*diff
                count += 1
        return error

class Sphere(Domain):

    def __init__(self,Dim=None):
        self.Dim = Dim;
        self.Min = 0.0
        self.L = 2.0
        self.F = self.Gradient
        self.Fun = self.AddDefaultReports({'Value':self.Value,'Gradient':self.Gradient,'Error':self.Error})

    def Value(self,Data):
        return np.sum(Data**2)

    def Gradient(self,Data):
        return 2.0*Data

    def Error(self,Data):
        return self.Value(Data)-self.Min

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