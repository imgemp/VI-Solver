import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class GempRK2(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-2,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        if hasattr(Domain,'Sigma'):
            self.Sigma = Domain.Sigma
        else:
            self.Sigma = 1.

        # Determine Batch Size
        k = 3.
        c = 5.*k
        N = max(int((c*self.Sigma)**2./Delta0/1.),1)
        print(N)
        N /= 10000
        self.N = N

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        Step = Record.TempStorage['Step'][-1]
        Fs = np.zeros((2,Data.shape[0]))

        # Initialize Storage
        TempData = {}

        # Perform Updates
        _NewData = np.zeros_like(Data)
        NewData = np.zeros_like(Data)
        F_avg = np.zeros_like(Data)

        for n in xrange(self.N):
            Fs[0,:] = self.F(Data)
            _NewData_n = self.Proj.P(Data,Step,Fs[0,:])
            Fs[1,:] = self.F(_NewData_n)
            NewData_n = self.Proj.P(Data,Step,0.5*np.sum(Fs,axis=0))

            # Note that convex combination of results will still be in K
            _NewData += _NewData_n/self.N
            NewData += NewData_n/self.N
            F_avg += 0.5*np.sum(Fs,axis=0)/self.N

        # Adjust Stepsize - P(|Diff|>=k/c*Delta) <= 1/k^2
        Delta = max(abs(NewData-_NewData))
        if Delta == 0:
            Step = 2 * Step
        else:
            growth = min((self.Delta0/Delta)**0.5, self.GrowthLimit)
            Step = max(min(Step*growth, self.MaxStep), self.MinStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = F_avg
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2*self.N + \
            self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2*self.N + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
