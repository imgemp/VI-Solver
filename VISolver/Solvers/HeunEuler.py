import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class HeunEuler(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-2,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

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
        Fs = np.zeros((2,Data.shape[0]))
        Fs[0,:] = Record.TempStorage[self.F][-1]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Perform Update
        _NewData = self.Proj.P(Data,Step,Fs[0,:])
        Fs[1,:] = self.F(_NewData)
        NewData = self.Proj.P(Data,Step,0.5*np.sum(Fs,axis=0))

        # Adjust Stepsize
        Delta = max(abs(NewData-_NewData))
        if Delta == 0:
            growth = self.GrowthLimit
        else:
            growth = min((self.Delta0/Delta)**0.5, self.GrowthLimit)
        Step = np.clip(growth*Step,self.MinStep,self.MaxStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
