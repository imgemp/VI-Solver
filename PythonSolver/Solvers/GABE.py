import numpy as np

from Projection import *
from Path import *
from Utilities import *
from Solver import Solver

class GABE(Solver):

    def __init__(self,Domain,P=IdentityProjection(),Delta0=1e-2,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.F = Domain.F

        self.Proj = P

        self.TempStorage = {'Data': [np.NaN,np.NaN], self.F: [np.NaN,np.NaN], 'Step': [np.NaN,np.NaN], 'F Evaluations': [np.NaN,np.NaN], 'Projections': [np.NaN,np.NaN]}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'][-1] = Start
        self.TempStorage[self.F][-1] = self.F(Start)
        self.TempStorage['Step'][-1] = Options.Init.Step
        self.TempStorage['F Evaluations'][-1] = 1
        self.TempStorage['Projections'][-1] = 0

        return self.TempStorage

    def BookKeeping(self,TempData):

        for item in self.TempStorage:
            self.TempStorage[item].pop(0)
            self.TempStorage[item].append(TempData[item])

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        _Data = Record.TempStorage['Data'][-2]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        if (Record.thisPermIndex%2 == 0):

            # Perform Euler Update
            F = Record.TempStorage[self.F][-1]
            NewData = self.Proj.P(Data,Step,F)

            # Record Projections
            TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        else:

            # Perform Euler Update
            Fs = Record.TempStorage[self.F]
            _NewData = self.Proj.P(Data,Step,Fs[-1])

            # Perform Adams Bashforth Update
            __NewData = self.Proj.P(Data,Step,-0.5*Fs[-2]+1.5*Fs[-1])

            # Approximate Gradient of ||F||^2
            grad_norm = (np.linalg.norm(Fs[-1])-np.linalg.norm(Fs[-2]))/(Data-_Data)
            # print(grad_norm)
            # print(Data-_Data)
            # grad_norm = 0.

            # Perform Adams Bashforth Update again with Gradient Norm
            NewData = self.Proj.P(Data,Step,-0.5*Fs[-2]+1.5*Fs[-1]-0.5*grad_norm) #divergence is zero for DummyMARL

            # Adjust Stepsize
            Delta = max(abs(__NewData-_NewData))
            if Delta == 0.: Step = 2.*Step
            else: Step = max(min(Step*min((self.Delta0/Delta)**0.5,self.GrowthLimit),self.MaxStep),self.MinStep)

            # Record Projections
            TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['F Evaluations'] = 1 + self.TempStorage['F Evaluations'][-1]
        self.BookKeeping(TempData)
        
        return self.TempStorage






