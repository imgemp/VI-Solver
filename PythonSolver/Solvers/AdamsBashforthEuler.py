import numpy as np

from Projection import *
from Path import *
from Utilities import *
from Solver import Solver

class ABEuler(Solver):

    def __init__(self,Domain,P=IdentityProjection(),Delta0=1e-2,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.F = Domain.F

        self.Proj = P

        self.TempStorage = {'Data': ['?','?'], self.F: ['?','?'], 'Step': ['?','?'], 'F Evaluations': ['?','?'], 'Projections': ['?','?']}

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

            # Perform Adams Bashforth Update
            Fs = Record.TempStorage[self.F]
            NewData = self.Proj.P(Data,Step,-0.5*Fs[0]+1.5*Fs[1])

            # Perform Euler Update
            _NewData = self.Proj.P(Data,Step,Fs[1])

            # Adjust Stepsize
            Delta = max(abs(NewData-_NewData))
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






