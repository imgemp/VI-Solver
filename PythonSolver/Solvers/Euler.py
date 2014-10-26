import numpy as np

from Projection import *
from Path import *
from Utilities import *
from Solver import Solver

class Euler(Solver):

    def __init__(self,Domain,P=IdentityProjection()):
        
        self.F = Domain.F

        self.Proj = P

        self.TempStorage = {'Data': ['?'], self.F: ['?'], 'scount': ['?'], 's': ['?'], 'Step': ['?'], 'F Evaluations': ['?'], 'Projections': ['?']}

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'][-1] = Start
        self.TempStorage[self.F][-1] = self.F(Start)
        self.TempStorage['scount'][-1] = 0
        self.TempStorage['s'][-1] = 1
        self.TempStorage['Step'][-1] = Options.Init.Step
        self.TempStorage['F Evaluations'][-1] = 1
        self.TempStorage['Projections'][-1] = 0

        self.InitStep = Options.Init.Step

        return self.TempStorage

    def BookKeeping(self,TempData):

        for item in self.TempStorage:
            self.TempStorage[item].pop(0)
            self.TempStorage[item].append(TempData[item])

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        F = Record.TempStorage[self.F][-1]
        scount = self.TempStorage['scount'][-1]
        s = self.TempStorage['s'][-1]

        # Use Decreasing Step Size Scheme
        if scount >= s:
            scount = 0;
            s += 1;
        scount += 1;
        Step = self.InitStep/s

        # Initialize Storage
        TempData = {}

        # Perform Update
        NewData = self.Proj.P(Data,Step,F)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['F Evaluations'] = 1 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)
        
        return self.TempStorage






