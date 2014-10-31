import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver

class AG(Solver):

    def __init__(self,Domain,P=IdentityProjection()):
        
        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage['_Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['scount'] = self.StorageSize*[0]
        self.TempStorage['s'] = self.StorageSize*[1]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        self.InitStep = Options.Init.Step

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record): # A lot of weird things with AG - track gap(_x), x updated with _x, look into these

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        _Data = Record.TempStorage['_Data'][-1]
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
        NewData = self.Proj.P(_Data,Step,F)
        ext = (Record.thisPermIndex-1)/(Record.thisPermIndex+2)
        _NewData = NewData + ext*(NewData - Data)

        # Store Data
        TempData['Data'] = NewData
        TempData['_Data'] = _NewData
        TempData[self.F] = self.F(_NewData)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['F Evaluations'] = 1 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)
        
        return self.TempStorage






