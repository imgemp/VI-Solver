from Projection import IdentityProjection
from Solver import Solver


class EG(Solver):

    def __init__(self,Domain,P=IdentityProjection()):

        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['scount'] = self.StorageSize*[0]
        self.TempStorage['s'] = self.StorageSize*[1]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        self.InitStep = Options.Init.Step

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        F = Record.TempStorage[self.F][-1]
        scount = self.TempStorage['scount'][-1]
        s = self.TempStorage['s'][-1]

        # Use Decreasing Step Size Scheme
        if scount >= s:
            scount = 0
            s += 1
        scount += 1
        Step = self.InitStep/s

        # Initialize Storage
        TempData = {}

        # Perform Update
        _NewData = self.Proj.P(Data,Step,F)
        NewData = self.Proj.P(Data,Step,self.F(_NewData))

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
