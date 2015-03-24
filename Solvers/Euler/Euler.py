import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver


class Euler(Solver):

    def __init__(self, Domain, P=IdentityProjection()):
        self.F = Domain.f

        self.Proj = P

        self.StorageSize = 1

        self.temp_storage = {}

    def init_temp_storage(self, Start, Domain, Options):
        self.temp_storage['Data'] = self.StorageSize * [Start]
        self.temp_storage[self.F] = self.StorageSize * [self.F(Start)]
        self.temp_storage['scount'] = self.StorageSize * [0]
        self.temp_storage['s'] = self.StorageSize * [1]
        self.temp_storage['Step'] = self.StorageSize * [Options.Init.Step]
        self.temp_storage['f Evaluations'] = self.StorageSize * [1]
        self.temp_storage['Projections'] = self.StorageSize * [0]

        self.InitStep = Options.Init.Step

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    def update(self, record):
        # Retrieve Necessary Data
        Data = record.TempStorage['Data'][-1]
        F = record.TempStorage[self.F][-1]
        scount = self.temp_storage['scount'][-1]
        s = self.temp_storage['s'][-1]

        # Use Decreasing Step Size Scheme
        if scount >= s:
            scount = 0
            s += 1
        scount += 1
        Step = self.InitStep / s

        # Initialize Storage
        TempData = {}

        # Perform update
        NewData = self.Proj.P(Data, Step, F)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['f Evaluations'] = 1 + self.temp_storage['f Evaluations'][-1]
        TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]
        self.book_keeping(TempData)

        return self.temp_storage
