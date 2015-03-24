import numpy as np

from VISolver.Projection import *
from VISolver.Utilities import *
from Solver import Solver


class AG(Solver):

    def __init__(self, domain, p=IdentityProjection()):
        self.F = domain.f

        self.Proj = p

        self.StorageSize = 1

        self.temp_storage = {}

    def init_temp_storage(self, Start, Domain, Options):
        self.temp_storage['Data'] = self.StorageSize * [Start]
        self.temp_storage['_Data'] = self.StorageSize * [Start]
        self.temp_storage[self.F] = self.StorageSize * [self.F(Start)]
        self.temp_storage['scount'] = self.StorageSize * [0]
        self.temp_storage['s'] = self.StorageSize * [1]
        self.temp_storage['Step'] = self.StorageSize * [Options.Init.Step]
        self.temp_storage['f Evaluations'] = self.StorageSize * [1]
        self.temp_storage['Projections'] = self.StorageSize * [0]

        self.InitStep = Options.Init.Step

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    # A lot of weird things with AG - track gap(_x), x updated with _x, look
    # into these
    def update(self, record):

        # Retrieve Necessary Data
        Data = record.TempStorage['Data'][-1]
        _Data = record.TempStorage['_Data'][-1]
        F = record.TempStorage[self.F][-1]
        scount = self.temp_storage['scount'][-1]
        s = self.temp_storage['s'][-1]

        # Use Decreasing Step Size Scheme
        if scount >= s:
            scount = 0
            s += 1
        scount += 1
        Step = self.InitStep / s

        # Perform update
        NewData = self.Proj.P(_Data, Step, F)
        ext = (record.thisPermIndex - 1) / (record.thisPermIndex + 2)
        _NewData = NewData + ext * (NewData - Data)

        # Store Data
        temp_data = {}
        temp_data['Data'] = NewData
        temp_data['_Data'] = _NewData
        temp_data[self.F] = self.F(_NewData)
        temp_data['scount'] = scount
        temp_data['s'] = s
        temp_data['Step'] = Step
        temp_data['f Evaluations'] = 1 + self.temp_storage['f Evaluations'][-1]
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]
        self.book_keeping(temp_data)

        return self.temp_storage
