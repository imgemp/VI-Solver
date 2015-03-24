import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver


class GABE(Solver):

    def __init__(
            self,
            Domain,
            P=IdentityProjection(),
            Delta0=1e-2,
            GrowthLimit=2,
            MinStep=-1e10,
            MaxStep=1e10):

        self.F = Domain.f

        self.Proj = P

        self.StorageSize = 2

        self.temp_storage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def init_temp_storage(self, Start, Domain, Options):

        self.temp_storage['Data'] = self.StorageSize * [Start]
        self.temp_storage[self.F] = self.StorageSize * [self.F(Start)]
        self.temp_storage['Step'] = self.StorageSize * [Options.Init.Step]
        self.temp_storage['f Evaluations'] = self.StorageSize * [1]
        self.temp_storage['Projections'] = self.StorageSize * [0]

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    def update(self, record):

        # Retrieve Necessary Data
        Data = record.TempStorage['Data'][-1]
        _Data = record.TempStorage['Data'][-2]
        Step = record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        if record.thisPermIndex % 2 == 0:

            # Perform Euler update
            F = record.TempStorage[self.F][-1]
            NewData = self.Proj.P(Data, Step, F)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        else:

            # Perform Euler update
            Fs = record.TempStorage[self.F]
            _NewData = self.Proj.P(Data, Step, Fs[-1])

            # Perform Adams Bashforth update
            __NewData = self.Proj.P(Data, Step, -0.5 * Fs[-2] + 1.5 * Fs[-1])

            # Approximate Gradient of ||f||^2
            grad_norm = (
                np.linalg.norm(Fs[-1]) - np.linalg.norm(Fs[-2])) / (Data - _Data)
            # print(grad_norm)
            # print(Data-_Data)
            # grad_norm = 0.

            # Perform Adams Bashforth update again with Gradient Norm
            # divergence is zero for DummyMARL
            NewData = self.Proj.P(
                Data, Step, -0.5 * Fs[-2] + 1.5 * Fs[-1] - 0.5 * grad_norm)

            # Adjust Stepsize
            Delta = max(abs(__NewData - _NewData))
            if Delta == 0.:
                Step *= 2.
            else:
                Step = max(min(Step * min((self.Delta0 / Delta) ** 0.5,
                                          self.GrowthLimit),
                               self.MaxStep),
                           self.MinStep)

            # Record Projections
            TempData['Projections'] = 2 + self.temp_storage['Projections'][-1]

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['f Evaluations'] = 1 + self.temp_storage['f Evaluations'][-1]
        self.book_keeping(TempData)

        return self.temp_storage
