import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Utilities import *
from VISolver.Solvers.Solver import Solver


class HeunEuler(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-2,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):
        self.F = Domain.f
        self.Proj = P
        self.StorageSize = 1
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
        Fs = np.zeros((2, Data.shape[0]))
        Fs[0, :] = record.TempStorage[self.F][-1]
        Step = record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Perform update
        _NewData = self.Proj.P(Data, Step, Fs[0, :])
        Fs[1, :] = self.F(_NewData)
        NewData = self.Proj.P(Data, Step, 0.5 * np.sum(Fs, axis=0))

        # Adjust Stepsize
        Delta = max(abs(NewData - _NewData))
        if Delta == 0.:
            Step *= 2.
        else:
            Step = max(min(Step * min((self.Delta0 / Delta) ** 0.5,
                                      self.GrowthLimit),
                           self.MaxStep),
                       self.MinStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['f Evaluations'] = 2 + self.temp_storage['f Evaluations'][-1]
        TempData['Projections'] = 2 + self.temp_storage['Projections'][-1]
        self.book_keeping(TempData)

        return self.temp_storage
