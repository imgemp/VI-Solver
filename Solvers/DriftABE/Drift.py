import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver


class Drift(Solver):

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

        self.StorageSize = 3

        self.temp_storage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.Mod = 1000

        self.Agg = 10

    def init_temp_storage(self, Start, Domain, Options):

        self.temp_storage['Policy'] = self.StorageSize * [Start]
        self.temp_storage[self.F] = self.StorageSize * [self.F(Start)]
        self.temp_storage['Step'] = self.StorageSize * [Options.init.step]
        self.temp_storage['f Evaluations'] = self.StorageSize * [1]
        self.temp_storage['Projections'] = self.StorageSize * [0]
        self.temp_storage['F_2norm'] = self.StorageSize * \
            [np.dot(self.temp_storage[self.F][-1], self.temp_storage[self.F][-1])]

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    def update(self, record):

        # Retrieve Necessary Data
        Data = record.temp_storage['Policy'][-1]
        Step = record.temp_storage['Step'][-1]

        # Initialize Storage
        TempData = {}

        if (record.this_perm_index >= self.Mod) and (
                record.this_perm_index % self.Mod == 0):

            # Freeze Agent i's Policy
            agent_i = np.random.randint(Data.shape[0])
            F = record.temp_storage[self.F][-1]
            F[agent_i] = 0

            # Perform Euler update
            NewData = self.Proj.P(Data, Step, F)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        elif (record.this_perm_index >= self.Mod) and (record.this_perm_index % self.Mod == 1):

            # Approximate Gradient of ||f||^2
            G_k = self.temp_storage['F_2norm'][-1]
            G_km1 = self.temp_storage['F_2norm'][-2]
            G_km2 = self.temp_storage['F_2norm'][-3]
            x_km1 = self.temp_storage['Policy'][-2]
            x_km2 = self.temp_storage['Policy'][-3]
            dG_dxi = (G_k - 2 * G_km1 + G_km2) / (x_km1 - x_km2)

            # Perform Euler update
            F = record.temp_storage[self.F][-1]
            NewData = self.Proj.P(Data, Step, F - self.Agg * 0.5 * dG_dxi)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        else:

            # Perform Euler update
            F = record.temp_storage[self.F][-1]
            NewData = self.Proj.p(Data, Step, F)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        TempData['Policy'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['f Evaluations'] = 1 + self.temp_storage['f Evaluations'][-1]
        TempData['F_2norm'] = np.dot(TempData[self.F], TempData[self.F])
        self.book_keeping(TempData)

        return self.temp_storage
