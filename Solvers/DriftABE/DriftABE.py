from VISolver.Projection import *
from VISolver.Solvers import Solver
from VISolver.Utilities import *


class DriftABE(Solver):

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
        self.Mod = 6  # (100)
        self.Agg = 1  # (10)
        self.agent_i = 0

    def init_temp_storage(self, Start, Domain, Options):

        self.temp_storage['Data'] = self.StorageSize * [Start]
        self.temp_storage[self.F] = self.StorageSize * [self.F(Start)]
        self.temp_storage['Step'] = self.StorageSize * [Options.Init.Step]
        self.temp_storage['f Evaluations'] = self.StorageSize * [1]
        self.temp_storage['Projections'] = self.StorageSize * [0]
        self.temp_storage['F_2norm'] = self.StorageSize * \
            [np.dot(self.temp_storage[self.F][-1], self.temp_storage[self.F][-1])]

        return self.temp_storage

    # book_keeping(self,TempData) defined in super class 'Solver'

    def update(self, record):

        # Retrieve Necessary Data
        Data = record.TempStorage['Data'][-1]
        Step = record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # May want to use a Finite State Machine approach rather than if
        # statements

        if (record.thisPermIndex >= self.Mod) and (
                record.thisPermIndex % self.Mod == 0):

            # Choose Agent for Curl Component
            self.agent_i = np.random.randint(Data.shape[0])

            # Freeze Agent i's Policy
            F = record.TempStorage[self.F][-1]
            F[self.agent_i] = 0

            # Perform Euler update
            NewData = self.Proj.P(Data, Step, F)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        elif (record.thisPermIndex >= self.Mod) and (record.thisPermIndex % self.Mod == 1):

            # Approximate Gradient of ||f||^2 with respect to agent i
            G_k = self.temp_storage['F_2norm'][-1]
            G_km1 = self.temp_storage['F_2norm'][-2]
            G_km2 = self.temp_storage['F_2norm'][-3]
            x_km1 = self.temp_storage['Data'][-2][self.agent_i]
            x_km2 = self.temp_storage['Data'][-3][self.agent_i]
            dG_dxi = (-G_k + 2 * G_km1 - G_km2) / (x_km1 - x_km2)

            # Compute Adjusted f
            F = record.TempStorage[self.F][-1]
            F[self.agent_i] += - self.Agg * 0.5 * dG_dxi

            # Perform Euler update
            NewData = self.Proj.P(Data, Step, F)

            # Record Projections
            TempData['Projections'] = 1 + self.temp_storage['Projections'][-1]

        else:
            if record.thisPermIndex % 2 == 0:
                # Perform Euler update
                F = record.TempStorage[self.F][-1]
                NewData = self.Proj.P(Data, Step, F)

                # Record Projections
                TempData['Projections'] = 1 + \
                    self.temp_storage['Projections'][-1]

            else:
                # Perform Adams Bashforth update
                Fs = record.TempStorage[self.F]
                NewData = self.Proj.P(Data, Step, -0.5 * Fs[-2] + 1.5 * Fs[-1])

                # Perform Euler update
                _NewData = self.Proj.P(Data, Step, Fs[-1])

                # Adjust Stepsize
                Delta = max(abs(NewData - _NewData))
                if Delta == 0.:
                    Step *= 2.
                else:
                    Step = max(min(Step * min((self.Delta0 / Delta) ** 0.5,
                                              self.GrowthLimit),
                                   self.MaxStep),
                               self.MinStep)

                # Record Projections
                TempData['Projections'] = 2 + \
                    self.temp_storage['Projections'][-1]

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['f Evaluations'] = 1 + self.temp_storage['f Evaluations'][-1]
        TempData['F_2norm'] = np.dot(TempData[self.F], TempData[self.F])
        self.book_keeping(TempData)

        return self.temp_storage
