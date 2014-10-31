import numpy as np

from Projection import *
from Utilities import *
from Solver import Solver

class Drift(Solver):

    def __init__(self,Domain,P=IdentityProjection(),Delta0=1e-2,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 3

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.Mod = 1000

        self.Agg = 10

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]
        self.TempStorage['F_2norm'] = self.StorageSize*[np.dot(self.TempStorage[self.F][-1],self.TempStorage[self.F][-1])]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        if (Record.thisPermIndex>=self.Mod) and (Record.thisPermIndex%self.Mod == 0):

            # Freeze Agent i's Policy
            agent_i = np.random.randint(Data.shape[0])
            F = Record.TempStorage[self.F][-1]
            F[agent_i] = 0

            # Perform Euler Update
            NewData = self.Proj.P(Data,Step,F)

            # Record Projections
            TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        elif (Record.thisPermIndex>=self.Mod) and (Record.thisPermIndex%self.Mod == 1):

            # Approximate Gradient of ||F||^2
            G_k = self.TempStorage['F_2norm'][-1]
            G_km1 = self.TempStorage['F_2norm'][-2]
            G_km2 = self.TempStorage['F_2norm'][-3]
            x_km1 = self.TempStorage['Data'][-2]
            x_km2 = self.TempStorage['Data'][-3]
            dG_dxi = (G_k-2*G_km1+G_km2)/(x_km1-x_km2)

            # Perform Euler Update
            F = Record.TempStorage[self.F][-1]
            NewData = self.Proj.P(Data,Step,F-self.Agg*0.5*dG_dxi)

            # Record Projections
            TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        else:

            # Perform Euler Update
            F = Record.TempStorage[self.F][-1]
            NewData = self.Proj.P(Data,Step,F)

            # Record Projections
            TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['F Evaluations'] = 1 + self.TempStorage['F Evaluations'][-1]
        TempData['F_2norm'] = np.dot(TempData[self.F],TempData[self.F])
        self.BookKeeping(TempData)
        
        return self.TempStorage





