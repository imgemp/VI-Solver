import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class HeunEuler(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-2,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        F_Start = self.F(Start)
        F_Start_Norms = np.abs(F_Start)
        self.TempStorage['Norms'] = self.StorageSize*[F_Start_Norms]
        self.TempStorage[self.F] = self.StorageSize*[F_Start/F_Start_Norms]
        self.TempStorage['F_Norms'] = self.StorageSize*[F_Start_Norms]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def PhaseSpaceMultiplier(self,Tl,Tr):
        delta = 1e-15
        psi = .7
        Bmin = .01
        Bmax = .1
        alpha1 = 5

        if Tr > delta:
            r = Tl/Tr
        elif Tl <= delta:
            r = Bmax
        else:
            r = psi

        if r <= Bmin:
            mult = alpha1
        elif Bmin < r and r <= Bmax:
            mult = (alpha1*(Bmax-r)+(r-Bmin))/(Bmax-Bmin)
        elif Bmax < r and r <= psi:
            mult = ((psi-r)+.5*(r-Bmax))/(psi-Bmax)
        else:
            mult = .5

        return mult

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        Norm = Record.TempStorage['F_Norms'][-1]
        Fs_Data = np.zeros((2,Data.shape[0]))
        Fs_Data[0,:] = Record.TempStorage[self.F][-1]
        Fs_Norm = np.zeros((2,Data.shape[0]))
        Fs_Norm[0,:] = Record.TempStorage['F_Norms'][-1]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Perform Update
        direction_data = Fs_Data[0,:]
        direction_norm = Fs_Norm[0,:]
        _NewData = self.Proj.P(Data,Step,direction_data)
        _NewNorm = self.Proj.P(Norm,Step,direction_data)
        Fs[1,:] = self.F(_NewData)
        direction = 0.5*np.sum(Fs,axis=0)
        NewData = self.Proj.P(Data,Step,direction)

        # Compute Delta + Traditional Stepsize
        Delta = max(abs(NewData-_NewData))
        if Delta == 0.:
            growth_est = self.GrowthLimit
        else:
            growth_est = (self.Delta0/Delta)**0.5

        # Compute Tl & Tr + Phase Space Stepsize
        NewF = self.F(NewData)
        Tl = max(abs(direction-.5*(NewF+Fs[0])))
        Tr = .5*max(abs(NewF+Fs[0]))
        growth_ps = self.PhaseSpaceMultiplier(Tl,Tr)

        # Conservative Adjustment
        growth = min(growth_est,growth_ps)

        # Adjust Stepsize
        Step = np.clip(growth*Step,self.MinStep,self.MaxStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = NewF
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
