import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class HeunEuler(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-2,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        self.Proj = P

        self.Proj_Norm = IdentityProjection()

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['F_Norms'] = self.StorageSize*[np.zeros_like(Start)]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]
        self.TempStorage['Time'] = abs(Options.Init.Step)

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
        Fs_Norm[0,:] = np.abs(Fs_Data[0,:])
        Step = Record.TempStorage['Step'][-1]
        Time = Record.TempStorage['Time'][-1]

        # Initialize Storage
        TempData = {}

        # Perform Update
        direction_data = Fs_Data[0,:]
        direction_norm = Fs_Norm[0,:]
        _NewData = self.Proj.P(Data,Step,direction_data)
        _NewNorm = self.Proj_Norm.P(Norm,Step,direction_norm)
        F_NewData = self.F(_NewData)
        Fs_Data[1,:] = F_NewData/_NewNorm
        Fs_Norm[1,:] = (np.abs(F_NewData) - _NewNorm)/(Time + Step)
        direction_data = 0.5*np.sum(Fs_Data,axis=0)
        direction_norm = 0.5*np.sum(Fs_Norm,axis=0)
        NewData = self.Proj.P(Data,Step,direction_data)
        NewNorm = self.Proj_Norm.P(Norm,Step,direction_norm)

        # Compute Delta + Traditional Stepsize
        Delta_data = max(abs(NewData-_NewData))
        Delta_norm = max(abs(NewNorm-_NewNorm))
        Delta = max(Delta_data,Delta_norm)
        if Delta == 0.:
            growth_est = self.GrowthLimit
        else:
            growth_est = (self.Delta0/Delta)**0.5

        # Compute Tl & Tr + Phase Space Stepsize
        NewF_data = self.F(NewData)
        NewF_norm = (np.abs(NewF_data) - Norm)/(Time + Step)
        Tl_data = abs(direction_data-.5*(NewF_data+Fs_Data[0]))
        Tl_norm = abs(direction_norm-.5*(NewF_norm+Fs_Norm[0]))
        Tl = max(Tl_data,Tl_norm)
        Tr_data = abs(NewF_data+Fs_Data[0])
        Tr_norm = abs(NewF_norm+Fs_Norm[0])
        Tr = .5*max(Tr_data,Tr_norm)
        growth_ps = self.PhaseSpaceMultiplier(Tl,Tr)

        # Conservative Adjustment
        growth = min(growth_est,growth_ps)

        # Adjust Stepsize
        Step = np.clip(growth*Step,self.MinStep,self.MaxStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = NewF_data
        TempData['F_Norms'] = NewNorm
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
