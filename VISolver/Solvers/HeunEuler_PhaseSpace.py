import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class HeunEuler_PhaseSpace(Solver):

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

        # self.DimWise = DimWise

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        # if self.DimWise:
        #     step = Options.Init.Step*np.ones_like(Start)
        # else:
        step = Options.Init.Step
        self.TempStorage['Step'] = self.StorageSize*[step]
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
        Fs = np.zeros((2,Data.shape[0]),dtype=Data.dtype)
        Fs[0,:] = Record.TempStorage[self.F][-1]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Perform Update
        direction = Fs[0,:]
        _NewData = self.Proj.P(Data,Step,direction)
        Fs[1,:] = self.F(_NewData)
        direction = 0.5*np.sum(Fs,axis=0)
        NewData = self.Proj.P(Data,Step,direction)

        # Compute Delta + Traditional Stepsize
        # if self.DimWise:
        #     Delta = abs(NewData-_NewData)
        # else:
        #     Delta = np.array([max(abs(NewData-_NewData))])
        Delta = max(abs(NewData-_NewData))
        # growth_est = self.GrowthLimit*np.ones_like(Delta)
        # print('check1')
        # growth_est[Delta>0.] = (self.Delta0/Delta[Delta>0.])**0.5
        # print('check2')
        if Delta == 0.:
            growth_est = self.GrowthLimit
        else:
            growth_est = (self.Delta0/Delta)**0.5  # safety factor, theta, is set to 1

        # http://www.math.mcgill.ca/humphries/index.html
        # P.59 of Dynamics of Adaptive Time-Stepping ODE Solvers
        # Sections 7 & 8, P. 15 of Phase Space Error Control for Dynamic Systems
        # Compute Tl & Tr + Phase Space Stepsize
        # Using L-infinty norm for Tl and Tr
        NewF = self.F(NewData)

        # if self.DimWise:
        #     Tl = abs(direction-.5*(NewF+Fs[0]))
        #     Tr = .5*abs(NewF+Fs[0])
        #     growth_ps = []
        #     for tl,tr in zip(Tl,Tr):
        #         growth_ps += [self.PhaseSpaceMultiplier(tl,tr)]
        #     growth_ps = np.array(growth_ps)
        #     growth = np.min(np.vstack((growth_est,growth_ps)),axis=0)
        # else:
            # growth_est = growth_est[0]
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
