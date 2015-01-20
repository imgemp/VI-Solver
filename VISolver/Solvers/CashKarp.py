import numpy as np

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver


class CashKarp(Solver):

    def __init__(self, Domain, P=IdentityProjection(), Delta0=1e-4,
                 GrowthLimit=2, MinStep=-1e10, MaxStep=1e10):

        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.BT = np.array([
            [1./5.,0.,0.,0.,0.,0.],
            [3./40.,9./40.,0.,0.,0.,0.],
            [3./10.,-9./10.,6./5.,0.,0.,0.],
            [-11./54.,5./2.,-70./27.,35./27.,0.,0.],
            [1631./55296.,175./512.,575./13824.,44275./110592.,253./4096.,0.],
            [37./378.,0.,250./621.,125./594.,0.,512./1771.],
            [2825./27648.,0.,18575./48384.,13525./55296.,277./14336.,0.25]
        ])

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        Fs = np.zeros((6,Data.shape[0]))
        Fs[0,:] = Record.TempStorage[self.F][-1]
        Step = Record.TempStorage['Step'][-1]

        # Initialize Storage
        TempData = {}

        # Calculate k values (gradients)
        for i in xrange(5):
            direction = np.einsum('i,i...', self.BT[i,:i+1], Fs[:i+1])
            _NewData = self.Proj.P(Data, Step, direction)
            Fs[i+1,:] = self.F(_NewData)

        # Compute order-p, order-p+1 data points
        direction = np.einsum('i,i...', self.BT[6,:6], Fs[:6])
        _NewData = self.Proj.P(Data, Step, direction)
        direction = np.einsum('i,i...', self.BT[5,:6], Fs[:6])
        NewData = self.Proj.P(Data, Step, direction)

        # Adjust Stepsize
        Delta = max(abs(NewData-_NewData))
        if Delta == 0.:
            Step = 2. * Step
        else:
            growth = min(self.GrowthLimit, (self.Delta0/Delta)**0.2)
            Step = max(min(Step*growth, self.MaxStep), self.MinStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['F Evaluations'] = 6 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 6 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
