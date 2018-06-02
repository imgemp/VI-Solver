from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver
import numpy as np
from IPython import embed

class RipCurlEx(Solver):

    def __init__(self,Domain,P=IdentityProjection(),FixStep=True,factor=0.5):

        self.F = Domain.F

        self.Jac = Domain.J

        self.Proj = P

        self.FixStep = FixStep

        self.StorageSize = 1

        self.TempStorage = {}

        self.factor = factor

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)]
        self.TempStorage['scount'] = self.StorageSize*[0]
        self.TempStorage['s'] = self.StorageSize*[1]
        self.TempStorage['Step'] = self.StorageSize*[Options.Init.Step]
        self.TempStorage['F Evaluations'] = self.StorageSize*[1]
        self.TempStorage['Projections'] = self.StorageSize*[0]

        self.InitStep = Options.Init.Step

        return self.TempStorage

    # BookKeeping(self,TempData) defined in super class 'Solver'

    def Update(self,Record):

        # Retrieve Necessary Data
        Data = Record.TempStorage['Data'][-1]
        F = Record.TempStorage[self.F][-1]
        scount = self.TempStorage['scount'][-1]
        s = self.TempStorage['s'][-1]

        if self.FixStep:
            Step = self.InitStep
        else:  # Use Decreasing Step Size Scheme
            if scount >= s:
                scount = 0
                s += 1
            scount += 1
            Step = self.InitStep/s

        # Initialize Storage
        TempData = {}

        # Modify F
        J = self.Jac(Data)
        # if np.all(np.linalg.eigvals(J+J.T)>=-1e-8):
            # print('PSD!')
        F_perp = np.dot(J-J.T,F)
        modF = F - self.factor*F_perp
        # embed()
        # assert False

        # Perform Update
        NewData = self.Proj.P(Data,Step,modF)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['F Evaluations'] = 1 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 1 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)
        return self.TempStorage
