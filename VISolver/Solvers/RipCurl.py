from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solver
from VISolver.Utilities import RandUnit
import numpy as np
from IPython import embed

class RipCurl(Solver):

    def __init__(self,Domain,P=IdentityProjection(),eps=1e-8):

        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.eps = eps

        self.factor = 1.

    def InitTempStorage(self,Start,Domain,Options):

        self.TempStorage['Data'] = self.StorageSize*[Start]
        # v = RandUnit(Start)*self.eps
        # self.TempStorage['v'] = self.StorageSize*[v]
        self.TempStorage[self.F] = self.StorageSize*[self.F(Start)] #+v)]
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
        # v = Record.TempStorage['v'][-1]
        scount = self.TempStorage['scount'][-1]
        s = self.TempStorage['s'][-1]

        # J^TF  # should just use finite differences on the norm of F
        v = RandUnit(Data)*self.eps
        Fv = self.F(Data+v)
        f = 0.5*np.linalg.norm(Fv) - 0.5*np.linalg.norm(F)
        JTF = F.size/self.eps*f*v

        # Use Decreasing Step Size Scheme
        if scount >= s:
            scount = 0
            s += 1
        scount += 1
        Step = self.InitStep/s

        # Initialize Storage
        TempData = {}

        # Perform Update
        _NewData = self.Proj.P(Data,self.factor*Step,F)
        JF = (F-self.F(_NewData))/abs(Step)
        # modF = self.F(_NewData) + abs(self.factor*Step)*JTF  # (I - Step*J)*F(xk) + Step*J^T*F(xk)
        modF = F + (JF-JTF)
        NewData = self.Proj.P(Data,Step,modF)
        # embed()
        # assert False
        # Store Data
        TempData['Data'] = NewData
        # v = RandUnit(NewData)/self.eps
        # TempData['v'] = v
        TempData[self.F] = self.F(NewData) #+v)
        TempData['scount'] = scount
        TempData['s'] = s
        TempData['Step'] = Step
        TempData['F Evaluations'] = 2 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 2 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)

        return self.TempStorage
