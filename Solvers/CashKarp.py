import numpy as np

from Projection import IdentityProjection
from Solver import Solver

class CashKarp(Solver):

    def __init__(self,Domain,P=IdentityProjection(),Delta0=1e-4,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.F = Domain.F

        self.Proj = P

        self.StorageSize = 1

        self.TempStorage = {}

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

        self.BT = np.array([[1.0/5.0,0.0,0.0,0.0,0.0,0.0],\
                            [3.0/40.0,9.0/40.0,0.0,0.0,0.0,0.0],\
                            [3.0/10.0,-9.0/10.0,6.0/5.0,0.0,0.0,0.0],\
                            [-11.0/54.0,5.0/2.0,-70.0/27.0,35.0/27.0,0.0,0.0],\
                            [1631.0/55296.0,175.0/512.0,575.0/13824.0,44275.0/110592.0,253.0/4096.0,0.0],\
                            [37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0],\
                            [2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25]\
                           ]);

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

        # Perform Update
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[0,0:1],Fs[0:1],axes=(0,0)))
        Fs[1,:] = self.F(_NewData)
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[1,0:2],Fs[0:2],axes=(0,0)))
        Fs[2,:] = self.F(_NewData)
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[2,0:3],Fs[0:3],axes=(0,0)))
        Fs[3,:] = self.F(_NewData)
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[3,0:4],Fs[0:4],axes=(0,0)))
        Fs[4,:] = self.F(_NewData)
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[4,0:5],Fs[0:5],axes=(0,0)))
        Fs[5,:] = self.F(_NewData)
        _NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[6,0:6],Fs[0:6],axes=(0,0)))
        NewData = self.Proj.P(Data,Step,np.tensordot(self.BT[5,0:6],Fs[0:6],axes=(0,0)))

        # Adjust Stepsize
        Delta = max(abs(NewData-_NewData))
        if Delta == 0.: Step = 2.*Step
        else: Step = max(min(Step*min((self.Delta0/Delta)**0.2,self.GrowthLimit),self.MaxStep),self.MinStep)

        # Store Data
        TempData['Data'] = NewData
        TempData[self.F] = self.F(NewData)
        TempData['Step'] = Step
        TempData['F Evaluations'] = 6 + self.TempStorage['F Evaluations'][-1]
        TempData['Projections'] = 6 + self.TempStorage['Projections'][-1]
        self.BookKeeping(TempData)
        
        return self.TempStorage






