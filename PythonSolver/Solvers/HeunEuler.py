import numpy as np

from Projection import *
from Path import *
from Utilities import *
from Solver import Solver

class HeunEuler(Solver):

    def __init__(self,Function,P=IdentityProjection(),History=0,Delta0=1e-2,GrowthLimit=2,MinStep=-1e10,MaxStep=1e10):
        
        self.F = Function

        self.Proj = P

        self.H = History

        self.Delta0 = Delta0

        self.GrowthLimit = GrowthLimit

        self.MinStep = MinStep

        self.MaxStep = MaxStep

    def Update(self,Record,Domain,Step): #Could ask method to store a temp report as a property (Method.TempReport) and then pass Method to bookkeeping and just return Data and Step

        Data = Record.Data[Record.CurrData]
        Fs = np.zeros((2,Data.shape[0]))
        Fs[0,:] = Domain.F(Data)
        _NewData = self.Proj.P(Data,Step,Fs[0,:])
        Fs[1,:] = Domain.F(_NewData)
        NewData = self.Proj.P(Data,Step,0.5*np.sum(Fs,axis=0))

        FEvals = 2 #Should track number of projections here too

        Delta = max(abs(NewData-_NewData))
        if Delta == 0.: Step = 2.*Step
        else: Step = max(min(Step*min((self.Delta0/Delta)**0.5,self.GrowthLimit),self.MaxStep),self.MinStep)
        
        return NewData, Step, FEvals