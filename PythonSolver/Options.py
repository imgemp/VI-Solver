import numpy as np

#Descent Options
class Initialization:

    def __init__(self,Step=None):
        self.Step = Step

class Termination:

    def __init__(self,MaxIters=1,Tols=[]):
        self.Tols = np.append([MaxIters],Tols);

    def CheckTols(self,Requests):
        for tol in self.Tols[1:]:
            if not (tol in Requests):
                self.Tols = self.Tols.pop(tol,None)
                print(tol+' cannot be used as a terminal condition because it is not tracked during the descent.')
        return self.Tols

    def IsTerminal(self,Record):
        for tol in self.Tols.keys():
            if tol == 'Iter':
                if (Record.CurrRec>=self.Tols['Iter']):
                    return True
            elif (tol in Record.Report):
                if (Record.Report[tol][Record.CurrRec]<=self.Tols[tol]):
                    return True
            
class Reporting:

    def __init__(self,MaxData=1,Requests=[]):
        self.MaxData = MaxData
        self.Requests = Requests

class Miscellaneous:

    def __init__(self,Min=None):
        self.Min = Min

class DescentOptions:

    def __init__(self,Init=Initialization(),Term=Termination(),Repo=Reporting(),Misc=Miscellaneous()):
        self.Init = Init
        self.Term = Term
        self.Repo = Repo
        self.Misc = Misc

    def CheckOptions(self,Method,Domain):
        if self.Init.Step == None: self.Init.Step = Method.AdaptStepTrad(Domain,0)
        if not self.Misc.Min == None: Domain.Min = self.Misc.Min
        self.Repo.Requests = Domain.CheckRequests(self.Repo.Requests)
        self.Term.Tols = self.Term.CheckTols(self.Repo.Requests)
        if (not Method.Function == None) and (Method.Function in Domain.Fun):
            Domain.F = Domain.Fun[Method.Function]