import numpy as np

class Initialization:

    def __init__(self,Step=0):
        self.Step = Step

class Termination:

    def __init__(self,MaxIter=1,Tols=[]):
        self.Tols = [MaxIter]
        self.Tols.append(Tols)

    def CheckTols(self,Requests):
        for tol in self.Tols[1:][0]:
            if not (tol[0] in Requests):
                self.Tols.remove(tol)
                print(`tol[0].func_name`+' cannot be used as a terminal condition because it is not tracked during the descent.')
        return self.Tols
    # IsTerminal should really just check the temp report not the long term report
    def IsTerminal(self,Record):
        if (Record.CurrRec>=self.Tols[0]):
            return True
        for tol in self.Tols[1:][0]:
            if (Record.Report[tol[0]][Record.CurrRec]<=tol[1]):
                    return True
            
class Reporting:

    def __init__(self,MaxData=1,Requests=[]): # need short term requests and long term requests
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
        if not self.Misc.Min == None: Domain.Min = self.Misc.Min
        self.Term.Tols = self.Term.CheckTols(self.Repo.Requests)




