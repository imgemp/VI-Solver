import numpy as np

class Path:

    def __init__(self,Data0,Options):
        self.CurrData = 0
        self.MaxData = Options.Repo.MaxData
        self.CurrRec = 0
        self.MaxRecord = Options.Term.Tols[0]+1

        emptyData0 = np.array(Data0) # could use Data = np.empty((self.MaxDataData,)+Data0.shape); Data[:] = np.NaN;
        emptyData0[:] = np.NaN
        Data = np.array([emptyData0 for i in xrange(self.MaxData)])
        Data[0] = Data0
        self.Data = Data

        Steps = np.array([np.NaN for i in xrange(self.MaxRecord)])
        Steps[0] = Options.Init.Step
        self.Steps = Steps

        FEvals = np.array([np.NaN for i in xrange(self.MaxRecord)])
        FEvals[0] = 0
        self.FEvals = FEvals

        self.Report = {}
        for req in Options.Repo.Requests:
            report = req(Data0)
            if isinstance(report,np.ndarray): self.Report[req] = np.zeros((self.MaxRecord,)+report.Shape)
            else: self.Report[req] = np.zeros((self.MaxRecord,))
            self.Report[req][:] = np.NaN
            self.Report[req][0] = report

    def BookKeeping(self,NewData,NewStep,FEvals):

        # Update Data Record
        if self.CurrData < self.MaxData-1:
            self.CurrData += 1
            self.Data[self.CurrData] = NewData
        else:
            self.Data = np.concatenate((self.Data[1:],[NewData]))

        # Update Report Record
        self.CurrRec += 1
        self.Steps[self.CurrRec] = NewStep
        self.FEvals[self.CurrRec] = FEvals
        for req in self.Report:
            report = req(NewData)
            self.Report[req][self.CurrRec] = report

    def RemoveUnused(self):
        self.Data = self.Data[:min(self.CurrRec+1,self.MaxData)]
        self.Steps = self.Steps[:self.CurrRec+1]
        self.FEvals = self.FEvals[:self.CurrRec+1]
        for req in self.Report:
            self.Report[req] = self.Report[req][:self.CurrRec+1]



