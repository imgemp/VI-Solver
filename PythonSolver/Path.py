import numpy as np

#Path Stats
class Path:

    def __init__(self,Data0,Options,Report):
        self.CurrData = 0
        self.MaxData = Options.Repo.MaxData
        self.CurrRec = 0
        self.MaxRecord = Options.Term.Tols['Iter']+1

        emptyData0 = np.array(Data0)
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

        self.Report = dict(Report)
        for item in self.Report:
            if isinstance(Report[item],np.ndarray):
                emptyItem = np.array(Report[item])
                emptyItem[:] = np.NaN
            else:
                emptyItem = np.NaN
            self.Report[item] = np.array([emptyItem for i in xrange(self.MaxRecord)])
            self.Report[item][0] = Report[item]

    def BookKeeping(self,NewData,NewStep,FEvals,Report):
        if self.CurrData < self.MaxData-1:
            self.CurrData += 1
            self.Data[self.CurrData] = NewData
        else:
            self.Data = np.concatenate((self.Data[1:],[NewData]))
        self.CurrRec += 1
        self.Steps[self.CurrRec] = NewStep
        self.FEvals[self.CurrRec] = FEvals
        for item in self.Report:
            self.Report[item][self.CurrRec] = Report[item]

    def RemoveUnused(self):
        self.Data = self.Data[:min(self.CurrRec+1,self.MaxData)]
        self.Steps = self.Steps[:self.CurrRec+1]
        self.FEvals = self.FEvals[:self.CurrRec+1]
        for item in self.Report:
            self.Report[item] = self.Report[item][:self.CurrRec+1]