import numpy as np

from Projections import *
from LineSearch import *
from Path import *
from Utilities import *

class DescentMethod(Projection,LineSearch):

    def __init__(self,BTableau,Function=None,AS='Traditional',Delta0=None,P=IdentityProjection(),LS=None,Metric=None):
        self.Function = Function

        self.Proj = P

        self.LineSearch = LS

        self.Metric = Metric
        #add checks for valid metric / if metric needed

        if AS == None:
            self.Table = BTableau.Table[0:BTableau.Table.shape[1],:]
            self.TableRK = None
            self.Order = BTableau.Order
            self.Mem = BTableau.Mem
            self.AdaptStep = None
            self.Delta0 = None
        elif AS in BTableau.EAS:
            if AS == 'Traditional':
                self.Table = BTableau.Table[0:BTableau.Table.shape[1],:]
                self.TableRK = None
                self.Order = BTableau.Order
                self.Mem = BTableau.Mem
                self.AdaptStep = self.AdaptStepTrad
                self.Delta0 = None
            elif (AS == 'R-K Local Extrapolation') or (AS == 'Runge-Kutta'):
                if Delta0 == None:
                    print('You must specify a desired accuracy for the adaptive step size.')
                else:
                    if (AS == 'Runge-Kutta'):
                        if BTableau.Table.shape[0] > BTableau.Table.shape[1] + 1:
                            self.Table = BTableau.Table[BTableau.Table.shape[1]:,:]
                            self.TableRK = BTableau.Table[0:BTableau.Table.shape[1],:]
                        else:
                            self.Table = np.concatenate((BTableau.Table[0:BTableau.Table.shape[1]-1,:],BTableau.Table[-1,:][None]))
                            self.TableRK = BTableau.Table[BTableau.Table.shape[1]-1,:][None]
                    else:
                        self.Table = BTableau.Table[0:BTableau.Table.shape[1],:]
                        self.TableRK = BTableau.Table[BTableau.Table.shape[1]:,:]
                    self.Order = BTableau.Order
                    self.Mem = BTableau.Mem
                    self.Delta0 = Delta0
                    self.AdaptStep = self.AdaptStepRK
        else:
            print(str(AS)+'is not an elligible adaptive step size option.')

    def Update(self,Record,Domain,Step):
        #assuming mem is 0 for now
        Data = Record.Data[Record.CurrData-self.Mem:Record.CurrData+1][-1]

        Fs = np.zeros((self.Table.shape[1],Data[None].shape[1]))
        Fs[0,:] = Domain.F(Data)

        #add in a while loop functionality
        for k in xrange(1,Fs.shape[0]):
            Direc = np.dot(self.Table[k-1,0:k],Fs[0:k,:])
            Fs[k,:] = Domain.F(self.Proj.P(Data,Step,Direc))

        Direc = np.dot(self.Table[-1,:],Fs)
        NewData = self.Proj.P(Data,Step,Direc)

        if not self.LineSearch == None: NewData = self.LineSearch.LS(Domain,self.Proj.P,self.Metric,Data,NewData,Direc,Step)

        FEvals = self.Table.shape[0]-1

        if not self.AdaptStep == None: 
            Step = self.AdaptStep(Domain,Record.CurrRec+1,NewData,Data,Step,Fs)
            if not self.TableRK == None:
                FEvals += self.TableRK.shape[0]-1
        
        return NewData, Step, FEvals

    def AdaptStepTrad(self,Domain,CurrRec,NewData1=None,Data=None,Step=None,Fs=None):
        Step = -np.sqrt((2.0*np.log(Domain.Dim))/((Domain.L**2.0)*(float(CurrRec)+1.0)))
        return Step

    def AdaptStepRK(self,Domain,CurrRec,NewData1,Data,Step,Fs):
        if self.TableRK.shape[0] == 1:
            Direc = np.dot(self.TableRK[0,:],Fs)
            NewData2 = self.Proj.P(Data,Step,Direc)
        else:
            Fs = np.zeros((self.TableRK.shape[1],Data[None].shape[1]))
            Fs[0,:] = Domain.F(Data)

            for k in xrange(1,Fs.shape[0]):
                Direc = np.dot(self.TableRK[k-1,0:k],Fs[0:k,:])
                Fs[k,:] = Domain.F(self.Proj.P(Data,Step,Direc))

            Direc = np.dot(self.TableRK[-1,:],Fs)
            NewData2 = self.Proj.P(Data,Step,Direc)

        Delta = max(abs(NewData1-NewData2))
        #formalize this step bound of -1.0e10 - create a variable in options or something
        if Delta == 0.0:
            Step = -1.0e10 #should keep the current step size the same or double? double/use same bound in else statement
        else:
            Step = max(Step*((self.Delta0/Delta)**(1/self.Order)),-1.0e10)
        #Step = Step*((self.Delta0/Delta)**(1/self.Order))
        return Step

#Descent Controller
def Descend(Start,Method,Domain,Options):

    #Record Data Dimension
    Domain.Dim = Start.size

    #Check Validity of Options
    Options.CheckOptions(Method,Domain)
    Step = Options.Init.Step

    #Calculate Initial Additional Stats
    Report = Domain.Report(Start,Options.Repo.Requests)

    #Create Path Object for Record Keeping
    Record = Path(Start,Options,Report)

    #Begin Descent
    while not Options.Term.IsTerminal(Record):

        #Compute New Data Using Update Method
        Data, Step, FEvals = Method.Update(Record,Domain,Step)

        #Compute Additional Stats
        Report = Domain.Report(Data,Options.Repo.Requests)

        #Record Path Stats
        Record.BookKeeping(Data,Step,FEvals,Report)

    #Remove Unused Entries
    Record.RemoveUnused()

    return Record