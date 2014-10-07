import numpy as np

from Utilities import *

class AdaptiveStep:

    def __init__(self):
        print('This is a generic adaptive step size object.  You need to pick a specific adaptive step size operator to use.')

class AdaptiveStep_Traditional:

    def __init__(self):
        print('Check properties are satisfied (i.e. Domain.L, etc)')

    def AS(self,Domain,CurrRec,NewData1=None,Data=None,Step=None,Fs=None):
        Step = -np.sqrt((2.0*np.log(Domain.Dim))/((Domain.L**2.0)*(float(CurrRec)+1.0)))
        return Step

class AdaptiveStep_RK:

    def __init__(self):
        print('Check properties are satisfied (i.e. BTableau, etc)')

    def AS(self,Domain,CurrRec,NewData1,Data,Step,Fs):
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