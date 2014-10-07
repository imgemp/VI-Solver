import numpy as np

from Utilities import *

class LineSearch:

    def __init__(self):
        print('This is a line search object.  You need to pick a specific line search to use.')

    def LS(self,Domain,P,Metric,Data,NewData,Direc,Step):
        return None

class DangLan:

    def __init__(self,factor=0.8):
        self.factor = factor

    def LS(self,Domain,P,Metric,Data,NewData,Direc,Step):
        DomainF_Data = Domain.F(Data)
        # print('Alpha = '+str(Metric.alpha))
        # print('NewData = '+str(NewData))
        # print('Diff = '+str(np.sum(np.abs(Data-NewData))**2))
        # print('Step = '+str(Step))
        unacceptable = True
        # print('####################')
        while unacceptable:
            left = np.sum(np.abs(DomainF_Data-Domain.F(NewData)))**2  #using 1-norm right now
            right = Metric.alpha/(Step**2)*self.V(Metric,Data,NewData)
            if left <= right:
                # print('left = '+str(left))
                # print('right = '+str(right))
                # print('Step = '+str(Step))
                # print('####################')
                unacceptable = False
            else:
                # print('left = '+str(left))
                # print('right = '+str(right))
                # print('Step = '+str(Step))
                Step = Step*self.factor
                NewData = P(Data,Step,Direc)
        NewData = P(NewData,Step,Domain.F(NewData))
        return NewData

    def V(self,Metric,Data,NewData):
        return np.sum(Metric.w(NewData)-Metric.w(Data)-Metric.grad_w(Data)*(NewData-Data))