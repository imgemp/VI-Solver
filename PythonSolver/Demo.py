# Solver Modules
from Options import *
from Domains import *
from Metrics import *
from Descent import *
from ButcherTableaus import *
import time
import numpy as np

# Plotting Modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Datetime Module - Output File Names
import datetime

def run_demo_DangLan_LS():

	#Set Options
    Init = Initialization(Step=-0.8)
    Term = Termination(Tols={'Iter':100,'GenError':1e-3})
    Repo = Reporting(MaxData=1,Requests=['GenError'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    #Set Method
    Table = 'Euler'
    BT = ButcherTableau(Table)
    Function = 'Gradient'
    AS = None
    P = EntropicProjection()
    LS = DangLan()
    Metric = Entropy(alpha=10.0)

    #Write Options to Output
    params = 'Table: '+str(Table)+', Parameters: '
    for tol in Term.Tols.keys():
        params += tol+': '+str(Term.Tols[tol])+', '
    print(params[:-2])

    ##################################################
    print('Kojima-Shindo')

    #Set Delta0
    Delta0 = 1e-1
    Method = DescentMethod(BT,Function=Function,AS=AS,Delta0=Delta0,P=P,LS=LS,Metric=Metric)
    print('Method: '+'F = '+Function+', P = '+P.Name())

    #Define Dimension and Domain
    Domain = KojimaShindo()

    #Initialize Starting Point
    Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

    tic = time.time()
    KS_Results = Descend(Start,Method,Domain,Options)
    toc = time.time() - tic
    print('Steps, CPU Time, Error, Min |X_i|, Max |X_i| = '+str(KS_Results.FEvals.shape[0]-1)+','+str(toc)+','+str(KS_Results.Report['GenError'][-1])+\
        ','+str(max(abs(KS_Results.Data[-1])))+','+str(min(abs(KS_Results.Data[-1]))))
    print('Num Projections = '+str(Method.Proj.NP))
    Method.Proj.NP = 0

    ##################################################
    # print('Watson')

    # #Set Delta0
    # Delta0 = 1e-1
    # Method = DescentMethod(BT,Function=Function,AS=AS,Delta0=Delta0,P=P,LS=LS,Metric=Metric)
    # print('Method: '+'F = '+Function+', P = '+P.Name())

    # WAT_Results = [[] for i in xrange(10)]

    # for p in xrange(10):

    #     #Define Dimension and Domain
    #     Domain = Watson(Pos=p)

    #     #Initialize Starting Point
    #     Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

    #     tic = time.time()
    #     WAT_Results[p] = Descend(Start,Method,Domain,Options)
    #     toc = time.time() - tic
    #     print('Steps, CPU Time, Error, Min |X_i|, Max |X_i| ('+str(p+1)+') = '+str(WAT_Results[p].FEvals.shape[0]-1)+','+str(toc)+','+str(WAT_Results[p].Report['GenError'][-1])+\
    #     ','+str(max(abs(WAT_Results[p].Data[-1])))+','+str(min(abs(WAT_Results[p].Data[-1]))))
    #     print('Num Projections = '+str(Method.Projections[P][1]))
    #     Method.Projections[P][1] = 0

    ##################################################
    # print('Sun')

    # #Set Delta0
    # Delta0 = 1e-1
    # Method = DescentMethod(BT,Function=Function,AS=AS,Delta0=Delta0,P=P,LS=LS,Metric=Metric)
    # print('Method: '+'F = '+Function+', P = '+P.Name())

    # Sun_Results = [[] for i in xrange(8000,30000+1,2000)]

    # for n in xrange(8000,30000+1,2000):

    #     #Define Dimension and Domain
    #     Domain = Sun(Dim=n)

    #     #Initialize Starting Point
    #     Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

    #     tic = time.time()
    #     ind = n/2000-4
    #     Sun_Results[ind] = Descend(Start,Method,Domain,Options)
    #     toc = time.time() - tic
    #     print('Steps, CPU Time, Error, Min |X_i|, Max |X_i| ('+str(n)+') = '+str(Sun_Results[ind].FEvals.shape[0]-1)+','+str(toc)+','+str(Sun_Results[ind].Report['GenError'][-1])+\
    #     ','+str(max(abs(Sun_Results[ind].Data[-1])))+','+str(min(abs(Sun_Results[ind].Data[-1]))))
    #     print('Num Projections = '+str(Method.Projections[P][1]))
    #     Method.Projections[P][1] = 0

if __name__ == '__main__':
  run_demo_DangLan_LS()






