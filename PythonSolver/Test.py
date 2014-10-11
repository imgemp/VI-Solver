import time
import datetime
import numpy as np

from Options import *
from Domains import *
from Solver import *
from HeunEuler import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def runTest():

    #Define Dimension and Domain
    Domain = Sphere(Dim=100)

    #Set Method
    Method = HeunEuler(Function=Domain.F,P=IdentityProjection(),History=0,Delta0=1e-2)

	#Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=1000,Tols=[[Domain.f_Error,1e-3]])
    Repo = Reporting(MaxData=1,Requests=[Domain.f_Error])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    #Write Options to Output
    params = 'Parameters: MaxIter: '+`Term.Tols[0]`+', '
    for tol in Term.Tols[1:][0]:
        params += `tol[0].func_name`+': '+`tol[1]`+', '
    print(params[:-2])

    ##################################################
    print('Domain: '+`Domain.__class__.__name__`)
    print('Method: '+'Function = '+`Method.F.func_name`+', Projection = '+`Method.Proj.P.func_name`)

    #Initialize Starting Point
    Start = 100*np.ones(Domain.Dim)

    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic
    print('Steps, CPU Time, Error, Min |X_i|, Max |X_i| = '+str(Results.FEvals.shape[0]-1)+','+str(toc)+','+str(Results.Report[Domain.f_Error][-1])+\
        ','+str(max(abs(Results.Data[-1])))+','+str(min(abs(Results.Data[-1]))))
    print('Num Projections = '+str(Method.Proj.NP))
    Method.Proj.NP = 0

if __name__ == '__main__':
  runTest()






