# Solver Modules
from Options import *
from Domains import *
# from Metrics import *
# from Descent import *
# from ButcherTableaus import *
from Solver import *
from HeunEuler import *
import time
import numpy as np

# Plotting Modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Datetime Module - Output File Names
import datetime

def runTest():

	#Set Options
    Init = Initialization(Step=-0.8)
    Term = Termination(Tols={'Iter':1000,'f_Error':1e-3})
    Repo = Reporting(MaxData=1,Requests=['f_Error'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    #Set Method
    Function = 'F'
    P = IdentityProjection()
    Method = HeunEuler(Function=Function,P=P,History=0,Delta0=1e-3)

    #Write Options to Output
    params = 'Parameters: '
    for tol in Term.Tols.keys():
        params += tol+': '+str(Term.Tols[tol])+', '
    print(params[:-2])

    ##################################################
    print('Domain: Sphere')

    #Set Delta0
    Delta0 = 1e-1
    print('Method: '+'F = '+Function+', P = '+P.Name())

    #Define Dimension and Domain
    Domain = Sphere(Dim=2);

    #Initialize Starting Point
    Start = 10*np.ones(Domain.Dim)

    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic
    print('Steps, CPU Time, Error, Min |X_i|, Max |X_i| = '+str(Results.FEvals.shape[0]-1)+','+str(toc)+','+str(Results.Report['Value'][-1])+\
        ','+str(max(abs(Results.Data[-1])))+','+str(min(abs(Results.Data[-1]))))
    print('Num Projections = '+str(Method.Proj.NP))
    Method.Proj.NP = 0

if __name__ == '__main__':
  runTest()