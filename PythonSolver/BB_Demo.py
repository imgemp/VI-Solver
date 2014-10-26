import time
import datetime
import numpy as np

from Domains.BloodBank import *

from Solvers.HeunEuler import *

from Solver import Solve
from Options import *
from Log import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def Demo():

    #__BLOOD_BANK__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(nC=2,nB=2,nD=2,nR=2,seed=0)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method # just pass a domain, a projection, and any method specific options (e.g. Delta0)
    Method = HeunEuler(Function=Domain.F,P=RPlusProjection(),History=0,Delta0=1e-5) #don't need history, method should know this itself, maybe just pass Domain instead of Domain.F

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

	# Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=10000,Tols=[[Domain.gap_rplus,1e-6*gap_0]])
    Repo = Reporting(MaxData=1,Requests=[Domain.gap_rplus]) #Could use Method.TempReport.[property] for short term report, should get rid of max data
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    BloodBank_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(BloodBank_Results,Method,toc)

    # Zero Projections for Later Use
    Method.Proj.NP = 0

if __name__ == '__main__':
  Demo()







