import time
import datetime
import numpy as np

from Domains.BloodBank import *

from Solvers.Euler import *
from Solvers.Extragradient import *
from Solvers.AcceleratedGradient import *
from Solvers.HeunEuler import *
from Solvers.AdamsBashforthEuler import *
from Solvers.CashKarp import *

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
    Network = CreateRandomNetwork(nC=2,nB=2,nD=2,nR=3,seed=0)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

	# Set Options
    # Init = Initialization(Step=-1e-10)
    Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations','Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    BloodBank_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results,Method,toc)

    # Zero Projections for Later Use
    Method.Proj.NP = 0

if __name__ == '__main__':
  Demo()







