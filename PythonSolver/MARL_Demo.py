import time
import datetime
import numpy as np

from Domains.DummyMARL import *

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

    #__ROSENBROCK__##################################################

    # Define Domain
    Domain = MARL()

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    Method = HeunEuler(Domain=Domain,P=IdentityProjection(),Delta0=1e-6)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = np.array([0,1])

	# Set Options
    Init = Initialization(Step=1e-2)
    # Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=2000,Tols=[(Domain.Origin_Error,1e-4)])
    Repo = Reporting(Requests=[Domain.Origin_Error,'Step','F Evaluations','Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    MARL_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,MARL_Results,Method,toc)

    # Zero Projections for Later Use
    Method.Proj.NP = 0

    # print(MARL_Results.PermStorage['Data'])

if __name__ == '__main__':
  Demo()







