import time
import numpy as np

from Domains.Rosenbrock import Rosenbrock

# from Solvers.Euler import Euler
# from Solvers.Extragradient import EG
# from Solvers.AcceleratedGradient import AG
# from Solvers.HeunEuler import HeunEuler
# from Solvers.AdamsBashforthEuler import ABEuler
from Solvers.CashKarp import CashKarp

from Projection import RPlusProjection
from Solver import Solve
from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from Log import PrintSimResults, PrintSimStats


def Demo():

    #__ROSENBROCK__##################################################

    # Define Domain
    Domain = Rosenbrock(Dim=1000)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = -0.5*np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=20000,Tols=[(Domain.f_Error,1e-6)])
    Repo = Reporting(Requests=[Domain.f_Error, 'Step', 'F Evaluations',
                               'Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Rosenbrock_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Rosenbrock_Results,Method,toc)

    # Zero Projections for Later Use
    Method.Proj.NP = 0

if __name__ == '__main__':
  Demo()