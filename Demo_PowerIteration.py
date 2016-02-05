import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.PowerIteration import PowerIteration

# from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.HeunEuler import HeunEuler
# from VISolver.Solvers.AdamsBashforthEuler import ABEuler
from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from IPython import embed


def Demo():

    #__POWER_ITERATION__##################################################

    # Define Domain
    A = np.asarray([[-4,10],[7,5]])
    # size = 10000
    # A = np.random.rand(size,size)
    Domain = PowerIteration(A=A)

    # Set Method
    # Method = Euler(Domain=Domain,FixStep=True)
    # Method = HeunEuler(Domain=Domain,Delta0=1e-3)
    # Method = ABEuler(Domain=Domain,Delta0=1e-3)
    Method = CashKarp(Domain=Domain,Delta0=1e-3)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=0.5)
    Term = Termination(MaxIter=100,Tols=[(Domain.res_norm,1e-4)])
    Repo = Reporting(Requests=[Domain.res_norm, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    PowerIteration_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,PowerIteration_Results,Method,toc)

    # Plot Results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = PowerIteration_Results.PermStorage['Data']
    res = PowerIteration_Results.PermStorage[Domain.res_norm]
    ax.plot(res)
    plt.show()
    embed()

if __name__ == '__main__':
    Demo()
