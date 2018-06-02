import time
import numpy as np

from VISolver.Domains.Lienard import Lienard

from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from matplotlib import pyplot as plt


def Demo():

    #__LIENARD_SYSTEM__##################################################

    # Define Network and Domain
    Domain = Lienard()

    # Set Method
    Method = Euler_LEGS(Domain=Domain,FixStep=True,NTopLEs=2)
    # Method = HeunEuler_LEGS(Domain=Domain,Delta0=1e-5,NTopLEs=2)
    # Method = CashKarp_LEGS(Domain=Domain,Delta0=1e-5,NTopLEs=2)

    # Initialize Starting Point
    Start = np.array([1.5,0])

    # Set Options
    Init = Initialization(Step=1e-3)
    Term = Termination(MaxIter=1e5,Tols=[(Domain.gap,1e-3)])
    Repo = Reporting(Requests=['Step','Data','Lyapunov',Domain.gap])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Lienard_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Lienard_Results,Method,toc)

    # Plot Lyapunov Exponents
    plt.plot(Lienard_Results.PermStorage['Lyapunov'])
    plt.show()

    # Plot System Trajectory
    data = ListONP2NP(Lienard_Results.PermStorage['Data'])
    plt.plot(data[:,0],data[:,1])
    ax = plt.gca()
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    Demo()
