import time
import numpy as np

from VISolver.Domains.Test import Test

from VISolver.Solvers.HeunEuler import HeunEuler

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from matplotlib import pyplot as plt


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Domain = Test()

    # Set Method
    Method = HeunEuler(Domain=Domain,Delta0=1e-5)

    # Initialize Starting Point
    # Start = np.array([0,-1.0e-0])
    # Start = np.array([0,-2.0e-0])
    # Start = np.array([1.05151222,2.11233182])
    # Start = np.array([0,0])
    Start = np.array([0.,.1])

    # Set Options
    Init = Initialization(Step=1e-3)
    # Init = Initialization(Step=.1)
    Term = Termination(MaxIter=1e4)
    Repo = Reporting(Requests=['Step','Data'])
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
