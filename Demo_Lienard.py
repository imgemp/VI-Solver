import time
import numpy as np

from VISolver.Domains.Lienard import Lienard

# from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
# from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
# from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from matplotlib import pyplot as plt


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Domain = Lienard()

    # Set Method
    # Method = Euler_LEGS(Domain=Domain)
    Method = HeunEuler_LEGS(Domain=Domain,Delta0=1e-5)
    # Method = ABEuler_LEGS(Domain=Domain,Delta0=1e-4)
    # Method = CashKarp_LEGS(Domain=Domain,Delta0=1e-6)

    # Initialize Starting Point
    # Start = np.array([0,-1.0e-0])
    Start = np.array([0,-2.0e-0])
    # Start = np.array([1.05151222,2.11233182])
    # Start = np.array([0,0])

    # Set Options
    # Init = Initialization(Step=1e-3)
    Init = Initialization(Step=.1)
    Term = Termination(MaxIter=1e5)
    Repo = Reporting(Requests=['Step','Data','Lyapunov'])
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
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    plt.show()

    # probs = [1/50] * 50
    # np.random.multinomial(1, probs).argmax()
    # np.random.choice(50, p=probs)
    # grid = [np.array([.1,10.,6])]*5+[np.array([.1,1.,6])]*5
    # grid = ListONP2NP(grid)
    # Dinv = np.diag(grid[:,3])

if __name__ == '__main__':
    Demo()
