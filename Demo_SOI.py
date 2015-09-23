import time
import numpy as np

from VISolver.Domains.SOI import SOI, CreateRandomNetwork

from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__SERVICE_ORIENTED_INTERNET__##############################################

    # Define Network and Domain
    Network = CreateRandomNetwork(m=3,n=2,o=2,seed=0)
    Domain = SOI(Network=Network,alpha=2)

    # Set Method
    Method = CashKarp(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SOI_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SOI_Results,Method,toc)

if __name__ == '__main__':
    Demo()
