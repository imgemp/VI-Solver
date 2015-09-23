import time
import numpy as np

from VISolver.Domains.SupplyChain import (
    SupplyChain, CreateRandomNetwork)

from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__SUPPLY_CHAIN__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(I=2,Nm=2,Nd=2,Nr=1,seed=0)
    Domain = SupplyChain(Network=Network,alpha=2)

    # Set Method
    Method = CashKarp(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    x = 10*np.ones(np.product(Domain.x_shape))
    gam = np.ones(np.sum([np.product(g) for g in Domain.gam_shapes]))
    lam = np.zeros(np.sum([np.product(l) for l in Domain.lam_shapes]))
    Start = np.concatenate((x,gam,lam))

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SupplyChain_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SupplyChain_Results,Method,toc)

if __name__ == '__main__':
    Demo()
