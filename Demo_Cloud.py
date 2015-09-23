import time
import numpy as np

from VISolver.Domains.CloudServices import (
    CloudServices, CreateNetworkExample)

from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=5)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-3)
    Term = Termination(MaxIter=1e5,Tols=[(Domain.gap_rplus,1e-12*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations',
                               'Projections','Data',Domain.eig_stats,
                               'Lyapunov'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    CloudServices_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,CloudServices_Results,Method,toc)

    x = CloudServices_Results.PermStorage['Data'][-1]
    print('p,q,Q,pi,Nash?')
    print(x[:Domain.Dim//2])
    print(x[Domain.Dim//2:])
    print(Domain.Demand(x)[0])
    print(Domain.CloudProfits(x))
    print(all(Domain.Nash(x)[0]))

if __name__ == '__main__':
    Demo()
