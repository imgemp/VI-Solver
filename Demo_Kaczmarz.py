import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.NewDomain import NewDomain

from VISolver.Solvers.Euler import Euler

from VISolver.Projection import HyperplaneProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    # __KACZMARZ__###############################################################

    # Define Domain
    Domain = NewDomain(F=lambda x: 0,Dim=5)

    # Define Hyperplanes
    dim = max(Domain.Dim-1,1)
    N = 3
    hyps = [np.random.rand(Domain.Dim,dim) for n in range(N)]

    # Set Method
    P_random = HyperplaneProjection(hyperplanes=hyps,sequence='random')
    Method = Euler(Domain=Domain,FixStep=True,P=P_random)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=0.)
    Term = Termination(MaxIter=50)
    Repo = Reporting(Requests=['Step', 'Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    data_random = Results.PermStorage['Data']

    # Set Method
    P_cyclic = HyperplaneProjection(hyperplanes=hyps,sequence='cyclic')
    Method = Euler(Domain=Domain,FixStep=True,P=P_cyclic)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    data_cyclic = Results.PermStorage['Data']

    # Set Method
    P_distal = HyperplaneProjection(hyperplanes=hyps,sequence='distal')
    Method = Euler(Domain=Domain,FixStep=True,P=P_distal)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results,Method,toc)

    data_distal = Results.PermStorage['Data']

    # diff_norm_random = np.linalg.norm(np.diff(data_random),axis=1)
    # diff_norm_cyclic = np.linalg.norm(np.diff(data_cyclic),axis=1)
    # diff_norm_distal = np.linalg.norm(np.diff(data_distal),axis=1)\
    # plt.plot(diff_norm_random,label='random')
    # plt.plot(diff_norm_cyclic,label='cyclic')
    # plt.plot(diff_norm_distal,label='distal')
    er = [np.linalg.norm(P_random.errors(d),axis=1).max() for d in data_random]
    ec = [np.linalg.norm(P_cyclic.errors(d),axis=1).max() for d in data_cyclic]
    ed = [np.linalg.norm(P_distal.errors(d),axis=1).max() for d in data_distal]
    plt.plot(er,label='random')
    plt.plot(ec,label='cyclic')
    plt.plot(ed,label='distal')
    plt.legend()
    plt.title('Kaczmarz Algorithm')
    plt.xlabel('Iterations')
    plt.ylabel('Maximum distance from any hyperplane')
    plt.show()


if __name__ == '__main__':
    Demo()
