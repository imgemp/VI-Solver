import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.NewDomain import NewDomain

from VISolver.Solvers.Euler import Euler

from VISolver.Projection import PolytopeProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    # __CONSTRAINED_OPTIMIZATION__##############################################

    # Define Domain: f(x) = x_1 + 2*x_2 - 1
    Domain = NewDomain(F=lambda x: np.array([1,2]),Dim=2)
    Domain.Min = 0.
    Domain.f_Error = lambda x: x[0] + 2*x[1] - 1 - Domain.Min

    # Define Polytope Constraints: Gx<=h & Ax=b
    G = -np.eye(Domain.Dim)
    h = np.zeros(Domain.Dim)
    A = np.ones((1,Domain.Dim))
    b = 1.

    # Set Method
    P = PolytopeProjection(G=G,h=h,A=A,b=b)
    Method = Euler(Domain=Domain,FixStep=True,P=P)

    # Initialize Starting Point
    Start = np.array([0,1])

    # Set Options
    Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=1000,Tols=[(Domain.f_Error,1e-6)])
    Repo = Reporting(Requests=[Domain.f_Error, 'Step', 'F Evaluations',
                               'Projections', 'Data'])
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

    error = np.asarray(Results.PermStorage[Domain.f_Error])

    plt.plot(error,'-o',lw=2)
    plt.title('Projected Subgradient')
    plt.xlabel('Iterations')
    plt.ylabel('Objective error')
    plt.ylim([-.1,1.1])
    plt.show()


if __name__ == '__main__':
    Demo()
