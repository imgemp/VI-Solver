import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from VISolver.Domains.Rosenbrock import Rosenbrock

from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__ROSENBROCK__##################################################

    # Define Domain
    Domain = Rosenbrock(Dim=1000)

    # Set Method
    Method = CashKarp(Domain=Domain,Delta0=1e-6)

    # Initialize Starting Point
    Start = -0.5*np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=20000,Tols=[(Domain.f_Error,1e-6)])
    Repo = Reporting(Requests=[Domain.f_Error, 'Step', 'F Evaluations',
                               'Projections','Data'])
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

    # Plot Results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-2, 2, 0.25)
    Y = np.arange(-1, 3, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in xrange(Z.shape[0]):
        for j in xrange(Z.shape[1]):
            Z[i,j] = Domain.f(np.array([X[i,j],Y[i,j]]))
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,
                    linewidth=0,antialiased=False)
    data = Rosenbrock_Results.PermStorage['Data']
    res = Rosenbrock_Results.PermStorage[Domain.f_Error]
    trajX = []
    trajY = []
    trajZ = []
    for i in xrange(len(data)):
        trajX.append(data[i][0])
        trajY.append(data[i][1])
        trajZ.append(res[i])
    ax.plot(trajX,trajY,trajZ)
    plt.show()

if __name__ == '__main__':
    Demo()
