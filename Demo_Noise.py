import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from VISolver.Domains.NoisySphere import NoisySphere
from VISolver.Domains.NoisyRosenbrock import NoisyRosenbrock

# from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.Extragradient import EG
# from VISolver.Solvers.AcceleratedGradient import AG
# from VISolver.Solvers.HeunEuler import HeunEuler
# from VISolver.Solvers.AdamsBashforthEuler import ABEuler
# from VISolver.Solvers.CashKarp import CashKarp
from VISolver.Solvers.GempRK2 import GempRK2

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats
from IPython import embed


def Demo():

    #__ROSENBROCK__##################################################

    # Define Domain
    Domain = NoisySphere(Dim=2,Sigma=5.)
    Domain = NoisyRosenbrock(Dim=1000,Sigma=1.)

    # Set Method
    # Method = Euler(Domain=Domain,P=IdentityProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)
    # Method = GempRK2(Domain=Domain,P=IdentityProjection(),Delta0=1e-1)
    Method = GempRK2(Domain=Domain,P=IdentityProjection(),Delta0=1e-2)

    # Initialize Starting Point
    Start = 3*np.ones(Domain.Dim)
    Start = -0.5*np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-01)
    Term = Termination(MaxIter=10000,Tols=[(Domain.f_Error,1e-3)])
    Repo = Reporting(Requests=[Domain.f_Error, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    Sphere_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Sphere_Results,Method,toc)

    # Plot Results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    X = np.arange(-2, 2, 0.25)
    Y = np.arange(0, 3, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in xrange(Z.shape[0]):
        for j in xrange(Z.shape[1]):
            Z[i,j] = Domain.f(np.array([X[i,j],Y[i,j]]))
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,
                    linewidth=0,antialiased=False)
    data = Sphere_Results.PermStorage['Data']
    res = Sphere_Results.PermStorage[Domain.f_Error]
    trajX = []
    trajY = []
    trajZ = []
    for i in xrange(len(data)):
        trajX.append(data[i][0])
        trajY.append(data[i][1])
        trajZ.append(res[i])
    ax.plot(trajX,trajY,trajZ)

    embed()

if __name__ == '__main__':
    Demo()
