import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.MonotoneCycle import MonotoneCycle

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG

from VISolver.Projection import IdentityProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP


def Demo():

    #__SPHERE__##################################################

    # Define Dimension and Domain
    Domain = MonotoneCycle()

    # Set Method
    MethodEuler = Euler(Domain=Domain,P=IdentityProjection(),FixStep=True)
    MethodEG = EG(Domain=Domain,P=IdentityProjection(),FixStep=True)

    # Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=100)
    Repo = Reporting(Requests=['Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Print Stats
    PrintSimStats(Domain,MethodEuler,Options)

    # Start Solver
    tic = time.time()
    Euler_Results = Solve(Start,MethodEuler,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,Euler_Results,MethodEuler,toc)

    # Print Stats
    PrintSimStats(Domain,MethodEG,Options)

    # Start Solver
    tic = time.time()
    EG_Results = Solve(Start,MethodEG,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,EG_Results,MethodEG,toc)

    data_Euler = ListONP2NP(Euler_Results.PermStorage['Data'])
    data_EG = ListONP2NP(EG_Results.PermStorage['Data'])
    

    X, Y = np.meshgrid(np.arange(-2.5, 2.5, .2), np.arange(-2.5, 2.5, .2))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = -Domain.F([X[i,j],Y[i,j]])
            U[i,j] = vec[0]
            V[i,j] = vec[1]

    # plt.figure()
    # plt.title('Arrows scale with plot width, not view')
    # Q = plt.quiver(X, Y, U, V, units='width')
    # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    #                    coordinates='figure')

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    plt.title("Extragradient vs Simultaneous Gradient Descent")
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
                   pivot='mid', units='inches')
    vec_Euler = -Domain.F(data_Euler[-1,:])
    vec_EG = -Domain.F(data_EG[-1,:])
    print(data_Euler[-1])
    print(vec_Euler)
    plt.quiver(data_Euler[-1][0], data_Euler[-1][1], vec_Euler[0], vec_Euler[1], pivot='mid', units='inches', color='b',
            headwidth=5)
    plt.quiver(data_EG[-1][0], data_EG[-1][1], vec_EG[0], vec_EG[1], pivot='mid', units='inches', color='r',
            headwidth=5)
    plt.scatter(X[::3, ::3], Y[::3, ::3], color='gray', s=5)
    plt.plot(data_Euler[:,0],data_Euler[:,1],'b',linewidth=5,label='Simultaneous\nGradient Descent')
    plt.plot(data_EG[:,0],data_EG[:,1],'r',linewidth=5,label='Extragradient')
    plt.plot([0],[0],linestyle="None",marker=(5,1,0),markersize=20,color='gold',label='Equilibrium')
    plt.axis([-2.5,2.5,-2.5,2.5])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)
    # plt.ion()
    # plt.show()
    plt.savefig('EGvsEuler.png')


if __name__ == '__main__':
    Demo()
