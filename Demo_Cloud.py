import time
import numpy as np

from VISolver.Domains.CloudServices2 import (
    CloudServices, CreateRandomNetwork, CreateNetworkExample)

# from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
# from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
# from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import pyplot as plt

from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(5,4,seed=0)
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    # Method = Euler_LEGS(Domain=Domain,P=BoxProjection(lo=eps))
    Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)
    # Method = ABEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    # Method = CashKarp_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)
    # Start = np.array([ 0.5,  4.5,  2.1,  4.5,  4.5,  2.9,  0.5,  2.9,  0.5,  1.3])
    # Start = np.array([ 2.1,  4.5,  1.3,  1.3,  2.1,  3.7,  1.3,  2.1,  3.7,  1.3])
    # Start = np.array([ 0.5,  4.5,  2.1,  4.5,  3.7,  2.9,  0.5,  2.9,  0.5,  2.1])
    # Start = np.array([ 0.5,  2.9,  1.1,  0.5,  2.3,  2.3,  3.5,  1.1,  3.5,  1.7])
    # Start = np.array([ 2.9,  2.9,  2.9,  0.5,  1.7,  0.5,  3.5,  2.9,  3.5,  3.5])

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)
    print(Domain.CloudProfits(Start))
    print(Domain.dCloudProfits(Start))

    # Set Options
    Init = Initialization(Step=-1e-3)
    # Init = Initialization(Step=-0.00001)
    Term = Termination(MaxIter=1e5,Tols=[(Domain.gap_rplus,1e-8*gap_0)])
                                         # (Domain.valid,False)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations',
                               'Projections','Data',Domain.eig_stats,
                               'Lyapunov'])
    # Repo = Reporting()
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
    print('[p...q]')
    print(x)
    print('Qj')
    print(Domain.Demand(x)[0])
    print('Profits')
    print(Domain.CloudProfits(x))
    print('[dpj...dqj')
    print(Domain.dCloudProfits(x))
    print('Qij')
    print(Domain.Demand_IJ(x)[0])

    # data = CloudServices_Results.PermStorage['Data']
    # data = ListONP2NP(data)
    # plt.plot(data)
    # plt.show()

    # t = np.abs(np.cumsum(CloudServices_Results.PermStorage['Step']))
    # plt.plot(t,data)
    # plt.show()

    # # Plot Lyapunov Exponents
    # plt.plot(CloudServices_Results.PermStorage['Lyapunov'])
    # plt.show()

    # for i in xrange(data.shape[1]/2):
    #     plt.plot(data[:,i],data[:,i+data.shape[1]/2])
    # plt.show()

    # gaps = CloudServices_Results.PermStorage[Domain.gap_rplus]
    # gaps = ListONP2NP(gaps)
    # plt.plot(gaps)
    # plt.show()

    # X = np.arange(.1,5,0.01)
    # Y = np.arange(.1,5,0.01)
    # X, Y = np.meshgrid(X,Y)
    # Z = np.zeros_like(X)
    # for i in xrange(X.shape[0]):
    #     for j in xrange(Y.shape[0]):
    #         pJ = np.sum(x[:Domain.nClouds])-x[0]
    #         qJ = np.sum(x[Domain.nClouds:])-x[Domain.nClouds]
    #         pi = X[i,j]
    #         qi = Y[i,j]
    #         xi = np.array([pi,qi])
    #         Z[i,j] = Domain.CloudProfit(0,xi,pJ,qJ)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,
    #                        cmap=cm.coolwarm,linewidth=0,
    #                        antialiased=False)
    # ax.set_zlim(0,35)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    embed()

if __name__ == '__main__':
    Demo()
