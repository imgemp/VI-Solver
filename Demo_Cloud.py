import time
import numpy as np

from VISolver.Domains.CloudServices3 import \
    CloudServices, CreateRandomNetwork, CreateNetworkExample

# from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.Extragradient import EG
# from VISolver.Solvers.AcceleratedGradient import AG
# from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.AdamsBashforthEuler import ABEuler
# from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from matplotlib import pyplot as plt

from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    # Network = CreateRandomNetwork(5,20,seed=0)
    Network = CreateNetworkExample(ex=4)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    # Method = Euler(Domain=Domain,P=BoxProjection(lo=eps))
    # Method = EG(Domain=Domain,P=BoxProjection(lo=eps))
    # Method = AG(Domain=Domain,P=BoxProjection(lo=eps))
    # Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-5)
    # Method = ABEuler(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-4)
    # Method = CashKarp(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-6)

    # Method = Euler_LEGS(Domain=Domain,P=BoxProjection(lo=eps))
    # Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    # Method = ABEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    Method = CashKarp_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-6)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)
    # Start = np.array([23.26294228,17.08159513,13.91994974,8.95500211,7.55630701,
    #                   0.2379811,0.26155469,0.32003472,0.4396889,0.51636412])
    # Start = np.array([25.47437987,17.98056718,14.51108149,9.21580934,7.76625486,
                    # 0.19607033,0.25692857,0.31355422,0.43426494,0.51038626])
    # Start = 2.5*np.ones(Domain.Dim)
    # np.random.seed(0)
    # Start = 5*np.random.rand(Domain.Dim)
    # Start = np.array([9.52791657,11.63158087,12.42043654,10.46083131,23.0806941,
    #                   2.53941643,0.45046114,0.41310984,0.44217791,0.26479585])
    # Start = np.array([9.52791657,11.73396707,12.53424296,10.55151842,23.44745641,
    #                   2.53941643,0.44840324,0.41112556,0.44019972,0.26233588])
    # Start = np.array([9.52791657,12.,13.,11.,24.,
    #                   2.53941643,.5,0.5,0.5,0.26])
   #  9.52791657,11.82611837,12.63622284,10.63321631,23.77959998
   # 2.53941643,0.4465767,0.40937549,0.43844125,0.26016232

    # Calculate Initial Gap
    # gap_0 = Domain.gap_rplus(Start)
    print(Domain.CloudProfits(Start))
    print(Domain.dCloudProfits(Start))

    # Set Options
    Init = Initialization(Step=-1e-5)
    # Init = Initialization(Step=-0.00001)
    Term = Termination(MaxIter=1e4)  # ,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
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
    print(Domain.Demand(x))
    print('Profits')
    print(Domain.CloudProfits(x))
    print('[dpj...dqj')
    print(Domain.dCloudProfits(x))
    print('Qij')
    print(Domain.Demand_IJ(x))

    # print(Domain.c_clouds)
    # print(Domain.pref_bizes)

    data = CloudServices_Results.PermStorage['Data']
    plt.plot(data)
    plt.show()

    # Plot Lyapunov Exponents
    plt.plot(CloudServices_Results.PermStorage['Lyapunov'])
    plt.show()

    data = ListONP2NP(data)
    for i in xrange(data.shape[1]/2):
        plt.plot(data[:,i],data[:,i+data.shape[1]/2])
    plt.show()

    gaps = CloudServices_Results.PermStorage[Domain.gap_rplus]
    gaps = ListONP2NP(gaps)
    plt.plot(gaps)
    plt.show()

    # Start = CloudServices_Results.PermStorage['Data'][-1]
    # CloudServices_Results = Solve(Start,Method,Domain,Options)

    embed()

if __name__ == '__main__':
    Demo()
