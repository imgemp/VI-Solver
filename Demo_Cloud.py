import time
import numpy as np

from VISolver.Domains.CloudServices3 import CloudServices, CreateRandomNetwork

# from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.Extragradient import EG
# from VISolver.Solvers.AcceleratedGradient import AG
# from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.AdamsBashforthEuler import ABEuler
# from VISolver.Solvers.CashKarp import CashKarp

# from VISolver.Solvers.Euler_LEGS import Euler_LEGS
# from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
# from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Projection import RPlusProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from matplotlib import pyplot as plt

from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(5,20,seed=0)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-4)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Method = Euler_LEGS(Domain=Domain)
    # Method = HeunEuler_LEGS(Domain=Domain,Delta0=1e-5)
    Method = ABEuler_LEGS(Domain=Domain,Delta0=1e-4)
    # Method = CashKarp_LEGS(Domain=Domain,Delta0=1e-6)

    # Initialize Starting Point
    # Start = 2.5*np.ones(Domain.Dim)
    Start = 3*np.random.rand(Domain.Dim)

    # Calculate Initial Gap
    # gap_0 = Domain.gap_rplus(Start)
    print(Domain.CloudProfits(Start))
    print(Domain.dCloudProfits(Start))

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-0.00001)
    Term = Termination(MaxIter=10e4)  # ,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
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

    print(Domain.c_clouds)
    print(Domain.pref_bizes)

    data = CloudServices_Results.PermStorage['Data']
    plt.plot(data)
    plt.show()

    # Plot Lyapunov Exponents
    plt.plot(CloudServices_Results.PermStorage['Lyapunov'])
    plt.show()

    embed()

if __name__ == '__main__':
    Demo()
