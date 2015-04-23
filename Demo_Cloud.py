import time
import numpy as np

from VISolver.Domains.CloudServicesNew import CloudServices, CreateRandomNetwork

from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.Extragradient import EG
# from VISolver.Solvers.AcceleratedGradient import AG
# from VISolver.Solvers.HeunEuler import HeunEuler
# from VISolver.Solvers.AdamsBashforthEuler import ABEuler
# from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import RPlusProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats
from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork()
    Domain = CloudServices(Network=Network,alpha=2)

    # Set Method
    Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-8)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = .5*np.ones(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)
    print(Domain.CloudProfits(Start))
    print(Domain.dCloudProfits(Start))
    print(Domain.maxfirm_profits(Start))
    print(Domain.argmax_firm_profits(Start))

    # Set Options
    # Init = Initialization(Step=-1e-10)
    Init = Initialization(Step=-0.001)
    Term = Termination(MaxIter=5,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations',
                               'Projections','Data'])
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
    # print(x)
    print(Domain.CloudProfits(x))
    print(Domain.dCloudProfits(x))
    print(Domain.maxfirm_profits(x))
    print(Domain.argmax_firm_profits(x))
    embed()

if __name__ == '__main__':
    Demo()
