import time
import datetime
import numpy as np

from VISolver.Domains.CloudServices3 import CloudServices, CreateRandomNetwork

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG
from VISolver.Solvers.AcceleratedGradient import AG
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.AdamsBashforthEuler import ABEuler
from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import RPlusProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from Log import PrintSimResults, PrintSimStats


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(nClouds=2,nBiz=5,seed=0)
    Domain = CloudServices(Network=Network,alpha=2)

    # Set Method
    Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-8)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = 5*np.ones(Domain.Dim)
    
    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)
    print(Domain.CloudProfit(Start))
    print(Domain.BizProfits(Start))
    print(Domain.dCloudProfit(Start))
    print(Domain.dBizProfits(Start))

	# Set Options
    # Init = Initialization(Step=-1e-10)
    Init = Initialization(Step=-0.001)
    Term = Termination(MaxIter=1000000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations','Projections','Data'])
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
    print(Domain.CloudProfit(x))
    print(Domain.BizProfits(x))
    print(Domain.dCloudProfit(x))
    print(Domain.dBizProfits(x))

if __name__ == '__main__':
  Demo()