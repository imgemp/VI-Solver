import time
import numpy as np

from VISolver.Domains.BloodBank import BloodBank, CreateRandomNetwork

from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__BLOOD_BANK__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(nC=2,nB=2,nD=2,nR=2,seed=0)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    Method = CashKarp(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-6)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    BloodBank_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results,Method,toc)

    opt_path_flows = BloodBank_Results.TempStorage['Data'] [-1]
    f_1C, f_CB, f_BP, f_PS, f_SD, f_DR = Domain.PathFlow2LinkFlow_x2f(Domain.UnpackPathFlows(opt_path_flows))
    print(f_1C)

if __name__ == '__main__':
    Demo()
