import time

from VISolver.Solvers.Extragradient import *
from VISolver.Solvers.CashKarp import *
from VISolver.Solvers.Solver import solve
from VISolver.Log import *
from Domains.Analytical.BloodBank import CreateRandomNetwork
from Domains.Analytical import BloodBank
from VISolver.Solvers.CashKarp import CashKarp
from VISolver.Projection import RPlusProjection
from VISolver.Solver import *
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import print_sim_stats, print_sim_results


def Demo():
    # __BLOOD_BANK__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(nC=2, nB=2, nD=2, nR=2, seed=0)
    Domain = BloodBank(Network=Network, alpha=2)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = CashKarp(Domain=Domain, P=RPlusProjection(), Delta0=1e-6)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(step=-1e-10)
    # init = Initialization(Step=-0.1)
    Term = Termination(max_iter=25000, tols=[(Domain.gap_rplus, 1e-6 * gap_0)])
    Repo = Reporting(
        requests=[
            Domain.gap_rplus,
            'Step',
            'f Evaluations',
            'Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init, Term, Repo, Misc)

    # Print Stats
    print_sim_stats(Domain, Method, Options)

    # Start Solver
    tic = time.time()
    BloodBank_Results = solve(Start, Method, Domain, Options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(Options, BloodBank_Results, Method, toc)


if __name__ == '__main__':
    Demo()
