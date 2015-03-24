import time

from Domains.Rosenbrock import *
from Solvers.Euler.Extragradient import *
from Solvers.CashKarp import *
from VISolver.Solvers.Solver import solve
from Options import *
from Log import *


def Demo():
    # __ROSENBROCK__##################################################

    # Define Domain
    Domain = Rosenbrock(Dim=1000)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)
    # Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = CashKarp(Domain=Domain, P=RPlusProjection(), Delta0=1e-6)

    # Initialize Starting Point
    Start = -0.5 * np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(step=-1e-10)
    # init = Initialization(Step=-0.1)
    Term = Termination(max_iter=20000, tols=[(Domain.f_Error, 1e-6)])
    Repo = Reporting(
        requests=[
            Domain.f_Error,
            'Step',
            'f Evaluations',
            'Projections'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init, Term, Repo, Misc)

    # Print Stats
    print_sim_stats(Domain, Method, Options)

    # Start Solver
    tic = time.time()
    Rosenbrock_Results = solve(Start, Method, Domain, Options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(Options, Rosenbrock_Results, Method, toc)

    # Zero Projections for Later Use
    Method.Proj.NP = 0


if __name__ == '__main__':
    Demo()
