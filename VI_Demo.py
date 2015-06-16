import time

from Domains.Analytical.Sphere import *
from VISolver.Domains.Watson import *
from VISolver.Domains.KojimaShindo import *
from VISolver.Domains.Sun import *
from VISolver.Solvers.Euler.HeunEuler import *
from VISolver.Solvers.Solver import solve
from VISolver.Options import *
from VISolver.Log import *


def demo():
    # __SPHERE__##################################################

    # Define Dimension and Domain
    Domain = Sphere(Dim=100)

    # Set Method
    Method = HeunEuler(
        Function=Domain.F,
        P=IdentityProjection(),
        History=0,
        Delta0=1e-2)

    # Set Options
    Init = Initialization(step=-1e-1)
    Term = Termination(max_iter=1000, tols=[[Domain.f_Error, 1e-3]])
    Repo = Reporting(MaxData=1, requests=[Domain.f_Error])
    Misc = Miscellaneous()
    Options = DescentOptions(Init, Term, Repo, Misc)

    # Initialize Starting Point
    Start = 100 * np.ones(Domain.Dim)

    # Print Stats
    print_sim_stats(Domain, Method, Options)

    # Start Solver
    tic = time.time()
    SPHERE_Results = solve(Start, Method, Domain, Options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(SPHERE_Results, Method, toc)

    # Zero Projections
    Method.Proj.NP = 0

    #__KOJIMA-SHINDO__##################################################

    # Define Dimension and Domain
    Domain = KojimaShindo()

    # Set Method
    Method = HeunEuler(
        Function=Domain.F,
        P=EntropicProjection(),
        History=0,
        Delta0=1e-1)

    # Set Options
    Init = Initialization(step=-1e-1)
    Term = Termination(max_iter=1000, tols=[[Domain.gap_simplex, 1e-3]])
    Repo = Reporting(MaxData=1, requests=[Domain.gap_simplex])
    Misc = Miscellaneous()
    Options = DescentOptions(Init, Term, Repo, Misc)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim) / np.double(Domain.Dim)

    # Print Stats
    print_sim_stats(Domain, Method, Options)

    # Start Solver
    tic = time.time()
    KS_Results = solve(Start, Method, Domain, Options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(KS_Results, Method, toc)

    # Zero Projections
    Method.Proj.NP = 0

    #__WATSON__##################################################

    trials = xrange(10)
    WAT_Results = [[] for i in trials]

    for p in trials:
        # Define Dimension and Domain
        Domain = Watson(Pos=p)

        # Set Method
        Method = HeunEuler(
            Function=Domain.F,
            P=EntropicProjection(),
            History=0,
            Delta0=1e-1)

        # Set Options
        Init = Initialization(step=-1e-1)
        Term = Termination(max_iter=1000, tols=[[Domain.gap_simplex, 1e-3]])
        Repo = Reporting(MaxData=1, requests=[Domain.gap_simplex])
        Misc = Miscellaneous()
        Options = DescentOptions(Init, Term, Repo, Misc)

        # Initialize Starting Point
        Start = np.ones(Domain.Dim) / np.double(Domain.Dim)

        # Print Stats
        print_sim_stats(Domain, Method, Options)

        tic = time.time()
        WAT_Results[p] = solve(Start, Method, Domain, Options)
        toc = time.time() - tic

        # Print Results
        print_sim_results(WAT_Results[p], Method, toc)

        # Zero Projections
        Method.Proj.NP = 0

    #__SUN__##################################################

    trials = xrange(8000, 10000 + 1, 2000)
    Sun_Results = [[] for i in trials]

    for n in trials:
        # Define Dimension and Domain
        Domain = Sun(Dim=n)

        # Set Method
        Method = HeunEuler(
            Function=Domain.F,
            P=EntropicProjection(),
            History=0,
            Delta0=1e-1)

        # Set Options
        Init = Initialization(step=-1e-1)
        Term = Termination(max_iter=1000, tols=[[Domain.gap_simplex, 1e-3]])
        Repo = Reporting(MaxData=1, requests=[Domain.gap_simplex])
        Misc = Miscellaneous()
        Options = DescentOptions(Init, Term, Repo, Misc)

        # Initialize Starting Point
        Start = np.ones(Domain.Dim) / np.double(Domain.Dim)

        # Print Stats
        print_sim_stats(Domain, Method, Options)

        tic = time.time()
        ind = n / 2000 - 4
        Sun_Results[ind] = solve(Start, Method, Domain, Options)
        toc = time.time() - tic

        # Print Results
        print_sim_results(Sun_Results[ind], Method, toc)

        # Zero Projections
        Method.Proj.NP = 0


if __name__ == '__main__':
    demo()
