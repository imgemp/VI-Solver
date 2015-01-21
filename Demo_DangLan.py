import time
import numpy as np

from VISolver.Domains.Sphere import Sphere
from VISolver.Domains.Watson import Watson
from VISolver.Domains.KojimaShindo import KojimaShindo
from VISolver.Domains.Sun import Sun

from VISolver.Solvers.HeunEuler import HeunEuler

from VISolver.Projection import EntropicProjection, IdentityProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__SPHERE__##################################################

    # Define Dimension and Domain
    Domain = Sphere(Dim=100)

    # Set Method
    Method = HeunEuler(Domain=Domain,P=IdentityProjection(),Delta0=1e-2)

    # Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=1000,Tols=[[Domain.f_Error,1e-3]])
    Repo = Reporting(Requests=[Domain.f_Error])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Initialize Starting Point
    Start = 100*np.ones(Domain.Dim)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SPHERE_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SPHERE_Results,Method,toc)

    #__KOJIMA-SHINDO__##################################################

    # Define Dimension and Domain
    Domain = KojimaShindo()

    # Set Method
    Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-1)

    # Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
    Repo = Reporting(Requests=[Domain.gap_simplex])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    KS_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,KS_Results,Method,toc)

    #__WATSON__##################################################

    trials = xrange(10)
    WAT_Results = [[] for i in trials]

    for p in trials:

        #Define Dimension and Domain
        Domain = Watson(Pos=p)

        # Set Method
        Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-1)

        # Set Options
        Init = Initialization(Step=-1e-1)
        Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
        Repo = Reporting(Requests=[Domain.gap_simplex])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        #Initialize Starting Point
        Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        tic = time.time()
        WAT_Results[p] = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,WAT_Results[p],Method,toc)

    #__SUN__##################################################

    trials = xrange(8000,10000+1,2000)
    Sun_Results = [[] for i in trials]

    for n in trials:

        #Define Dimension and Domain
        Domain = Sun(Dim=n)

        # Set Method
        Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-1)

        # Set Options
        Init = Initialization(Step=-1e-1)
        Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
        Repo = Reporting(Requests=[Domain.gap_simplex])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        #Initialize Starting Point
        Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        tic = time.time()
        ind = n/2000-4
        Sun_Results[ind] = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,Sun_Results[ind],Method,toc)

if __name__ == '__main__':
    Demo()