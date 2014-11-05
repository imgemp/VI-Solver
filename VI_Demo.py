import time
import numpy as np

from Domains.Sphere import Sphere
from Domains.Watson import Watson
from Domains.KojimaShindo import KojimaShindo
from Domains.Sun import Sun

from Solvers.HeunEuler import HeunEuler

from Projection import EntropicProjection, IdentityProjection
from Solver import Solve
from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from Log import PrintSimResults, PrintSimStats


def Demo():

    #__SPHERE__##################################################

    # Define Dimension and Domain
    Domain = Sphere(Dim=100)

    # Set Method
    Method = HeunEuler(Function=Domain.F,P=IdentityProjection(),History=0,Delta0=1e-2)

	# Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=1000,Tols=[[Domain.f_Error,1e-3]])
    Repo = Reporting(MaxData=1,Requests=[Domain.f_Error])
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
    PrintSimResults(SPHERE_Results,Method,toc)

    # Zero Projections
    Method.Proj.NP = 0

    #__KOJIMA-SHINDO__##################################################

    # Define Dimension and Domain
    Domain = KojimaShindo()

    # Set Method
    Method = HeunEuler(Function=Domain.F,P=EntropicProjection(),History=0,Delta0=1e-1)

    # Set Options
    Init = Initialization(Step=-1e-1)
    Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
    Repo = Reporting(MaxData=1,Requests=[Domain.gap_simplex])
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
    PrintSimResults(KS_Results,Method,toc)

    # Zero Projections
    Method.Proj.NP = 0

    #__WATSON__##################################################

    trials = xrange(10)
    WAT_Results = [[] for i in trials]

    for p in trials:

        #Define Dimension and Domain
        Domain = Watson(Pos=p)

        # Set Method
        Method = HeunEuler(Function=Domain.F,P=EntropicProjection(),History=0,Delta0=1e-1)

        # Set Options
        Init = Initialization(Step=-1e-1)
        Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
        Repo = Reporting(MaxData=1,Requests=[Domain.gap_simplex])
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
        PrintSimResults(WAT_Results[p],Method,toc)

        # Zero Projections
        Method.Proj.NP = 0

    #__SUN__##################################################

    trials = xrange(8000,10000+1,2000)
    Sun_Results = [[] for i in trials]

    for n in trials:

        #Define Dimension and Domain
        Domain = Sun(Dim=n)

        # Set Method
        Method = HeunEuler(Function=Domain.F,P=EntropicProjection(),History=0,Delta0=1e-1)

        # Set Options
        Init = Initialization(Step=-1e-1)
        Term = Termination(MaxIter=1000,Tols=[[Domain.gap_simplex,1e-3]])
        Repo = Reporting(MaxData=1,Requests=[Domain.gap_simplex])
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
        PrintSimResults(Sun_Results[ind],Method,toc)

        # Zero Projections
        Method.Proj.NP = 0

if __name__ == '__main__':
  Demo()







