import time
import numpy as np

from VISolver.Domains.MHPH import MHPH

from VISolver.Solvers.HeunEuler import HeunEuler
# from VISolver.Solvers.RipCurl import RipCurl
from VISolver.Solvers.RipCurlEx import RipCurlEx as RipCurl
# from VISolver.Solvers.Extragradient import EG as RipCurl

from VISolver.Projection import EntropicProjection, IdentityProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats


def Demo():

    #__MHPH__##################################################

    trials = range(1000,8000+1,2000)
    MHPH_Results = [[] for i in trials]

    for n in trials:

        #Define Dimension and Domain
        Domain = MHPH(Dim=n)

        # Set Method
        Method = HeunEuler(Domain=Domain,P=EntropicProjection(),Delta0=1e-1)
        # Method = RipCurl(Domain=Domain,P=EntropicProjection(),factor=0.1,FixStep=True)

        # Set Options
        Init = Initialization(Step=-1)
        Term = Termination(MaxIter=100,Tols=[[Domain.gap_simplex,1e-3]])
        Repo = Reporting(Requests=[Domain.gap_simplex])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        #Initialize Starting Point
        Start = np.ones(Domain.Dim)/np.double(Domain.Dim)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        tic = time.time()
        ind = int(n/2000)-4
        MHPH_Results[ind] = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,MHPH_Results[ind],Method,toc)

if __name__ == '__main__':
    Demo()
