import time
import numpy as np

from VISolver.Domains.GMGAN import GMGAN, params

from VISolver.Solvers.Euler_LEGS import Euler_LEGS
from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.HeunEuler_PhaseSpace import HeunEuler_PhaseSpace

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from VISolver.Utilities import ListONP2NP

from matplotlib import pyplot as plt

from IPython import embed


def Demo():

    #__Gaussian_Mixture_GAN__#############################################

    # Define Network and Domain
    Domain = GMGAN(dyn='FCC')

    # Set Method
    # Method = Euler_LEGS(Domain=Domain,FixStep=True,NTopLEs=2)
    # Method = HeunEuler_LEGS(Domain=Domain,Delta0=1e-5,NTopLEs=2)
    # Method = CashKarp_LEGS(Domain=Domain,Delta0=1e-5,NTopLEs=2)
    # Method = HeunEuler(Domain=Domain,Delta0=1e-5)
    Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-3,MaxStep=-.5)
    # Method = Euler(Domain=Domain,FixStep=True)

    # Initialize Starting Point
    Start = Domain.get_weights()

    # Set Options
    Init = Initialization(Step=-1.)
    Term = Termination(MaxIter=3e2)
    # Repo = Reporting(Requests=['Step','Data','Lyapunov',Domain.gap])
    Repo = Reporting(Requests=['Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    GMGAN_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,GMGAN_Results,Method,toc)

    Domain.set_weights(GMGAN_Results.TempStorage['Data'][-1])
    fig, ax = Domain.visualize_dist()
    plt.savefig('gmgan_test.png')

    Steps = np.array(GMGAN_Results.PermStorage['Step'])
    plt.clf()
    plt.semilogy(-Steps)
    plt.savefig('steps.png')

    embed()

if __name__ == '__main__':
    Demo()
