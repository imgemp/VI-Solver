import time
import numpy as np

from VISolver.Domains.SOI import SOI, CreateRandomNetwork

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.HeunEuler import HeunEuler

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.pyplot as plt

from IPython import embed


def Demo():

    #__ONLINE_MONOTONE_EQUILIBRATION_DEMO_OF_A_SERVICE_ORIENTED_INTERNET__######

    # Define Number of Different VIs
    N = 10
    np.random.seed(0)

    # Define Initial Network and Domain
    World = np.random.randint(N)
    Worlds = [World]
    Network = CreateRandomNetwork(m=3,n=2,o=2,seed=World)
    Domain = SOI(Network=Network,alpha=2)

    # Define Initial Strategy
    Strategies = [np.zeros(Domain.Dim)]
    eta = 0.1

    for t in xrange(1000):

        #__PERFORM_SINGLE_UPDATE

        print('Time '+str(t))

        # Set Method
        Method = Euler(Domain=Domain,P=BoxProjection(lo=0))

        # Set Options
        Init = Initialization(Step=-eta)
        Term = Termination(MaxIter=1)
        Repo = Reporting(Requests=['Data'])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Run Update
        Result = Solve(Strategies[-1],Method,Domain,Options)

        # Get New Strategy
        Strategy = Result.PermStorage['Data'][-1]
        Strategies += [Strategy]

        #__DEFINE_NEXT_VI

        # Define Initial Network and Domain
        World = np.random.randint(N)
        Worlds += [World]
        Network = CreateRandomNetwork(m=3,n=2,o=2,seed=World)
        Domain = SOI(Network=Network,alpha=2)

    # Scrap Last Strategy / World
    Strategies = np.asarray(Strategies[:-1])
    Worlds = Worlds[:-1]

    # Store Equilibrium Strategies
    Equilibria = dict()

    for w in np.unique(Worlds):

        print('World '+str(w))

        #__FIND_EQUILIBRIUM_SOLUTION_OF_VI

        # Define Initial Network and Domain
        Network = CreateRandomNetwork(m=3,n=2,o=2,seed=w)
        Domain = SOI(Network=Network,alpha=2)

        # Set Method
        Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

        # Initialize Starting Point
        Start = np.zeros(Domain.Dim)

        # Calculate Initial Gap
        gap_0 = Domain.gap_rplus(Start)

        # Set Options
        Init = Initialization(Step=-1e-10)
        Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
        Repo = Reporting(Requests=[Domain.gap_rplus,'Step','Data'])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        # Start Solver
        tic = time.time()
        Results = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,Results,Method,toc)

        # Get Equilibrium Strategy
        Equilibrium = Results.PermStorage['Data'][-1]
        Equilibria[w] = Equilibrium

    # Matched Equilibria & Costs
    Equilibria_Matched = np.asarray([Equilibria[w] for w in Worlds])

    # Compute Mean of Equilibria
    Mean_Equilibrium = np.mean(Equilibria_Matched,axis=0)

    # Compute Strategies Distance From Mean Equilibrium
    Distance_From_Mean = np.linalg.norm(Strategies-Mean_Equilibrium,axis=1)

    # Plot Results
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(Distance_From_Mean,label='Distance from Mean')
    ax1.set_title('Online Monotone Equilibration of Dynamic SOI Network')
    ax1.legend()
    ax1.set_xlabel('Time')

    plt.savefig('OMEfast.png')

    embed()


if __name__ == '__main__':
    Demo()
