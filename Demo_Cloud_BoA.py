import numpy as np

from VISolver.Domains.CloudServices import (
    CloudServices, CreateNetworkExample)

from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimStats

from VISolver.Utilities import ListONP2NP
import time

from VISolver.BoA.Utilities import aug_grid
from VISolver.BoA.MCGrid_Enhanced import MCT
from VISolver.BoA.Plotting import plotBoA, plotDistribution

from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e0)

    # Set Options
    Init = Initialization(Step=-1e-3)
    Term = Termination(MaxIter=1e5)
    Repo = Reporting(Requests=['Data','Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
    args = (Method,Domain,Options)
    sim = Solve

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Construct grid
    grid = [np.array([.5,3.5,6])]*Domain.Dim
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])

    # Compute results
    results = MCT(sim,args,grid,nodes=16,limit=40,AVG=0.00,
                  eta_1=1.2,eta_2=.95,eps=1.,
                  L=16,q=8,r=1.1,Dinv=Dinv)
    ref, data, p, iters, avg, bndry_ids, starts = results

    # Save results
    sim_data = [results,Domain,grid]
    np.save('cloud_'+time.strftime("%Y%m%d_%H%M%S"),sim_data)

    # Plot BoAs
    obs = (4,9)  # Look at new green-tech company
    consts = np.array([3.45,2.42,3.21,2.27,np.inf,.76,.97,.75,1.03,np.inf])
    txt_locs = [(1.6008064516129032, 1.6015625),
                (3.2, 3.2),
                (3.33, 2.53)]
    xlabel = '$p_'+repr(obs[0])+'$'
    ylabel = '$q_'+repr(obs[1])+'$'
    title = 'Boundaries of Attraction for Cloud Services Market'
    figBoA, axBoA = plotBoA(ref,data,grid,obs=obs,consts=consts,
                            txt_locs=txt_locs,xlabel=xlabel,ylabel=ylabel,
                            title=title)

    # Plot Probablity Distribution
    figP, axP = plotDistribution(p,grid,obs=obs,consts=consts)

    embed()

if __name__ == '__main__':
    Demo()
