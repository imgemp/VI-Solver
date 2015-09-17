import numpy as np

from VISolver.Domains.Lienard import Lienard

from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS

from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimStats

from VISolver.Utilities import ListONP2NP

from VISolver.BoA.Utilities import aug_grid
from VISolver.BoA.MCGrid_Enhanced import MCT
from VISolver.BoA.Plotting import plotBoA

from IPython import embed


def Demo():

    #__LIENARD_SYSTEM__##################################################

    # Define Network and Domain
    Domain = Lienard()

    # Set Method
    Method = HeunEuler_LEGS(Domain=Domain,Delta0=1e-5)

    # Set Options
    Init = Initialization(Step=1e-5)
    Term = Termination(MaxIter=5e4)
    Repo = Reporting(Requests=['Data','Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
    args = (Method,Domain,Options)
    sim = Solve

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Construct grid
    grid = [np.array([-2.5,2.5,13])]*2
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])
    r = 1.1*max(grid[:,3])

    # Compute Results
    results = MCT(sim,args,grid,nodes=8,parallel=True,limit=5,AVG=-1,
                  eta_1=1.2,eta_2=.95,eps=1.,
                  L=8,q=2,r=r,Dinv=Dinv)
    ref, data, p, i, avg, bndry_ids, starts = results

    # Plot BoAs
    plotBoA(ref,data,grid,wcolor=True,wscatter=True)

    # plt.figure()
    # pmap = np.swapaxes(np.reshape(p,tuple(grid[:,2])),0,1)
    # plt.imshow(pmap,'jet',origin='lower')
    # plt.gca().set_aspect('equal')

    # def fun(file):
    #     cloud = np.load(file)
    #     if len(cloud) > 3:
    #         ref, data, p, iters, avg, bndry_ids, Domain, grid = cloud
    #         starts = [None]
    #     else:
    #         results, Domain, grid = cloud
    #         ref, data, p, iters, avg, bndry_ids, starts = results
    #     datanew = {}
    #     for cat,lam in enumerate(ref):
    #         datanew[hash(repr(lam))] = data[hash(str(lam))]
    #     results = ref, datanew, p, iters, avg, bndry_ids, starts
    #     np.save(file+'_new',[results,Domain,grid])

    # plt.show()

    embed()

if __name__ == '__main__':
    Demo()
