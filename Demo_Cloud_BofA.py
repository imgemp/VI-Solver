# import time
import numpy as np

from VISolver.Domains.CloudServices import (
    CloudServices, CreateNetworkExample)

# from VISolver.Solvers.Euler_LEGS import Euler_LEGS
# from VISolver.Solvers.HeunEuler_LEGS import HeunEuler_LEGS
from VISolver.Solvers.AdamsBashforthEuler_LEGS import ABEuler_LEGS
# from VISolver.Solvers.CashKarp_LEGS import CashKarp_LEGS

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimStats

from VISolver.Utilities import (
    ListONP2NP, aug_grid, MCLET_BofA_ID_par, ind2int)

from matplotlib.colors import colorConverter
from matplotlib import pyplot as plt
import matplotlib as mpl
from IPython import embed

from sklearn.svm import SVC


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    eps = 1e-2
    # Method = Euler_LEGS(Domain=Domain,P=BoxProjection(lo=eps))
    # Method = HeunEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    Method = ABEuler_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-1)
    # Method = CashKarp_LEGS(Domain=Domain,P=BoxProjection(lo=eps),Delta0=1e-6)

    # Set Options
    Init = Initialization(Step=-1e-5)
    Term = Termination(MaxIter=1e5)  # ,Tols=[(Domain.valid,False)])
    Repo = Reporting(Requests=['Data','Step'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)
    args = (Method,Domain,Options)
    sim = Solve

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    grid = [np.array([.5,4.5,6])]*Domain.Dim
    grid = ListONP2NP(grid)
    grid = aug_grid(grid)
    Dinv = np.diag(1./grid[:,3])

    results = MCLET_BofA_ID_par(sim,args,grid,nodes=16,limit=20,AVG=.01,
                                eta_1=1.2,eta_2=.95,eps=1.,
                                L=16,q=8,r=1.1,Dinv=Dinv)
    ref, data, p, iters, avg, bndry_ids, starts = results

    # # Observed dimensions + fixed values for others
    # obs = (0,1)
    # consts = np.ones(Domain.Dim)

    # # grid test cases for learned SVM classifier
    # xx, yy = np.meshgrid(np.linspace(grid[obs[0],0],grid[obs[0],1],500),
    #                      np.linspace(grid[obs[1],0],grid[obs[1],1],500))
    # padding = np.ones(len(xx.ravel()))
    # test = ()
    # for i in xrange(Domain.Dim):
    #     if i == obs[0]:
    #         test += (xx.ravel(),)
    #     elif i == obs[1]:
    #         test += (yy.ravel(),)
    #     else:
    #         test += (padding*consts[i],)
    # test = np.vstack(test).T

    # plt.figure()
    # c = plt.cm.hsv(np.random.rand(len(ref)))
    # white = colorConverter.to_rgba('white')
    # Zs = np.zeros((500,500,len(ref)))
    # for cat,lam in enumerate(ref):

    #     samples = data[hash(str(lam))]
    #     if samples != []:
    #         n = len(samples)
    #         m = len(samples[0][0])
    #         X = np.empty((n*2,m))
    #         for idx,sample in enumerate(samples):
    #             X[idx] = sample[0]
    #             X[idx+len(samples)] = sample[1]
    #         Y = np.zeros(len(samples)*2)
    #         Y[:n] = 1

    #         clf = SVC()
    #         clf.fit(X,Y)

    #         Z = clf.decision_function(test)
    #         Z = Z.reshape(xx.shape)
    #         Zs[:,:,cat] = Z

    #         plt.scatter(X[:n, obs[0]], X[:n, obs[1]], s=30, c=c[cat], zorder=2)

    # best_guess = np.argmax(Zs,axis=2)
    # for cat in set(best_guess.flatten()):
    #     Zma = np.ma.masked_where(best_guess != cat,Zs[:,:,cat])
    #     cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',
    #                                                         [white,c[cat]],256)
    #     cmap.set_bad(color='w',alpha=0.0)

    #     # for decision boundaries
    #     Zs_notcat = np.concatenate((Zs[:,:,:cat],Zs[:,:,cat+1:]),axis=2)
    #     diff = Zs[:,:,cat] - np.max(Zs_notcat,axis=2)
    #     plt.contour(xx, yy, diff, colors='k', levels=[0], linewidths=2,
    #                 linetypes='.-', zorder=1)

    #     plt.imshow(Zma, interpolation='nearest',
    #                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #                aspect='auto', origin='lower', cmap=cmap, zorder=0)

    # ax = plt.gca()
    # ax.set_xlim([grid[obs[0],0],grid[obs[0],1]])
    # ax.set_ylim([grid[obs[1],0],grid[obs[1],1]])
    # ax.set_aspect('equal')
    # plt.savefig('bndry_pts.png',format='png')

    # plt.figure()
    # pmap = np.zeros((grid[obs[1],2],grid[obs[0],2]))
    # ind_exp = np.array([int(i) for i in (consts-grid[:,0])//grid[:,3]])
    # for ind_x in xrange(int(grid[obs[0],2])):
    #     for ind_y in xrange(int(grid[obs[1],2])):
    #         ind_exp[obs[0]] = ind_x
    #         ind_exp[obs[1]] = ind_y
    #         p_id = ind2int(tuple(ind_exp),tuple(grid[:,2]))
    #         pmap[ind_y,ind_x] = p[p_id]
    # plt.imshow(pmap,cmap='jet',origin='lower')
    # plt.gca().set_aspect('equal')

    # plt.show()

    embed()

if __name__ == '__main__':
    Demo()
