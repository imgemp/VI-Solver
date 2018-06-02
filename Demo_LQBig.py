import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from VISolver.Domains.LQBig import LQ

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.Extragradient import EG
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.HeunEuler_PhaseSpace import HeunEuler_PhaseSpace

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats
from IPython import embed

from VISolver.Utilities import approx_jacobian

def Demo():

    #__LQ_GAN__##############################################

    # NEED TO WRITE CODE TO GENERATE N LQ-GANS PER DIMENSION FOR M DIMENSIONS
    # THEN RUN EACH ALGORITHM FROM L STARTING POINTS AND MEASURE RUNTIME AND STEPS
    # UNTIL DESIRED DEGREE OF ACCURACY IS MET, MEASURE ACCURACY WITH EUCLIDEAN DISTANCE TO X^*
    # AND KL DIVERGENCE https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians/60699
    # COMPUTE MIN/MAX/AVG/STD RUNTIME OVER N X L RUNS PER DIMENSION

    # Interpolation between F and RipCurl should probably be nonlinear
    # in terms of L2 norms of matrices if the norms essentially represent the largest eigenvalues

    # what's an example of a pseudomonotone field? stretch vertically linearly a bit? quasimonotone?

    # the extragradient method uses the same step size for the first and second step, as the step size goes
    # to zero, extragradient asymptotes to the projection method
    # modified extragradient methods use different step sizes. if we keep the first step size fixed to some
    # positive value and shrink the second, the dynamics of extragradient remain as desired
    # this is essentially what HE_PhaseSpace is showing
    # HE and HE_PhaseSpace are designed to "simulate" a trajectory - they do not actually change the effective
    # dynamics of the vector field

    # Define Network and Domain
    dim = 1
    s = (dim**2+dim)//2
    mu = 10*np.random.rand(dim)
    mu = np.array([0])
    L = 10*np.random.rand(dim,dim)
    L[range(dim),range(dim)] = np.clip(L[range(dim),range(dim)],1e-1,np.inf)
    L = np.tril(L)
    sig = np.dot(L,L.T)
    sig = np.array([[1]])

    np.set_printoptions(linewidth=200)

    print('mu, sig, sig eigs')
    print(mu)
    print(sig)
    print(np.linalg.eigvals(sig))

    # Set Constraints
    loA = -np.inf*np.ones((dim,dim))
    loA[range(dim),range(dim)] = 1e-2
    lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
    P = BoxProjection(lo=lo)

    xoff = 0
    yoff = 0
    scale = 30

    datas = []
    dists = []
    methods = ['ccGD','simGD','preEG','conGD','regGD','EG']
    # methods = ['simGD']
    # methods = ['ccGD']
    for method in methods:

        if method == 'EG':
            Step = -1e-2
            Iters = 10000
            Domain = LQ(mu=mu,sig=sig,method='simGD')
            Method = EG(Domain=Domain,FixStep=True,P=P)
        elif method == 'simGD':
            Step = -1e-3
            Iters = 100000
            Domain = LQ(mu=mu,sig=sig,method='simGD')
            Method = Euler(Domain=Domain,FixStep=True,P=P)
            # Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,MinStep=-1.,P=P)
        elif method == 'conGD':
            Step = -1e-6
            # Iters = 2750
            Iters = 3000
            Domain = LQ(mu=mu,sig=sig,method=method)
            Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
        else:
            Step = -1e-5
            Iters = 10000
            Domain = LQ(mu=mu,sig=sig,method=method)
            Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
        
        # Initialize Starting Point
        Start = np.array([50.,0.,30.,0.])

        # Set Options
        Init = Initialization(Step=Step)
        Term = Termination(MaxIter=Iters,Tols=[(Domain.dist,1e-4)])
        Repo = Reporting(Requests=['Step', 'F Evaluations',
                                   'Projections','Data',Domain.dist])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        # Start Solver
        tic = time.time()
        LQ_Results = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,LQ_Results,Method,toc)

        datas += [np.array(LQ_Results.PermStorage['Data'])]
        dists += [np.array(LQ_Results.PermStorage[Domain.dist])]

    X, Y = np.meshgrid(np.arange(-2*scale + xoff, 2*scale + xoff, .2*scale), np.arange(1e-2 + yoff, 4*scale + yoff, .2*scale))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = -Domain.F(np.array([X[i,j],0.,Y[i,j],0.]))
            U[i,j] = vec[0]
            V[i,j] = vec[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
                   pivot='mid', units='inches')
    colors = ['r','gray','b','k','g','m']
    # colors = ['gray']
    # colors = ['r']
    for data, color, method in zip(datas,colors,methods):
        if method == 'EG':
            ax.plot(data[:,0],data[:,2],'--',color=color,label=method)
        else:
            ax.plot(data[:,0],data[:,2],color=color,label=method)
    ax.plot([data[0,0]],[data[0,2]],'k*')
    ax.plot(mu,sig[0],'c*')
    ax.set_xlim([-2*scale + xoff,2*scale + xoff])
    ax.set_ylim([-.1*scale + yoff,4*scale + yoff])
    ax.set_xlabel(r'$w_2$')
    ax.set_ylabel(r'$a$')
    plt.title('Trajectories for Various Equilibrium Algorithms')
    plt.legend()
    # plt.show()
    # plt.savefig('original.png')
    # plt.savefig('EGoriginal.png')
    # plt.savefig('RipCurl.png')
    # plt.savefig('RipCurl2.png')
    # plt.savefig('EG.png')
    # plt.savefig('GReg.png')
    # plt.savefig('Testing.png')
    # plt.savefig('trajcomp_ccGD.png')
    # plt.savefig('trajcomp.png')
    plt.savefig('trajcomp_test.png')

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # # colors = ['r','gray','b','k','g']
    # # for dist, color, method in zip(dists,colors,methods):
    # #     ax.plot(dist,color=color,label=method)
    # # ax.set_xlabel('Iterations')
    # # ax.set_ylabel('Distance to Equilibrium')
    # # plt.legend()
    # # plt.title('Iteration vs Distance to Equilibrium for Various Equilibrium Algorithms')
    # # plt.savefig('runtimecomp.png')

    # fig,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
    # for dist, color, method in zip(dists,colors,methods):
    #     # if method == 'simGD':
    #     iters = np.arange(dist.shape[0])/1000
    #     ax.plot(iters,dist,color=color,label=method)
    #     ax2.plot(iters,dist,color=color,label=method)
    #     # else:
    #     #     ax.plot(dist,color=color,label=method)
    # ax.set_xlim(0,10)
    # ax2.set_xlim(50,100)
    # ax.set_ylim(-5,95)
    # ax2.set_ylim(-5,95)

    # # hide the spines between ax and ax2
    # ax.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    # ax2.yaxis.tick_right()
    # ax2.set_xticks([75,100])

    # d = .015 # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((1-d,1+d), (-d,+d), **kwargs)
    # ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    # ax2.plot((-d,+d), (-d,+d), **kwargs)

    # # ax.set_xlabel('Iterations')
    # fig.text(0.5, 0.02, 'Thousand Iterations', ha='center', fontsize=12)
    # ax.set_ylabel('Distance to Equilibrium',fontsize=12)
    # # ax.legend()
    # ax2.legend()
    # plt.suptitle('Iteration vs Distance to Equilibrium for Various Equilibrium Algorithms')
    # plt.savefig('runtimecomp.png')

    # # Set Constraints
    # loA = -np.inf*np.ones((dim,dim))
    # loA[range(dim),range(dim)] = 1e-4
    # lo = np.hstack(([-np.inf]*(dim+s), loA[np.tril_indices(dim)], [-np.inf]*dim))
    # P = BoxProjection(lo=lo)

    # xlo, xhi = -5, 5
    # ylo, yhi = 0, 2
    # # methods = ['ccGD','preEG','conGD','regGD']
    # # methods = ['conGD']
    # methods = ['ccGD','preEG']
    # # colors = ['r','b','k','g']
    # # colors = ['k']
    # colors = ['r','b']
    # for method, color in zip(methods,colors):
    #     Step = -1e-5
    #     Iters = 10000
    #     Domain = LQ(mu=mu,sig=sig,method=method)
    #     Method = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-5,P=P)
    #     # Iters = 10000
    #     # Method = Euler(Domain=Domain,FixStep=True)
        
    #     # Initialize Starting Point
    #     Start = np.array([3.,0.,0.2,0.])

    #     # Set Options
    #     Init = Initialization(Step=Step)
    #     Term = Termination(MaxIter=Iters,Tols=[(Domain.dist,1e-4)])
    #     Repo = Reporting(Requests=['Step', 'F Evaluations',
    #                                'Projections','Data',Domain.dist])
    #     Misc = Miscellaneous()
    #     Options = DescentOptions(Init,Term,Repo,Misc)

    #     # Print Stats
    #     PrintSimStats(Domain,Method,Options)

    #     # Start Solver
    #     tic = time.time()
    #     LQ_Results = Solve(Start,Method,Domain,Options)
    #     toc = time.time() - tic

    #     # Print Results
    #     PrintSimResults(Options,LQ_Results,Method,toc)

    #     data = np.array(LQ_Results.PermStorage['Data'])
        
    #     X, Y = np.meshgrid(np.linspace(xlo, xhi, 50), np.linspace(ylo, yhi, 50))
    #     U = np.zeros_like(X)
    #     V = np.zeros_like(Y)
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             vec = -Domain.F(np.array([X[i,j],0.,Y[i,j],0.]))
    #             U[i,j] = vec[0]
    #             V[i,j] = vec[2]

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
    #                    pivot='mid', units='inches')
    #     ax.plot(data[:,0],data[:,2],color=color,label=method)
    #     ax.plot([data[0,0]],[data[0,2]],'k*')
    #     ax.plot(mu,sig[0],'c*')
    #     ax.set_xlim([xlo, xhi])
    #     ax.set_ylim([0, yhi])
    #     ax.set_xlabel(r'$w_2$')
    #     ax.set_ylabel(r'$a$')
    #     plt.title('Dynamics for '+method)
    #     plt.savefig(method+'_dyn')

    # embed()

def MonotoneRegion():
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    delta = 0.025
    w2 = np.arange(-5.0, 5.0, delta)
    a = np.arange(0.0, 5.0, delta)
    W2, A = np.meshgrid(w2, a)
    LAM_cc = -1. + 5.*(A**2.) - np.sqrt(1. + A**4. + 2.*(A**2.)*(-1. + 8.*(W2**2.)))  # They're actually the same boundary!!!!
    # solve for when square root above is less than b^2 and you get same as below, LAM_eg
    LAM_eg = -1. + 3.*(A**2.) - 2.*(W2**2.)
    # You can force all the contours to be the same color.
    plt.figure()
    CS = plt.contour(W2, A, LAM_cc, 10,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Monotone Region: Crossing the Curl')
    plt.savefig('monregion_cc.png')
    plt.close()

    plt.figure()
    CS = plt.contour(W2, A, LAM_eg, 10,
                     colors='k',  # negative contours will be dashed by default
                     )
    plt.clabel(CS, fontsize=9, inline=1)
    plt.title('Monotone Region: Extragradient')
    plt.savefig('monregion_eg.png')
    plt.close()


if __name__ == '__main__':
    Demo()
    # MonotoneRegion()
