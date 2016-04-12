import time
import numpy as np

import matplotlib.pyplot as plt

from VISolver.Domains.PowerIteration import PowerIteration, Rayleigh

from VISolver.Solvers.Euler import Euler
from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.HeunEuler_PhaseSpace import HeunEuler_PhaseSpace
from VISolver.Solvers.CashKarp import CashKarp
from VISolver.Solvers.CashKarp_PhaseSpace import CashKarp_PhaseSpace

from VISolver.Projection import NormBallProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

# from scipy.linalg import eigh

from IPython import embed


def Demo():

    # __POWER_ITERATION__##################################################

    # Define Domain
    A = np.asarray([[-4,10],[7,5]])
    A = A.dot(A)  # symmetrize
    # mars = np.load('big_means.npy')
    # A = mars.T.dot(mars)
    eigs = np.linalg.eigvals(A)
    rho = max(eigs)-min(eigs)
    rank = np.count_nonzero(eigs)
    # Domain = PowerIteration(A=A)
    Domain = Rayleigh(A=A)

    # Set Method
    Method_Standard = Euler(Domain=Domain,FixStep=True,P=NormBallProjection())

    # Initialize Starting Point
    Start = np.ones(Domain.Dim)

    # Set Options
    Init = Initialization(Step=-1e-3)
    Term = Termination(MaxIter=100,Tols=[(Domain.res_norm,1e-6)])
    Repo = Reporting(Requests=[Domain.res_norm, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method_Standard,Options)

    # Start Solver
    tic = time.time()
    Results_Standard = Solve(Start,Method_Standard,Domain,Options)
    toc_standard = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results_Standard,Method_Standard,toc_standard)

    # data_standard = Results_Standard.PermStorage['Data']
    # eigval_standard = (A.dot(data_standard[-1])/data_standard[-1]).mean()
    # eigvec_standard = data_standard[-1]
    res_standard = Results_Standard.PermStorage[Domain.res_norm]

    # Set Method
    # Method_CK = CashKarp(Domain=Domain,Delta0=1e-4,P=NormBallProjection())
    Method_CK = HeunEuler(Domain=Domain,Delta0=1e-4,P=NormBallProjection())

    # Print Stats
    PrintSimStats(Domain,Method_CK,Options)

    # Start Solver
    tic = time.time()
    Results_CK = Solve(Start,Method_CK,Domain,Options)
    toc_CK = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results_CK,Method_CK,toc_CK)

    # data_CK = Results_CK.PermStorage['Data']
    # eigval_CK = (A.dot(data_CK[-1])/data_CK[-1]).mean()
    # eigvec_CK = data_CK[-1]
    res_CK = Results_CK.PermStorage[Domain.res_norm]

    # Set Method
    # Method_CKPS = CashKarp_PhaseSpace(Domain=Domain,Delta0=1e-4,
    #                                   P=NormBallProjection())
    Method_CKPS = HeunEuler_PhaseSpace(Domain=Domain,Delta0=1e-1,
                                       P=NormBallProjection())

    # Print Stats
    PrintSimStats(Domain,Method_CKPS,Options)

    # Start Solver
    tic = time.time()
    Results_CKPS = Solve(Start,Method_CKPS,Domain,Options)
    toc_CKPS = time.time() - tic

    # Print Results
    PrintSimResults(Options,Results_CK,Method_CK,toc_CKPS)

    # data_CKPS = Results_CKPS.PermStorage['Data']
    # eigval_CKPS = (A.dot(data_CKPS[-1])/data_CKPS[-1]).mean()
    # eigvec_CKPS = data_CKPS[-1]
    res_CKPS = Results_CKPS.PermStorage[Domain.res_norm]

    # tic = time.time()
    # eigval_NP, eigvec_NP = eigh(A,eigvals=(Domain.Dim-1,Domain.Dim-1))
    # toc_NP = time.time() - start

    # Plot Results
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)

    # label = 'Standard Power Iteration with scaling' +\
    #     r' $A \cdot v / ||A \cdot v||$'
    label = 'Standard'
    ax.plot(res_standard,label=label)

    fevals_CK = Results_CK.PermStorage['F Evaluations'][-1]
    # label = Method_CK.__class__.__name__+r' Power Iteration'
    # label += r' $\Delta_0=$'+'{:.0e}'.format(Method_CK.Delta0)
    label = 'CK'
    x = np.linspace(0,fevals_CK,len(res_CK))
    ax.plot(x,res_CK,label=label)

    fevals_CKPS = Results_CKPS.PermStorage['F Evaluations'][-1]
    # label = Method_CKPS.__class__.__name__+' Power Iteration'
    # label += r' $\Delta_0=$'+'{:.0e}'.format(Method_CKPS.Delta0)
    label = 'CKPS'
    x = np.linspace(0,fevals_CKPS,len(res_CKPS))
    ax.plot(x,res_CKPS,'-.',label=label)

    xlabel = r'# of $A \cdot v$ Evaluations'
    ax.set_xlabel(xlabel)

    ylabel = r'Norm of residual ($||\frac{A \cdot v}{||A \cdot v||}$'
    ylabel += r'$ - \frac{v}{||v||}||$)'
    ax.set_ylabel(ylabel)

    sizestr = str(A.shape[0])+r' $\times$ '+str(A.shape[1])
    if rho > 100:
        rhostr = r'$\rho(A)=$'+'{:.0e}'.format(rho)
    else:
        rhostr = r'$\rho(A)=$'+str(rho)
    rnkstr = r'$rank(A)=$'+str(rank)
    plt.title(sizestr+' Matrix with '+rhostr+', '+rnkstr)

    ax.legend()

    xlim = min(max(len(res_standard),fevals_CK,fevals_CKPS),Term.Tols[0])
    xlim = int(np.ceil(xlim/10.)*10)
    ax.set_xlim([0,xlim])

    ax.set_yscale('log',nonposy='clip')

    ax2 = fig.add_subplot(2,1,2)

    # label = 'Standard Power Iteration with scaling' +\
    #     r' $A \cdot v / ||A \cdot v||$'
    label = 'Standard'
    ax2.plot(res_standard,label=label)

    # label = Method_CK.__class__.__name__+r' Power Iteration'
    # label += r' $\Delta_0=$'+'{:.0e}'.format(Method_CK.Delta0)
    label = 'CK'
    ax2.plot(res_CK,label=label)

    # label = Method_CKPS.__class__.__name__+' Power Iteration'
    # label += r' $\Delta_0=$'+'{:.0e}'.format(Method_CKPS.Delta0)
    label = 'CKPS'
    ax2.plot(res_CKPS,'-.',label=label)

    xlabel = r'# of Iterations'
    ax2.set_xlabel(xlabel)

    ylabel = r'Norm of residual ($||\frac{A \cdot v}{||A \cdot v||}$'
    ylabel += r'$ - \frac{v}{||v||}||$)'
    ax2.set_ylabel(ylabel)

    ax2.legend()

    xlim = min(max(len(res_standard),len(res_CK),len(res_CKPS)),Term.Tols[0])
    xlim = int(np.ceil(xlim/10.)*10)
    ax2.set_xlim([0,xlim])

    ax2.set_yscale('log',nonposy='clip')

    plt.show()

    embed()

if __name__ == '__main__':
    Demo()
