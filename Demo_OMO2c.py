import time
import numpy as np
from scipy.sparse.linalg import svds

from VISolver.Utilities import approx_jacobian

from VISolver.Domains.SOI import SOI, CreateRandomNetwork
from VISolver.Domains.AverageDomains import AverageDomains
from VISolver.Domains.ContourIntegral import ContourIntegral, LineContour

from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.HeunEuler_PhaseSpace import HeunEuler_PhaseSpace

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.pyplot as plt

from IPython import embed


def Demo():

    #__SERVICE_ORIENTED_INTERNET__##############################################
    N = 10  # number of possible maps
    T = 1000  # number of time steps
    eta = 1e-3  # learning rate

    print('Creating Domains')

    # Define Domains and Compute Equilbria
    Domains = []
    X_Stars = []
    CurlBounds = []
    n = 0
    while len(Domains) < N:
        # Create Domain
        Network = CreateRandomNetwork(m=3,n=2,o=2,seed=None)
        Domain = SOI(Network=Network,alpha=2)

        # Initialize Starting Point
        Start = np.zeros(Domain.Dim)

        # Assert PD
        J = approx_jacobian(Domain.F,Start)
        eigs = np.linalg.eigvals(J+J.T)
        eigs_i = np.abs(np.linalg.eigvals(J-J.T))
        if not np.all(eigs > 0):
            continue
        print(eigs.min(),eigs.max())
        print(eigs_i.min(),eigs_i.max())
        _J = approx_jacobian(Domain.F,Start+0.5)
        assert np.allclose(J,_J,atol=1e-5)  # assert J is constant (unique for SOI)

        # Record Domain
        Domains += [Domain]

        # Calculate Initial Gap
        gap_0 = Domain.gap_rplus(Start)

        # Calculate Curl Bound
        CurlBounds += [np.sqrt(18)*svds(J,k=1,which='LM',return_singular_vectors=False).item()]

        # Set Method
        Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-3)

        # Set Options
        Init = Initialization(Step=-1e-10)
        Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
        Repo = Reporting(Requests=[Domain.gap_rplus])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        # Start Solver
        tic = time.time()
        SOI_Results = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,SOI_Results,Method,toc)

        # Record X_Star
        X_Star = SOI_Results.TempStorage['Data'][-1]
        X_Stars += [X_Star]
        n += 1
    X_Stars = np.asarray(X_Stars)

    print('Starting Online Learning')

    # Set First Prediction
    X = np.zeros(X_Stars.shape[1])
    # X = np.mean(X_Stars,axis=0)
    # X += np.random.rand(*X.shape)*np.linalg.norm(X)

    # Select First Domain
    # idx = np.argmax(np.linalg.norm(X_Stars - X,axis=1))
    idx = np.random.choice(len(Domains))

    # Domain Sequence
    idx_seq = []
    X_seq = []
    F_seq = []

    ts = range(T)
    for t in ts:
        print('t = '+str(t),end='\r')
        # record prediction
        X_seq += [X]
        # record domain
        idx_seq += [idx]
        # retrieve domain
        Domain = Domains[idx]
        # record F
        FX = Domain.F(X)
        F_seq += [FX]
        # update prediction
        X = BoxProjection(lo=0).P(X,-eta,FX)
        # update domain
        # idx = np.argmax(np.linalg.norm(X_Stars - X,axis=1))
        idx = np.random.choice(len(Domains))

    L = np.sqrt(np.mean(np.linalg.norm(F_seq,axis=1)**2.))

    print('Computing Optimal Strategy')

    weights = np.bincount(idx_seq,minlength=len(Domains))/len(idx_seq)
    print(weights)

    # Compute Equilibrium of Average Domain
    Domain = AverageDomains(Domains,weights=weights)

    # Set Method
    Method = HeunEuler_PhaseSpace(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Assert PSD - sum of PSD is PSD doesn't hurt to check
    J = approx_jacobian(Domain.F,Start)
    eigs = np.linalg.eigvals(J+J.T)
    eigs_i = np.abs(np.linalg.eigvals(J-J.T))
    assert np.all(eigs > 0)
    print(eigs.min(),eigs.max())
    print(eigs_i.min(),eigs_i.max())

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-10*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SOI_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SOI_Results,Method,toc)

    print('Computing Regrets')

    # Record X_Opt
    X_Opt = SOI_Results.TempStorage['Data'][-1]
    # X_Opt = X_Stars[0]
    B = np.linalg.norm(X_Opt)

    eta_opt = B/(L*np.sqrt(2*T))
    bound_opt = B*L*np.sqrt(2*T)
    reg_bound = (B**2)/(2*eta) + eta*T*L**2

    distances = []
    loss_infs = []
    regret_standards = []
    regret_news = []
    stokes = []
    ts = range(T)
    for t in ts:
        print('t = '+str(t),end='\r')
        idx = idx_seq[t]
        X = X_seq[t]
        # retrieve domain
        Domain = Domains[idx]
        # retrieve equilibrium / reference vector
        equi = X_Stars[idx]
        # equi = np.zeros_like(X_Stars[idx])
        # calculate distance
        distances += [np.linalg.norm(X_Opt-X)]
        # calculate infinity loss
        # loss_infs += [infinity_loss(Domain,X)]
        # calculate standard regret
        ci_predict = ContourIntegral(Domain,LineContour(equi,X))
        predict_loss = integral(ci_predict)
        ci_opt = ContourIntegral(Domain,LineContour(equi,X_Opt))
        predict_opt = integral(ci_opt)
        regret_standards += [predict_loss - predict_opt]
        # calculate new regret
        ci_new = ContourIntegral(Domain,LineContour(X_Opt,X))
        regret_news += [integral(ci_new)]
        # calculate bound
        # area = 0.5*np.prod(np.sort([np.linalg.norm(X_Opt-equi),np.linalg.norm(X-X_Opt),np.linalg.norm(equi-X)])[:2])  # area upper bound
        area = herons(X_Opt,X,equi)  # exact area
        stokes += [CurlBounds[idx]*area]
        # # update prediction
        # X = BoxProjection(lo=0).P(X,-eta,Domain.F(X))
        # # update domain
        # idx = np.argmax(np.linalg.norm(X_Stars - X,axis=1))
    # embed()
    ts_p1 = range(1,T+1)
    distances_avg = np.divide(np.cumsum(distances),ts_p1)
    # loss_infs_avg = np.divide(np.cumsum(loss_infs),ts_p1)
    regret_standards_avg = np.divide(np.cumsum(regret_standards),ts_p1)
    regret_news_avg = np.divide(np.cumsum(regret_news),ts_p1)
    stokes = np.divide(np.cumsum(stokes),ts_p1)

    # np.savez_compressed('NoRegret_MLN2c.npz',d_avg=distances_avg,
    #                     linf_avg=loss_infs_avg,rs_avg=regret_standards_avg,
    #                     rn_avg=regret_news_avg,stokes=stokes)

    plt.subplot(2, 1, 2)
    plt.plot(ts, distances_avg, 'k',label='Average Distance')
    # plt.title('Demonstration of No-Regret on MLN')
    plt.xlabel('Time Step')
    plt.ylabel('Euclidean Distance')
    plt.legend()

    plt.subplot(2, 1, 1)
    # plt.plot(ts, loss_infs_avg, 'k--', label=r'loss$_{\infty}$')
    plt.plot(ts, regret_standards_avg, 'r--o', markevery=T//20,
             label=r'regret$_{s}$')
    plt.plot(ts, regret_news_avg, 'b-', label=r'regret$_{n}$')
    # plt.fill_between(ts, regret_news_avg-stokes, regret_news_avg+stokes,
    #                  facecolor='c', alpha=0.2, zorder=0, label='Stokes Bound')
    plt.plot(ts, np.zeros_like(ts), 'w-', lw=1)
    # plt.xlabel('Time Step')
    plt.ylabel('Aggregate System-Wide Loss')
    plt.xlim([0,T])
    # plt.ylim([-200,200])
    plt.legend(loc='lower right')
    plt.title('Demonstration of No-Regret on MLN')

    plt.savefig('NoRegret_MLN2c')

    # data = np.load('NoRegret2.npz')
    # distances_avg = data['d_avg']
    # loss_infs_avg = data['linf_avg']
    # regret_standards_avg = data['rs_avg']
    # regret_news_avg = data['rn_avg']
    # stokes = data['stokes']
    # ts = range(len(distances_avg))

    embed()


def infinity_loss(Domain,Start):
    # Set Method
    Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-3)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)],verbose=False)
    Repo = Reporting(Requests=[Domain.gap_rplus,'Data',Domain.F])
    Misc = Miscellaneous(Timer=False)
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Start Solver
    SOI_Results = Solve(Start,Method,Domain,Options)

    # Record X_Star
    Data = SOI_Results.PermStorage['Data']
    dx = np.diff(Data,axis=0)
    F = SOI_Results.PermStorage[Domain.F][:-1]

    return -np.sum(F*dx)


def integral(Domain,N=100):
    # crude approximation for now (Euler with constant step size)
    trange = np.linspace(0,1,N,endpoint=False)
    F = np.asarray([Domain.F(t) for t in trange])

    return np.sum(F)/N


def herons(p1,p2,p3):
    a,b,c = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p1-p3)
    s = 0.5*(a+b+c)
    return np.sqrt(s*(s-a)*(s-b)*(s-c))


if __name__ == '__main__':
    Demo()
