import time
import numpy as np

from VISolver.Domains.SOI import SOI, CreateRandomNetwork
from VISolver.Domains.ContourIntegral import ContourIntegral, LineContour

from VISolver.Solvers.HeunEuler import HeunEuler

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
    eta = .01  # learning rate
    Ot = 0  # reference vector will be origin for all maps

    # Define Domains and Compute Equilbria
    Domains = []
    X_Stars = []
    for n in range(N):
        # Create Domain
        Network = CreateRandomNetwork(m=3,n=2,o=2,seed=n)
        Domain = SOI(Network=Network,alpha=2)

        # Record Domain
        Domains += [Domain]

        # Set Method
        Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-3)

        # Initialize Starting Point
        Start = np.zeros(Domain.Dim)

        # Calculate Initial Gap
        gap_0 = Domain.gap_rplus(Start)

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
    X_Stars = np.asarray(X_Stars)
    X_Opt = np.mean(X_Stars,axis=0)
    Ot = Ot*np.ones(X_Stars.shape[1])

    print('Starting Online Learning')

    # Set First Prediction
    X = np.zeros(X_Stars.shape[1])

    # Select First Domain
    idx = np.argmax(np.linalg.norm(X_Stars - X,axis=1))

    distances = []
    loss_infs = []
    regret_standards = []
    regret_news = []
    ts = range(T)
    for t in ts:
        print('t = '+str(t))
        # retrieve domain
        Domain = Domains[idx]
        # retrieve equilibrium
        equi = X_Stars[idx]
        # calculate distance
        distances += [np.linalg.norm(equi-X)]
        # calculate infinity loss
        loss_infs += [infinity_loss(Domain,X)]
        # calculate standard regret
        ci_predict = ContourIntegral(Domain,LineContour(Ot,X))
        predict_loss = integral(ci_predict)
        ci_opt = ContourIntegral(Domain,LineContour(Ot,X_Opt))
        predict_opt = integral(ci_opt)
        regret_standards += [predict_loss - predict_opt]
        # calculate new regret
        ci_new = ContourIntegral(Domain,LineContour(X_Opt,X))
        regret_news += [integral(ci_new)]
        # update prediction
        X = BoxProjection(lo=0).P(X,-eta,Domain.F(X))
        # update domain
        idx = np.argmax(np.linalg.norm(X_Stars - X,axis=1))
    ts_p1 = range(1,T+1)
    distances_avg = np.divide(distances,ts_p1)
    loss_infs_avg = np.divide(loss_infs,ts_p1)
    regret_standards_avg = np.divide(regret_standards,ts_p1)
    regret_news_avg = np.divide(regret_news,ts_p1)

    np.savez_compressed('NoRegret.npz',d_avg=distances_avg,
                        linf_avg=loss_infs_avg,rs_avg=regret_standards_avg,
                        rn_avg=regret_news_avg)

    plt.subplot(2, 1, 1)
    plt.plot(ts, distances_avg, 'k',label='Average Distance')
    plt.title('Demonstration of No-Regret on MLN')
    plt.ylabel('Euclidean Distance')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ts, loss_infs_avg, 'k--', label=r'loss$_{\infty}$')
    plt.plot(ts, regret_standards_avg, 'r--o', markevery=T//20,
             label=r'regret$_{s}$')
    plt.plot(ts, regret_news_avg, 'b-', label=r'regret$_{n}$')
    plt.xlabel('Time Step')
    plt.ylabel('Aggregate System-Wide Loss')
    plt.xlim([0,T])
    plt.ylim([-500,5000])
    plt.legend()

    plt.savefig('NoRegret')

    # data = np.load('NoRegret.npz')
    # distances_avg = data['d_avg']
    # loss_infs_avg = data['linf_avg']
    # regret_standards_avg = data['rs_avg']
    # regret_news_avg = data['rn_avg']
    # ts = range(len(distances_avg))


def infinity_loss(Domain,Start):
    # Set Method
    Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-3)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Data',Domain.F])
    Misc = Miscellaneous()
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


if __name__ == '__main__':
    Demo()
