import time
import numpy as np

from VISolver.Domains.SupplyChain import SupplyChain, CreateRandomNetwork
from VISolver.Domains.AverageDomains import AverageDomains
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

    #__SUPPLY_CHAIN_NETWORK__###################################################
    N = 10  # number of possible maps
    T = 1000  # number of time steps
    eta = .01  # learning rate

    # Define Domains and Compute Equilbria
    Domains = []
    X_Stars = []
    for n in range(N):
        # Create Domain
        Network = CreateRandomNetwork(I=3,Nm=2,Nd=2,Nr=1,seed=n)
        Domain = SupplyChain(Network=Network,alpha=2)

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
        Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
        Repo = Reporting(Requests=[Domain.gap_rplus])
        Misc = Miscellaneous()
        Options = DescentOptions(Init,Term,Repo,Misc)

        # Print Stats
        PrintSimStats(Domain,Method,Options)

        # Start Solver
        tic = time.time()
        SupplyChain_Results = Solve(Start,Method,Domain,Options)
        toc = time.time() - tic

        # Print Results
        PrintSimResults(Options,SupplyChain_Results,Method,toc)

        # Record X_Star
        X_Star = SupplyChain_Results.TempStorage['Data'][-1]
        X_Stars += [X_Star]
    X_Stars = np.asarray(X_Stars)

    # Compute Equilibrium of Average Domain
    Domain = AverageDomains(Domains)

    # Set Method
    Method = HeunEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-3)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SupplyChain_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SupplyChain_Results,Method,toc)

    # Record X_Opt
    X_Opt = SupplyChain_Results.TempStorage['Data'][-1]

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
        # retrieve equilibrium / reference vector
        equi = X_Stars[idx]
        # calculate distance
        distances += [np.linalg.norm(equi-X)]
        # calculate infinity loss
        loss_infs += [infinity_loss(Domain,X)]
        # calculate standard regret
        ci_predict = ContourIntegral(Domain,LineContour(equi,X))
        predict_loss = integral(ci_predict)
        ci_opt = ContourIntegral(Domain,LineContour(equi,X_Opt))
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

    np.savez_compressed('NoRegret2.npz',d_avg=distances_avg,
                        linf_avg=loss_infs_avg,rs_avg=regret_standards_avg,
                        rn_avg=regret_news_avg)

    # plt.subplot(2, 1, 1)
    # plt.plot(ts, distances_avg, 'k',label='Average Distance')
    # plt.title('Demonstration of No-Regret on MLN')
    # plt.ylabel('Euclidean Distance')
    # plt.legend()

    plt.subplot(1, 1, 1)
    plt.plot(ts, loss_infs_avg, 'k--', label=r'loss$_{\infty}$')
    plt.plot(ts, regret_standards_avg, 'r--o', markevery=T//20,
             label=r'regret$_{s}$')
    plt.plot(ts, regret_news_avg, 'b-', label=r'regret$_{n}$')
    plt.plot(ts, np.zeros_like(ts), 'w-', lw=1)
    plt.xlabel('Time Step')
    plt.ylabel('Aggregate System-Wide Loss')
    plt.xlim([0,T])
    plt.ylim([-250,1000])
    plt.legend()
    plt.title('Demonstration of No-Regret on Supply Chain Network')

    plt.savefig('NoRegret2b')

    # data = np.load('NoRegret2b.npz')
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
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Data',Domain.F])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Start Solver
    SupplyChain_Results = Solve(Start,Method,Domain,Options)

    # Record X_Star
    Data = SupplyChain_Results.PermStorage['Data']
    dx = np.diff(Data,axis=0)
    F = SupplyChain_Results.PermStorage[Domain.F][:-1]

    return -np.sum(F*dx)


def integral(Domain,N=100):
    # crude approximation for now (Euler with constant step size)
    trange = np.linspace(0,1,N,endpoint=False)
    F = np.asarray([Domain.F(t) for t in trange])

    return np.sum(F)/N


if __name__ == '__main__':
    Demo()
