import time
import numpy as np

from VISolver.Domains.CloudServices3 import CloudServices, CreateRandomNetwork

# from VISolver.Solvers.Euler import Euler
# from VISolver.Solvers.Extragradient import EG
# from VISolver.Solvers.AcceleratedGradient import AG
# from VISolver.Solvers.HeunEuler import HeunEuler
from VISolver.Solvers.AdamsBashforthEuler import ABEuler
from VISolver.Solvers.CashKarp import CashKarp

from VISolver.Projection import RPlusProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

from matplotlib import pyplot as plt

from IPython import embed


def Demo():

    #__CLOUD_SERVICES__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(5,20)
    Domain = CloudServices(Network=Network,gap_alpha=2)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-8)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = 2*np.ones(Domain.Dim)

    # Calculate Initial Gap
    # print('comp gap')
    gap_0 = Domain.gap_rplus(Start)
    # print('done')
    print(Domain.CloudProfits(Start))
    print(Domain.dCloudProfits(Start))
    # print(Domain.maxfirm_profits(Start))
    # print(Domain.argmax_firm_profits(Start))

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-0.00001)
    Term = Termination(MaxIter=15e4)  # ,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus,'Step','F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # embed()

    # Start Solver
    tic = time.time()
    CloudServices_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,CloudServices_Results,Method,toc)

    x = CloudServices_Results.PermStorage['Data'][-1]
    print('[p...q]')
    print(x)
    print('Qj')
    print(Domain.Demand(x))
    print('Profits')
    print(Domain.CloudProfits(x))
    print('[dpj...dqj')
    print(Domain.dCloudProfits(x))
    print('Qij')
    print(Domain.Demand_IJ(x))
    # print(Domain.maxfirm_profits(x))
    # print(Domain.argmax_firm_profits(x))

    # print('Testing New Domain')
    # Network = CreateRandomNetwork(nClouds=2)
    # Domain = CloudServices(Network=Network,alpha=2)

    # X,Y = np.meshgrid(np.arange(1,3,.2),np.arange(1,3,.2))
    # U,V = np.empty_like(X), np.empty_like(Y)
    # for i in xrange(X.shape[0]):
    #     for j in xrange(Y.shape[1]):
    #         print((i,j))
    #         prices = np.array([X[i,j],Y[i,j],1.,1.])
    #         grad = Domain.dCloudProfits(prices)
    #         U[i,j], V[i,j] = grad[0], grad[1]
    #         print(Domain.argmax_firm_profits(prices))
    # plt.figure()
    # Q = plt.quiver(X[::3, ::3],Y[::3, ::3],U[::3, ::3],V[::3, ::3],
    #                pivot='mid', color='r', units='inches')
    # qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
    #                    fontproperties={'weight': 'bold'})
    # plt.plot(X[::3, ::3],Y[::3, ::3],'k.')
    # plt.axis([-1, 4, -1, 4])
    # plt.title("pivot='mid'; every third arrow; units='inches'")
    # plt.show()

    print(Domain.c_clouds)
    print(Domain.pref_bizes)

    data = CloudServices_Results.PermStorage['Data']
    eigs = np.zeros((len(data),2))
    eig_test = np.zeros(len(data))
    for i in xrange(len(data)):
        hess = -Domain.Hess(data[i])
        eigs_r = np.real(np.linalg.eigvals(hess))
        eigs[i,:] = [min(eigs_r),max(eigs_r)]

        temp = np.diag(hess) - np.sum(np.abs(hess),axis=1) +\
            np.abs(np.diag(hess))
        eig_test[i] = min(temp)

    plt.figure(1)
    plt.plot(CloudServices_Results.PermStorage['Data'])
    plt.figure(2)
    plt.plot(eigs)
    plt.show()

    embed()

if __name__ == '__main__':
    Demo()
