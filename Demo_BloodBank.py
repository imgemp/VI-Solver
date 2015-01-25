import time
import numpy as np

from VISolver.Domains.BloodBank import BloodBank, CreateRandomNetwork, CreateNetworkExample1

from VISolver.Solvers.Euler import Euler
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation


def Demo():

    #__BLOOD_BANK__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(nC=6,nB=4,nD=3,nR=7,seed=0)
    # Network = CreateRandomNetwork(nC=4,nB=2,nD=3,nR=4,seed=0)
    # Network = CreateNetworkExample1()
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    # Method = Euler(Domain=Domain,P=RPlusProjection())
    # Method = EG(Domain=Domain,P=RPlusProjection())
    # Method = AG(Domain=Domain,P=RPlusProjection())
    # Method = HeunEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-5)
    # Method = CashKarp(Domain=Domain,P=RPlusProjection(),Delta0=1e-6)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)
    # Start = np.random.rand(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    # Init = Initialization(Step=-0.1)
    Term = Termination(MaxIter=1000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    BloodBank_Results = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results,Method,toc)

    # Animate Network
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
            comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)


    plt.ion()
    fig, ax = plt.subplots()
    # fig = plt.figure(frameon=False)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    mxRed = 67.#np.max(BloodBank_Results.PermStorage['Data']); print(mxRed)
    data0 = BloodBank_Results.PermStorage['Data'][0]
    Domain.BuildSkeleton(ax,data0,mxRed)
    plt.show()
    with writer.saving(fig, "writer_test.mp4", 100):
        # for data in BloodBank_Results.PermStorage['Data']:
        for frame in xrange(0,len(BloodBank_Results.PermStorage['Data']),len(BloodBank_Results.PermStorage['Data'])//75):
            data = BloodBank_Results.PermStorage['Data'][frame]
            Domain.UpdateFlowColors(ax,data,mxRed)
            time.sleep(.01)
            plt.draw()
            writer.grab_frame()
            # print('next')

if __name__ == '__main__':
    Demo()