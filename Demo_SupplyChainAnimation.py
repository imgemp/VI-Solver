import time
import numpy as np

from VISolver.Domains.SupplyChain import (
    SupplyChain, CreateRandomNetwork)

from VISolver.Solvers.AdamsBashforthEuler import ABEuler

from VISolver.Projection import RPlusProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def Demo():

    #__SUPPLY_CHAIN__##################################################

    # Define Network and Domain
    Network = CreateRandomNetwork(I=2,Nm=2,Nd=2,Nr=1,seed=0)
    Domain = SupplyChain(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=RPlusProjection(),Delta0=1e-2)

    # Initialize Starting Point
    x = 10*np.ones(np.product(Domain.x_shape))
    gam = np.ones(np.sum([np.product(g) for g in Domain.gam_shapes]))
    lam = np.zeros(np.sum([np.product(l) for l in Domain.lam_shapes]))
    Start = np.concatenate((x,gam,lam))

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=25000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
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

    ########################################################
    # Animate Network
    ########################################################

    # Construct MP4 Writer
    fps = 15
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='SupplyChain', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Collect Frames
    frame_skip = 5
    freeze = 5
    Frames = SupplyChain_Results.PermStorage['Data'][::frame_skip]
    # Frames = np.concatenate((SupplyChain_Results_Phase1.PermStorage['Data'],
    #                          [SupplyChain_Results_Phase1.PermStorage['Data'][-1]]*fps*frame_skip*freeze,
    #                          SupplyChain_Results_Phase2.PermStorage['Data'],
    #                          [SupplyChain_Results_Phase2.PermStorage['Data'][-1]]*fps*frame_skip*freeze),
    #                         axis=0)[::frame_skip]

    # Normalize Colormap by Flow at each Network Level
    Domain.FlowNormalizeColormap(Frames,cm.rainbow)

    # Mark Annotations
    # t1 = 0
    # t2 = t1 + len(SupplyChain_Results_Phase1.PermStorage['Data']) // frame_skip
    # t3 = t2 + fps*freeze
    # t4 = t3 + len(SupplyChain_Results_Phase2.PermStorage['Data']) // frame_skip
    # anns = sorted([(t1, plt.title, 'Control Network\n(Equilibrating)'),
    #                (t2, plt.title, 'Control Network\n(Converged)'),
    #                (t3, plt.title, 'Market 1 Increases Demand for Service 1 by Provider 1\n(Equilibrating)'),
    #                (t4, plt.title, 'Market 1 Increases Demand for Service 1 by Provider 1\n(Converged)')],
    #               key=lambda x:x[0], reverse=True)
    anns = []

    # Save Animation to File
    fig, ax = plt.subplots()
    SupplyChain_ani = animation.FuncAnimation(fig, Domain.UpdateVisual, init_func=Domain.InitVisual,
                                             frames=len(Frames), fargs=(ax, Frames, anns), blit=True)
    SupplyChain_ani.save('Videos/SupplyChain.mp4', writer=writer)

if __name__ == '__main__':
    Demo()
