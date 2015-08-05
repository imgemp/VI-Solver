import time
import numpy as np

from VISolver.Domains.SOI import SOI, CreateNetworkExample

from VISolver.Solvers.AdamsBashforthEuler import ABEuler

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def Demo():

    #__BLOOD_BANK__##################################################

    #############################################################
    # Example 1 from Nagurney's Paper
    #############################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=1)
    Domain = SOI(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-2)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=10000,Tols=[(Domain.gap_rplus,1e-6*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SOI_Results_Phase1 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SOI_Results_Phase1,Method,toc)

    #########################################################
    # Example 2 from Nagurney's Paper
    #########################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = SOI(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = SOI_Results_Phase1.PermStorage['Data'][-1]

    # Calculate Initial Gap
    gap_0 = Domain.gap_rplus(Start)

    # Set Options
    Init = Initialization(Step=-1e-10)
    Term = Termination(MaxIter=10000,Tols=[(Domain.gap_rplus,1e-3*gap_0)])
    Repo = Reporting(Requests=[Domain.gap_rplus, 'Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    SOI_Results_Phase2 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SOI_Results_Phase2,Method,toc)

    ########################################################
    # Animate Network
    ########################################################

    # Construct MP4 Writer
    fps = 15
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='SOI', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Collect Frames
    frame_skip = 5
    freeze = 5
    Frames = np.concatenate((SOI_Results_Phase1.PermStorage['Data'],
                             [SOI_Results_Phase1.PermStorage['Data'][-1]]*fps*frame_skip*freeze,
                             SOI_Results_Phase2.PermStorage['Data'],
                             [SOI_Results_Phase2.PermStorage['Data'][-1]]*fps*frame_skip*freeze),
                            axis=0)[::frame_skip]

    # Normalize Colormap by Flow at each Network Level
    Domain.FlowNormalizeColormap(Frames,cm.rainbow)

    # Mark Annotations
    t1 = 0
    t2 = t1 + len(SOI_Results_Phase1.PermStorage['Data']) // frame_skip
    t3 = t2 + fps*freeze
    t4 = t3 + len(SOI_Results_Phase2.PermStorage['Data']) // frame_skip
    anns = sorted([(t1, plt.title, 'Control Network\n(Equilibrating)'),
                   (t2, plt.title, 'Control Network\n(Converged)'),
                   (t3, plt.title, 'Market 1 Increases Demand for Service 1 by Provider 1\n(Equilibrating)'),
                   (t4, plt.title, 'Market 1 Increases Demand for Service 1 by Provider 1\n(Converged)')],
                  key=lambda x:x[0], reverse=True)

    # Save Animation to File
    fig, ax = plt.subplots()
    SOI_ani = animation.FuncAnimation(fig, Domain.UpdateVisual, init_func=Domain.InitVisual,
                                             frames=len(Frames), fargs=(ax, Frames, anns), blit=True)
    SOI_ani.save('Videos/SOI.mp4', writer=writer)

if __name__ == '__main__':
    Demo()
