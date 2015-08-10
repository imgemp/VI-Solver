import time
import numpy as np

from VISolver.Domains.SupplyChain import (
    SupplyChain, CreateNetworkExample)

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

    #__SUPPLY_CHAIN__##################################################

    #############################################################
    # Example 1 from Nagurney's Paper
    #############################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=1)
    Domain = SupplyChain(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-2)

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
    SupplyChain_Results_Phase1 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SupplyChain_Results_Phase1,Method,toc)

    #############################################################
    # Increased Demand of Firm 2's Product
    #############################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = SupplyChain(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-2)

    # Initialize Starting Point
    Start = SupplyChain_Results_Phase1.PermStorage['Data'][-1]

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
    SupplyChain_Results_Phase2 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,SupplyChain_Results_Phase2,Method,toc)

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
    Dyn_1 = SupplyChain_Results_Phase1.PermStorage['Data']
    Frz_1 = [Dyn_1[-1]]*fps*frame_skip*freeze
    Dyn_2 = SupplyChain_Results_Phase2.PermStorage['Data']
    Frz_2 = [Dyn_2[-1]]*fps*frame_skip*freeze
    Frames = np.concatenate((Dyn_1,Frz_1,Dyn_2,Frz_2),axis=0)[::frame_skip]

    # Normalize Colormap by Flow at each Network Level
    Domain.FlowNormalizeColormap(Frames,cm.rainbow)

    # Mark Annotations
    t1 = 0
    t2 = t1 + len(SupplyChain_Results_Phase1.PermStorage['Data']) // frame_skip
    t3 = t2 + fps*freeze
    t4 = t3 + len(SupplyChain_Results_Phase2.PermStorage['Data']) // frame_skip
    Dyn_1_ann = 'Control Network\n(Equilibrating)'
    Frz_1_ann = 'Control Network\n(Converged)'
    Dyn_2_ann = 'Market Increases Demand for Firm 2''s Product\n(Equilibrating)'
    Frz_2_ann = 'Market Increases Demand for Firm 2''s Product\n(Converged)'
    anns = sorted([(t1, plt.title, Dyn_1_ann),
                   (t2, plt.title, Frz_1_ann),
                   (t3, plt.title, Dyn_2_ann),
                   (t4, plt.title, Frz_2_ann)],
                  key=lambda x:x[0], reverse=True)

    # Save Animation to File
    fig, ax = plt.subplots()
    SupplyChain_ani = animation.FuncAnimation(fig, Domain.UpdateVisual,
                                              init_func=Domain.InitVisual,
                                              frames=len(Frames),
                                              fargs=(ax, Frames, anns),
                                              blit=True)
    SupplyChain_ani.save('Videos/SupplyChain.mp4', writer=writer)

if __name__ == '__main__':
    Demo()
