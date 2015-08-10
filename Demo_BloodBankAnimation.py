import time
import numpy as np

from VISolver.Domains.BloodBank import BloodBank, CreateNetworkExample

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
    # Phase 1 - Left hospital has relatively low demand for blood
    #############################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=1)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = np.zeros(Domain.Dim)

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
    BloodBank_Results_Phase1 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results_Phase1,Method,toc)

    #########################################################
    # Phase 2 - Left hospital has comparable demand for blood
    #########################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=2)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = BloodBank_Results_Phase1.PermStorage['Data'][-1]

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
    BloodBank_Results_Phase2 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results_Phase2,Method,toc)

    ########################################################
    # Phase 3 - Left hospital has excessive demand for blood
    ########################################################

    # Define Network and Domain
    Network = CreateNetworkExample(ex=3)
    Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    Method = ABEuler(Domain=Domain,P=BoxProjection(lo=0),Delta0=1e-5)

    # Initialize Starting Point
    Start = BloodBank_Results_Phase2.PermStorage['Data'][-1]

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
    BloodBank_Results_Phase3 = Solve(Start,Method,Domain,Options)
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,BloodBank_Results_Phase3,Method,toc)

    ########################################################
    # Animate Network
    ########################################################

    # Construct MP4 Writer
    fps = 15
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='BloodBank', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Collect Frames
    frame_skip = 5
    freeze = 5
    Dyn_1 = BloodBank_Results_Phase1.PermStorage['Data']
    Frz_1 = [Dyn_1[-1]]*fps*frame_skip*freeze
    Dyn_2 = BloodBank_Results_Phase2.PermStorage['Data']
    Frz_2 = [Dyn_2[-1]]*fps*frame_skip*freeze
    Dyn_3 = BloodBank_Results_Phase3.PermStorage['Data']
    Frz_3 = [Dyn_3[-1]]*fps*frame_skip*freeze
    Frames = np.concatenate((Dyn_1,Frz_1,Dyn_2,Frz_2,Dyn_3,Frz_3),
                            axis=0)[::frame_skip]

    # Normalize Colormap by Flow at each Network Level
    Domain.FlowNormalizeColormap(Frames,cm.Reds)

    # Mark Annotations
    t1 = 0
    t2 = t1 + len(BloodBank_Results_Phase1.PermStorage['Data']) // frame_skip
    t3 = t2 + fps*freeze
    t4 = t3 + len(BloodBank_Results_Phase2.PermStorage['Data']) // frame_skip
    t5 = t4 + fps*freeze
    t6 = t5 + len(BloodBank_Results_Phase3.PermStorage['Data']) // frame_skip
    Dyn_1_ann = '$Demand_{1}$ $<$ $Demand_{2}$ & $Demand_{3}$\n(Equilibrating)'
    Frz_1_ann = '$Demand_{1}$ $<$ $Demand_{2}$ & $Demand_{3}$\n(Converged)'
    Dyn_2_ann = '$Demand_{1}$ ~= $Demand_{2}$ & $Demand_{3}$\n(Equilibrating)'
    Frz_2_ann = '$Demand_{1}$ ~= $Demand_{2}$ & $Demand_{3}$\n(Converged)'
    Dyn_3_ann = '$Demand_{1}$ $>$ $Demand_{2}$ & $Demand_{3}$\n(Equilibrating)'
    Frz_3_ann = '$Demand_{1}$ $>$ $Demand_{2}$ & $Demand_{3}$\n(Converged)'
    anns = sorted([(t1, plt.title, Dyn_1_ann),
                   (t2, plt.title, Frz_1_ann),
                   (t3, plt.title, Dyn_2_ann),
                   (t4, plt.title, Frz_2_ann),
                   (t5, plt.title, Dyn_3_ann),
                   (t6, plt.title, Frz_3_ann)],
                  key=lambda x:x[0], reverse=True)

    # Save Animation to File
    fig, ax = plt.subplots()
    BloodBank_ani = animation.FuncAnimation(fig, Domain.UpdateVisual,
                                            init_func=Domain.InitVisual,
                                            frames=len(Frames),
                                            fargs=(ax, Frames, anns), blit=True)
    BloodBank_ani.save('BloodBank.mp4', writer=writer)

if __name__ == '__main__':
    Demo()
