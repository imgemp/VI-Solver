import time
import numpy as np

from VISolver.Domains.MLN import MLN, CreateRandomNetwork

from VISolver.Solvers.Euler import Euler

from VISolver.Projection import BoxProjection
from VISolver.Solver import Solve
from VISolver.Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import PrintSimResults, PrintSimStats

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from IPython import embed


def Demo():

    # __MACHINE_LEARNING_NETWORK____#################################################

    # Define Domain
    Network = CreateRandomNetwork()
    Domain = MLN(Network)

    # Set Method
    Method = Euler(Domain=Domain,FixStep=True,P=BoxProjection(lo=Domain.los,hi=Domain.his))

    # Set Options
    Term = Termination(MaxIter=500)
    Repo = Reporting(Requests=['Step', 'F Evaluations',
                               'Projections','Data'])
    Misc = Miscellaneous()
    Init = Initialization(Step=-1e-3)
    Options = DescentOptions(Init,Term,Repo,Misc)

    # Initialize Starting Point
    Start = np.random.rand(Domain.dim)*(Domain.his-Domain.los) + Domain.los
    Start[20:22] = [-.5,.5]
    Start[22:24] = [.5,.5]
    Start[24:26] = [.5,.5]
    Start[26:28] = [1.,0.]
    Start[28:32] = [-.5,-.25,-.75,-.5]

    # Print Stats
    PrintSimStats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    MLN_Results = Solve(Start,Method,Domain,Options)  # Use same Start
    toc = time.time() - tic

    # Print Results
    PrintSimResults(Options,MLN_Results,Method,toc)

    ########################################################
    # Animate Network
    ########################################################

    # Construct MP4 Writer
    fps = 5
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='MLN', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Collect Frames
    Frames = MLN_Results.PermStorage['Data'][::fps]

    # Save Animation to File
    fig, ax = plt.subplots()
    MLN_ani = animation.FuncAnimation(fig, Domain.UpdateVisual,
                                      init_func=Domain.InitVisual,
                                      frames=len(Frames),
                                      fargs=(ax, Frames))
    MLN_ani.save('MLN.mp4', writer=writer)


if __name__ == '__main__':
    Demo()