import time
from optparse import OptionParser

# from Domains.DummyMARL import *
# from Domains.DummyMARL2 import *
# from VISolver.Domains.Tricky import *

from Solvers.MySolver import *
from Solvers.MyIGA import *
# from Solvers.IGA.IGA import *

from VISolver.Solvers.Solver import solve
from Log import *
from Domains.MatchingPennies import MatchingPennies

# from Solvers.Euler import *
# from Solvers.Extragradient import *
# from Solvers.AcceleratedGradient import *
# from Solvers.HeunEuler import *
# from Solvers.AdamsBashforthEuler import *
# from Solvers.CashKarp import *
# from Solvers.GABE import *
# from Solvers.Drift import *
# from Solvers.DriftABE import *
# from Solvers.DriftABE_Exact import *
# from Solvers.DriftABE_BothExact import *
# from Solvers.DriftABE_VIteration import *
# from Solvers.DriftABE2 import *
# from Solvers.DriftABE3 import *
# from Solvers.DriftABE4 import *

from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import print_sim_results

import matplotlib.pyplot as plt


def demo(plotting):
    # __DUMMY_MARL__##################################################

    # Define Domain
    Domain = MatchingPennies()
    # Domain = Tricky()

    # Set Method
    box = np.array([0., 1.])
    epsilon = np.array([-0.01, 0.01])

    # Method = IGA(Domain, P=BoxProjection())
    # Method = WoLFIGA(Domain, P=BoxProjection(), min_step=1e-4, max_step=1e-3)
    # Method = MySolver(Domain, P=BoxProjection())
    Method = MyIGA(Domain, P=BoxProjection())

    # Initialize Starting Point
    # Start = np.array([0,1])
    Start = np.array([[.7, .3], [.6, .4]])

    # Set Options
    Init = Initialization(step=1e-4)
    # init = Initialization(Step=-0.1)
    Term = Termination(max_iter=20000, tols=[(Domain.ne_l2error, 1e-3)])
    Repo = Reporting(requests=[Domain.ne_l2error,
                               'Policy',
                               'Policy Learning Rate',
                               'Projections',
                               'Value Function',
                               'Value Variance'])
    Misc = Miscellaneous()
    Options = DescentOptions(Init, Term, Repo, Misc)

    # Print Stats
    # print_sim_stats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    MARL_Results = solve(Start, Method, Domain, Options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(Options, MARL_Results, Method, toc)

    Data = np.array(MARL_Results.perm_storage['Policy'])[:, :, 0]  # Just take probabilities for first action
    val_fun = np.array(MARL_Results.perm_storage['Value Function'])  # Just take probabilities for first action
    val_var = np.array(MARL_Results.perm_storage['Value Variance'])  # Just take probabilities for first action
    # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]
    print('Endpoint:')
    print(Data[-1])

    # creating plots, if desired:
    if plotting:
        fig, ax = plt.subplots(5)

        # Choose a color map, loop through the colors, and assign them to the color
        # cycle. You need NPOINTS-1 colors, because you'll plot that many lines
        # between pairs. In other words, your line is not cyclic, so there's
        # no line from end to beginning
        # cm = plt.get_cmap('winter')
        # ax.set_color_cycle([cm(1.*i/(Data.shape[0]-1)) for i in xrange(Data.shape[0]-1)])
        # for i in xrange(Data.shape[0]-1):
        #     ax.plot(Data[i:i+2,0],Data[i:i+2,1])

        ax[0].plot(Data[:, 0], Data[:, 1])
        ax[0].set_xlim(box + epsilon)
        ax[0].set_ylim(box + epsilon)
        ax[1].plot(Data[:, 0])
        # ax[1].set_ylim(box + epsilon)
        ax[2].plot(Data[:, 1])
        # ax[2].set_ylim(box + epsilon)
        ax[3].plot(val_fun)
        ax[4].plot(val_var)
        plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    (options, args) = parser.parse_args()
    demo(options.plot)
