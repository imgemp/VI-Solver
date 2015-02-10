import time

# from Domains.DummyMARL import *
# from Domains.DummyMARL2 import *
# from VISolver.Domains.Tricky import *

from Solvers.MySolver import *

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


def Demo():
    # __DUMMY_MARL__##################################################

    # Define Domain
    Domain = MatchingPennies()
    # Domain = Tricky()

    # Set Method
    box = np.array([0., 1.])
    epsilon = np.array([-0.01, 0.01])

    # Method = IGA(Domain, P=BoxProjection())
    # Method = WoLFIGA(Domain, P=BoxProjection(), min_step=1e-4, max_step=1e-3)
    Method = MySolver(Domain, P=BoxProjection())

    # Initialize Starting Point
    # Start = np.array([0,1])
    Start = np.array([[.7, .3], [.6, .4]])

    # Set Options
    Init = Initialization(step=1e-4)
    # init = Initialization(Step=-0.1)
    Term = Termination(max_iter=100000, tols=[(Domain.ne_l2error, 1e-3)])
    Repo = Reporting(requests=[Domain.ne_l2error, 'Policy', 'Policy Learning Rate', 'Projections'])
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
    # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]  # Just take probabilities for first action
    print('Endpoint:')
    print(Data[-1])
    # print(Method.goodbad)
    fig, ax = plt.subplots(1, 1)

    # Choose a color map, loop through the colors, and assign them to the color
    # cycle. You need NPOINTS-1 colors, because you'll plot that many lines
    # between pairs. In other words, your line is not cyclic, so there's
    # no line from end to beginning
    # cm = plt.get_cmap('winter')
    # ax.set_color_cycle([cm(1.*i/(Data.shape[0]-1)) for i in xrange(Data.shape[0]-1)])
    # for i in xrange(Data.shape[0]-1):
    #     ax.plot(Data[i:i+2,0],Data[i:i+2,1])

    ax.plot(Data[:, 0], Data[:, 1])
    # ax.plot(value_function_values[:, 0], value_function_values[:, 1])
    ax.set_xlim(box + epsilon)
    ax.set_ylim(box + epsilon)
    plt.show()


if __name__ == '__main__':
    Demo()
