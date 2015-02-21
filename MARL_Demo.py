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
    # Method = MyIGA(Domain, P=BoxProjection())
    method = MyIGA(Domain, P=LinearProjection())

    # Initialize Starting Point
    # Start = np.array([0,1])
    start_strategies = np.array([[.7, .3], [.6, .4]])

    # Set Options
    initialization_conditions = Initialization(step=1e-4)
    # init = Initialization(Step=-0.1)
    terminal_conditions = Termination(max_iter=200, tols=[(Domain.ne_l2error, 1e-3)])
    reporting_options = Reporting(requests=[Domain.ne_l2error,
                                            'Policy',
                                            'Policy Gradient (dPi)',
                                            'Policy Learning Rate',
                                            'Value Function',
                                            'Value Variance'])
    whatever_this_does = Miscellaneous()
    options = DescentOptions(initialization_conditions, terminal_conditions, reporting_options, whatever_this_does)

    # Print Stats
    # print_sim_stats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    marl_results = solve(start_strategies, method, Domain, options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(options, marl_results, method, toc)

    policy = np.array(marl_results.perm_storage['Policy'])[:, :, 0]  # Just take probabilities for first action
    val_fun = np.array(marl_results.perm_storage['Value Function'])  # Just take probabilities for first action
    val_var = np.array(marl_results.perm_storage['Value Variance'])  # Just take probabilities for first action
    pol_grad = np.array(marl_results.perm_storage['Policy Gradient (dPi)'])
    # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]
    print('Endpoint:')
    print(policy[-1])

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

        ax[0].plot(policy[:, 0], policy[:, 1])
        ax[0].set_xlim(box + epsilon)
        ax[0].set_ylim(box + epsilon)
        ax[0].set_title('The policy-policy space')
        ax[1].plot(policy[:, 0])
        ax[1].set_title('Policy of player 1')
        # ax[1].set_ylim(box + epsilon)
        ax[2].plot(policy[:, 1])
        ax[2].set_title('Policy of player 2')
        # ax[2].set_ylim(box + epsilon)
        ax[3].plot(val_fun)
        ax[3].set_title('The policy gradient')
        ax[4].plot(pol_grad)
        plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    (options, args) = parser.parse_args()
    demo(options.plot)
