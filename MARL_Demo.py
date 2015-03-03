import time
import config
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


def smooth_1d_sequence(sequence, sigma=15):
    # smoothing functions for more readable plotting
    from scipy.ndimage import gaussian_filter1d
    sequence = np.array(sequence)
    assert len(sequence.shape) <= 2, 'Cannot interpret an array with more than 2 dimensions as a tuple of 1d sequences.'
    # asserting that the data is in the rows and that the array has a second dimension (for the for loop)
    if max(sequence.shape) > min(sequence.shape):
        if sequence.shape[1] > sequence.shape[0]:
            sequence = sequence.T
    else:
        sequence = sequence[None]
    for i in range(sequence.shape[1]):
        val_interpol = np.interp(range(sequence.shape[0]), range(sequence.shape[0]), sequence[:, i])
        sequence[:, i] = gaussian_filter1d(val_interpol, sigma)
    return sequence


def demo():
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
    rand_start_one = np.random.random()
    rand_start_two = np.random.random()
    # start_strategies = np.array([[.5, .5], [.5, .5]])
    # start_strategies = np.array([[.51, .49], [.49, .51]])
    start_strategies = np.array([[rand_start_one, 1-rand_start_one], [rand_start_two, 1-rand_start_two]])

    # Set Options
    initialization_conditions = Initialization(step=1e-4)
    # init = Initialization(Step=-0.1)
    terminal_conditions = Termination(max_iter=1000, tols=[(Domain.ne_l2error, 1e-3)])
    reporting_options = Reporting(requests=[Domain.ne_l2error,
                                            'Policy',
                                            'Policy Gradient (dPi)',
                                            'Policy Learning Rate',
                                            'Reward',
                                            'Value Function',
                                            'True Value Function',
                                            'Value Variance',
                                            'Performance',
                                            'Am I winning?'])
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
    val_fun = np.array(marl_results.perm_storage['Value Function'])
    performance = np.array(marl_results.perm_storage['Performance'])
    true_val_fun = np.array(marl_results.perm_storage['True Value Function'])
    # val_var = np.array(marl_results.perm_storage['Value Variance'])
    pol_grad = np.array(marl_results.perm_storage['Policy Gradient (dPi)'])
    pol_lr = np.array(marl_results.perm_storage['Policy Learning Rate'])
    win_val = np.array(marl_results.perm_storage['Am I winning?'])
    # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]
    print('Percentage of games won:')
    print np.mean(.5 + np.array(marl_results.perm_storage['Reward']), axis=0)
    print('Endpoint:')
    print(policy[-1])
    print('The policy gradient histrogram:')
    a, b = np.histogram(pol_grad)
    pol_grad_hist = [[a[i], b[i]] for i in range(len(a))]
    for i in range(len(pol_grad_hist)):
        print '  ', pol_grad_hist[i][1], ': ', pol_grad_hist[i][0]
    # print(np.histogram(pol_grad))

    # creating plots, if desired:
    if config.show_plots:
        fig, ax = plt.subplots(3, 2)

        # ax[0, 1].plot(policy[:, 0], policy[:, 1])
        # ax[0, 1].set_xlim(box + epsilon)
        # ax[0, 1].set_ylim(box + epsilon)
        # ax[0, 1].set_title('The policy-policy space')
        # ax[0, 1].grid(True)

        ax[0, 1].plot(smooth_1d_sequence(performance))
        # ax[0, 1].plot(performance)
        ax[0, 1].set_title('Performance')
        ax[0, 1].grid(True)

        ax[1, 0].plot(true_val_fun)
        ax[1, 0].set_title('True Value Function')
        ax[1, 0].grid(True)

        ax[2, 0].plot(policy)
        ax[2, 0].set_title('The policies')
        ax[2, 0].set_ylim(box + epsilon)
        ax[2, 0].grid(True)

        ax[0, 0].plot(smooth_1d_sequence(val_fun, 4))
        # ax[0, 0].plot(val_fun)
        # ax[0, 0].set_title('The value function')
        ax[0, 0].set_title('Am I winning?')
        ax[0, 0].grid(True)

        ax[1, 1].plot(smooth_1d_sequence(pol_grad))
        # ax[1, 1].plot(pol_grad)
        ax[1, 1].set_title('The policy gradient')
        ax[1, 1].grid(True)

        ax[2, 1].plot(smooth_1d_sequence(pol_lr))
        # ax[2, 1].plot(pol_lr)
        ax[2, 1].set_title('The learning rate')
        ax[2, 1].grid(True)
        plt.show()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    parser.add_option('-v', '--show-output', action='store_false', dest='debug', default=True, help='show output')
    (options, args) = parser.parse_args()
    config.show_debug_output = options.debug
    config.show_plots = options.plot
    demo()
