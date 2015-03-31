import time
from optparse import OptionParser

# from Domains.DummyMARL import *
# from Domains.DummyMARL2 import *

from VISolver.Solvers.Solver import solve
from Domains.MatchingPennies import MatchingPennies
from Domains.Tricky import Tricky
from Domains.PrisonersDilemma import PrisonersDilemma
from Domains.BattleOfTheSexes import BattleOfTheSexes

from Domains.BloodBank import BloodBank, CreateRandomNetwork

from Solvers.MARL_prior.WPL import *
from Solvers.MARL_prior.WoLFIGA import *
from Solvers.MARL_prior.AWESOME import *
from Solvers.MARL_prior.PGA_APP import *
from Solvers.MultiAgentVI import MultiAgentVI

from Options import (
    DescentOptions, Miscellaneous, Reporting, Termination, Initialization)
from VISolver.Log import print_sim_results

from Helpers import *


def demo():
    # __DUMMY_MARL__##################################################

    # Define Domain
    Domain = MatchingPennies()
    # Domain = BattleOfTheSexes('traditional')
    # Domain = BattleOfTheSexes('new')
    # Domain = Tricky()
    # Domain = PrisonersDilemma()
    # Network = CreateRandomNetwork(nC=2,nB=2,nD=2,nR=2,seed=0)
    # Domain = BloodBank(Network=Network,alpha=2)

    # Set Method
    # box = np.array(Domain.reward_range)
    box = np.array([-1, 1])
    epsilon = np.array([-0.01, 0.01])

    # method = IGA(Domain, P=BoxProjection())
    # method = WoLFIGA(Domain, P=BoxProjection(), min_step=1e-4, max_step=1e-3 )
    # method = MySolver(Domain, P=BoxProjection())
    # method = MyIGA(Domain, P=BoxProjection())
    # method = MyIGA(Domain, P=LinearProjection())
    # method = WPL(Domain, P=LinearProjection())
    # method = WPL(Domain, P=BoxProjection(low=.001))
    # method = AWESOME(Domain, P=LinearProjection())
    # method = PGA_APP(Domain, P=LinearProjection())
    method = MultiAgentVI(Domain, P=LinearProjection)

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
    terminal_conditions = Termination(max_iter=500, tols=[(Domain.ne_l2error, 1e-3)])
    reporting_options = method.reporting_options()
    whatever_this_does = Miscellaneous()
    options = DescentOptions(initialization_conditions, terminal_conditions, reporting_options, whatever_this_does)

    # Print Stats
    # print_sim_stats(Domain,Method,Options)

    # Start Solver
    tic = time.time()
    # set random seed
    # np.random.seed(0)
    marl_results = solve(start_strategies, method, Domain, options)
    toc = time.time() - tic

    # Print Results
    print_sim_results(options, marl_results, method, toc)

    # creating plots, if desired:
    if config.show_plots:
        policy = np.array(marl_results.perm_storage['Policy'])[:, :, 0]  # Just take probabilities for first action
        # policy_est = np.array(marl_results.perm_storage['Policy Estimates'])
        val_fun = np.array(marl_results.perm_storage['Value Function'])[-1]
        true_val_fun = np.array(marl_results.perm_storage['True Value Function'])
        # val_var = np.array(marl_results.perm_storage['Value Variance'])
        # pol_grad = np.array([[marl_results.perm_storage['Policy Gradient (dPi)'][i][0, 0],
        #                       marl_results.perm_storage['Policy Gradient (dPi)'][i][1, 0]]
        #                      for i in range(1, len(marl_results.perm_storage['Policy Gradient (dPi)']))])
        pol_lr = np.array(marl_results.perm_storage['Policy Learning Rate'])
        # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]
        print('Ratio of games won:')
        win_ratio = np.mean(.5 + .5*np.array(marl_results.perm_storage['Reward']), axis=0).round(2)
        print 'Player 1:', win_ratio[0], 'Player 2:', win_ratio[1]
        print('Endpoint:')
        print(policy[-1])
        print 'The value function'


        # ax[0, 1].plot(policy[:, 0], policy[:, 1])
        # ax[0, 1].set_xlim(box + epsilon)
        # ax[0, 1].set_ylim(box + epsilon)
        # ax[0, 1].set_title('The policy-policy space')
        # ax[0, 1].grid(True)

        printing_data = {}
        printing_data['The value function'] = {'values': val_fun.T, 'smooth':-1}
        printing_data['Analytic Value of policies played'] = {'values': true_val_fun, 'yLimits': box, 'smooth':-1}
        printing_data['The policies'] = {'values': policy, 'yLimits': np.array([0, 1]) + epsilon, 'smooth':-1}
        # printing_data['The policy gradient'] = {'values': pol_grad}
        # printing_data['Policy Estimates'] = {'values': policy_est, 'yLimits': box}
        plot_results(printing_data)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    parser.add_option('-v', '--show-output', type='int', dest='debug', default=0, help='debug level')
    (options, args) = parser.parse_args()
    config.debug_output_level = options.debug
    # config.debug_output_level = False
    config.show_plots = options.plot
    demo()
