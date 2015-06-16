import time
from optparse import OptionParser

# from Domains.DummyMARL import *
# from Domains.DummyMARL2 import *

from VISolver.Solvers.Solver import solve
from Domains.Stochastic import BattleOfTheSexes, MatchingPennies, PureStrategyTest, Tricky

from VISolver.Solvers.MARL_prior.WPL import *
from VISolver.Solvers.MARL_prior.PGA_APP import *
from VISolver.Solvers.LEAP import LEAP
from VISolver.Solvers.MAHT_WPL import MAHT_WPL

from VISolver.Options import (
    DescentOptions, Miscellaneous, Termination, Initialization)
from VISolver.Log import print_sim_results
from VISolver.Utilities import *
import VISolver.config as config


def demo(domain, method, iterations=500, start_strategies=None):
    # __DUMMY_MARL__##################################################

    # Set Method
    # box = np.array(Domain.reward_range)
    box = np.array([np.vstack((domain.r_reward, domain.c_reward)).min(),
                    np.vstack((domain.r_reward, domain.c_reward)).max()])
    epsilon = np.array([-0.01, 0.01])

    # Initialize Starting Point
    # Start = np.array([0,1])
    if start_strategies is None:
        start_strategies = np.random.random((domain.players, domain.dim))
        for i in range(start_strategies.shape[0]):
            start_strategies[i] /= np.sum(start_strategies[i])
        # start_strategies = np.array([[.99, .01], [.5, .5]])

    # Set Options
    initialization_conditions = Initialization(step=1e-4)
    # init = Initialization(Step=-0.1)
    # terminal_conditions = Termination(max_iter=iterations, tols=[(domain.ne_l2error, 1e-20)])
    terminal_conditions = Termination(max_iter=iterations)
    reporting_options = method.reporting_options()
    whatever_this_does = Miscellaneous()
    options = DescentOptions(initialization_conditions, terminal_conditions, reporting_options, whatever_this_does)

    # Print Stats
    # print_sim_stats(domain,Method,Options)

    # Start Solver
    tic = time.time()
    # set random seed
    # np.random.seed(0)
    marl_results = solve(start_strategies, method, domain, options)
    toc = time.time() - tic

    # Print Results
    if config.debug_output_level != -1:
        print_sim_results(options, marl_results, method, toc)

    # computing the statistics
    reward = np.array(marl_results.perm_storage['Reward'])
    win_ratio = np.mean(.5 + .5*reward, axis=0).tolist()#.round(2)
    print_exception = False
    if win_ratio[1] < .4 and type(method) is LEAP:
        print 'Problem:', win_ratio[1]
        # print_exception = True

    # creating plots, if desired:
    if config.show_plots or print_exception:
        policy = np.array(marl_results.perm_storage['Policy'])[:, :, 0]  # Just take probabilities for first action
        if 'Forecaster Policies' in marl_results.perm_storage:
            forecaster = np.array(marl_results.perm_storage['Forecaster Policies'])[:, :, :, 0]  # Just take probabilities for first action
        # policy_est = np.array(marl_results.perm_storage['Policy Estimates'])
        print marl_results.perm_storage['Value Function'][-1]
        val_fun = np.array(marl_results.perm_storage['Value Function'][-1])
        print val_fun
        true_val_fun = np.array(marl_results.perm_storage['True Value Function'])
        # val_var = np.array(marl_results.perm_storage['Value Variance'])
        pol_grad = np.array([[marl_results.perm_storage['Policy Gradient (dPi)'][i][0, 0],
                              marl_results.perm_storage['Policy Gradient (dPi)'][i][1, 0]]
                             for i in range(1, len(marl_results.perm_storage['Policy Gradient (dPi)']))])
        pol_lr = np.array(marl_results.perm_storage['Policy Learning Rate'])
        reward = np.array(marl_results.perm_storage['Reward'])
        action = np.array(marl_results.perm_storage['Action'])
        # value_function_values = np.array(MARL_Results.perm_storage['Value Function'])[:, :, 0]
        print('Ratio of games won:')
        win_ratio = np.mean(.5 + .5*reward, axis=0).round(2)
        print 'Player 1:', win_ratio[0], 'Player 2:', win_ratio[1]
        print('Endpoint:')
        print(policy[-1])
        pol_grad = np.vstack(([0., 0.], pol_grad))
        print 'pol1, pol2, action1, action2, reward1, reward2, polgrad1, polgrad2'
        # for i in np.hstack((np.round(policy, 2), action, reward, np.round(pol_grad, 2))):
        #     print i

        # ax[0, 1].plot(policy[:, 0], policy[:, 1])
        # ax[0, 1].set_xlim(box + epsilon)
        # ax[0, 1].set_ylim(box + epsilon)
        # ax[0, 1].set_title('The policy-policy space')
        # ax[0, 1].grid(True)

        printing_data = {}
        # printing_data['The value function'] = {'values': val_fun.T, 'smooth':-1}
        printing_data['Expected Reward of policies played'] = {'values': true_val_fun, 'yLimits': box, 'smooth': -1}
        printing_data['The policies'] = {'values': policy, 'yLimits': np.array([0, 1]) + epsilon, 'smooth': -1}
        if 'Forecaster Policies' in marl_results.perm_storage:
            printing_data['Hypotheses - Player 1'] = {'values': forecaster[:, 0, :], 'yLimits': np.array([0, 1]) + epsilon, 'smooth': -1}
            printing_data['Hypotheses - Player 2'] = {'values': forecaster[:, 1, :], 'yLimits': np.array([0, 1]) + epsilon, 'smooth': -1}
        # printing_data['The policy gradient'] = {'values': pol_grad}
        # printing_data['Policy Estimates'] = {'values': policy_est, 'yLimits': box}
        plot_results(printing_data)

    # return win_ratio
    # normalizing the reward
    # reward_range = reward.max() - reward.min()
    # reward_offset = reward.min()
    # reward = (reward - reward_offset) / reward_range
    return reward


def wrapper_batch_testing_helper(domains, trials, iterations, bt_type):
    for dom in domains:
        domain = domains[dom]
        methods = [WPL(domain, P=BoxProjection(low=.001)),
                   MAHT_WPL(domain, P=LinearProjection()),
                   LEAP(domain, P=LinearProjection())]
        config.batch_testing = bt_type
        results = np.zeros((len(methods), trials, iterations, 2))
        for trial in range(trials):
            start_st = np.random.random((domain.players, domain.dim))
            for k in range(start_st.shape[0]):
                start_st[k] /= np.sum(start_st[k])
            start_st = np.array([[.5, .5], [.99, .01]])
            for method in range(len(methods)):
                results[method, trial, :, :] = demo(domain, methods[method], iterations, start_st)[1:]

        # store the results
        np.save(dom+'.WPL-MAHT-LEAP.npy', results)


def wrapper_batch_testing():
    # batch testing global settings
    config.show_plots = False
    config.debug_output_level = -1

    # method = IGA(domain, P=BoxProjection())
    # method = WoLFIGA(domain, P=BoxProjection(), min_step=1e-4, max_step=1e-3 )
    # method = MySolver(domain, P=BoxProjection())
    # method = MyIGA(domain, P=BoxProjection())
    # method1 = WPL(domain, P=BoxProjection(low=.001))
    # method = AWESOME(domain, P=LinearProjection())
    # method = PGA_APP(domain, P=LinearProjection())
    # method = MultiAgentVI(domain, P=LinearProjection())
    # method2 = LEAP(domain, P=LinearProjection())

    domains = {'tricky':    Tricky(),
               'mp':        MatchingPennies(),
               'battle':    BattleOfTheSexes(),
               'pure':      PureStrategyTest()}
    domains_b = {'deficientMP': MatchingPennies()}
    # results = [[] for _ in range(len(methods))]
    wrapper_batch_testing_helper(domains, 200, 200, 2)
    wrapper_batch_testing_helper(domains_b, 200, 8000, 1)


def wrapper_singular_runs():
    # domain = PureStrategyTest()
    domain = MatchingPennies()
    # start_st = np.array([[.5, .5], [.01, .99]])
    start_st = None
    # domain = Tricky()
    # domain = PrisonersDilemma()
    # method2 = WoLFGIGA(domain, P=LinearProjection())
    # method2 = LEAP(domain, P=LinearProjection())
    # method2 = WPL(domain, P=LinearProjection())
    method2 = MAHT_WPL(domain, P=LinearProjection())
    # config.batch_testing = 1
    demo(domain, method2, 1000, start_strategies=start_st)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--suppress-plots', action='store_false', dest='plot', default=True, help='suppress plots')
    parser.add_option('-v', '--show-output', type='int', dest='debug', default=0, help='debug level')
    (options, args) = parser.parse_args()
    config.debug_output_level = options.debug
    config.show_plots = options.plot

    # wrapper_batch_testing()
    wrapper_singular_runs()
    # import Analyses as ana
    # ana.do_analysis()