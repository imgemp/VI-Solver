__author__ = 'clemens'

import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *
from Options import Reporting

import config
from Estimators import decaying_average_estimator


class AWESOME(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 delta0=1e-2,
                 growth_limit=2,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=20,
                 averaging_meta_window=5):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.storage_size = 250
        self.temp_storage = {}
        self.min_step = min_step
        self.max_step = max_step
        self.averaging_window = averaging_window
        self.exploration_trials = exploration_trials
        self.amw = averaging_meta_window

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Action'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Policy Learning Rate'] = (np.ones((self.storage_size, 2))*options.init.step).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Variance'] = np.zeros((self.storage_size, 2)).tolist()
        # self.temp_storage['Projections'] = self.storage_size * [0]
        self.temp_storage['Reward'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Performance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Am I winning?'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Variance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Am I stable?'] = np.zeros((self.storage_size, 2)).tolist()

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   'Reward',
                                   'Value Function',
                                   'True Value Function',
                                   'Value Variance',
                                   'Performance',
                                   'Am I winning?',
                                   # 'Policy Estimates',
                                   ])

    def decaying_reward_action_estimator(self, reward_history, action_history, averaging_window, window_reps, decaying_rate=.9):
        """
        This is the decaying average estimator. It computes a weighted average, where the weights are
        decaying backwards.
        :param reward_history:
        :param averaging_window: the number of observations considered to be locally stable
        :param window_reps: the number of observation windows considered to be relevant
        :param decaying_rate: the rate at which the observations decay
        :param action_history: another value history where the gradients on the observation-window gaps are considered
        :return:
        """
        indices_grad = np.array([-1*averaging_window*i for i in range(1, window_reps+1)], dtype='int')
        indices_val = np.array([-1*averaging_window*i-1 for i in range(window_reps)], dtype='int')
        wlen = len(indices_grad)
        # if the second parameter, i.e. the gradient-weight, is not specified
        if action_history is None:
            val_diff = np.ones((window_reps))
        else:
            val_diff = np.array(action_history)[indices_val] - np.array(action_history)[indices_val-1]
        weights = np.array([decaying_rate**(wlen-i) for i in range(wlen)])
        return np.sum(np.multiply(np.multiply(np.array(reward_history)[indices_grad], val_diff), weights))

    def compute_value_function(self, reward_history, window_size):
        return np.mean(reward_history[-1*window_size:])

    def compute_reward_per_action(self, tmp_act, tmp_rew):
        # retrieving the different actions
        # actions = range(self.domain.Dim)
        reward_per_action = np.zeros((self.domain.Dim, ))
        actions_taken = np.ones((self.domain.Dim, ))
        for i in range(len(tmp_act)):
            actions_taken[tmp_act[i]] += 1
            reward_per_action[tmp_act[i]] += tmp_rew[tmp_act[i]]
        return np.divide(reward_per_action, actions_taken)

    def compute_random_policy(self):
        policy = np.random.random((self.domain.Dim, ))
        if not np.sum(policy) == 0.:
            policy *= 1/np.sum(policy)
        else:
            policy = np.ones((self.domain.Dim, )) / self.domain.Dim
        return policy

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_grad = self.temp_storage['Policy Gradient (dPi)']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
        tmp_act = self.temp_storage['Action']
        tmp_val = self.temp_storage['Value Function']
        tmp_perf = self.temp_storage['Performance']
        # tmp_act = self.temp_storage['Action']
        tmp_var = self.temp_storage['Value Variance']
        tmp_pol_var = self.temp_storage['Policy Variance']
        tmp_winning = np.array(self.temp_storage['Am I winning?'])
        last_stable = self.temp_storage['Am I stable?'][-1]

        comp_val_hist = np.array(record.perm_storage['Value Function'])

        # Initialize Storage
        temp_data = {}
        learning_rate = np.zeros((2,)).tolist()
        # policy_gradient = np.ones((2,))
        policy_gradient = np.ones((2, 2))#np.ones((2,))*.1 if tmp_pol_grad[0].any() == 0 else tmp_pol_grad[-1]
        value = np.zeros((2,)).tolist()
        performance = np.zeros((2,)).tolist()
        action = [0, 0]
        reward = [0., 0.]
        mean_val = [0., 0.]
        ne_strategy = self.domain.NE
        updated_policy = list(policy)
        val_var = list(tmp_var[-1])
        pol_var = list(tmp_pol_var[-1])
        winning = [False, False]
        fixpoint = [False, False]
        iteration = len(record.perm_storage['Policy'])
        # lavi = self.compute_last_update_index(self.averaging_window)
        epsilon_t_s = .1

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # start by playing the game:
        for player in range(self.domain.players):
            # playing the game
            action[player] = self.domain.action(policy[player])

        # compute the reward and the resulting value function:
        for player in range(self.domain.players):
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]
            # compute the value of the current strategy
            value[player] = self.compute_value_function(np.hstack((np.array(tmp_rew)[:, player], [reward[player]])),
                                                        self.averaging_window)
            val_var[player] = (np.array(tmp_val)[-1*self.averaging_window:, player]).tolist() + [value[player]]
        # value = self.domain.compute_value_function(policy)
        # print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        if config.debug_output_level >= 1:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        update = (iteration % self.averaging_window) == 0
        appe = True
        aps = True
        delta = False
        t = 0
        phi = ne_strategy[player]
        if update:
            # beginning an epoch
            if not appe:
                if not delta:
                    for player in range(self.domain.players):
                        if np.mean(np.array(tmp_rew[-1*self.averaging_window:-1, player])) - \
                                np.array(tmp_rew[-2*self.averaging_window:-1*self.averaging_window-1, player]) \
                                > epsilon_t_s:
                            aps = False
                delta = False
                for player in range(self.domain.players):
                    action_reward = self.compute_reward_per_action(np.array(tmp_act[:, player]),
                                                                   np.array(tmp_rew[:, player]))
                    highest_reward_action = np.argmax(action_reward)
                    if action_reward[highest_reward_action] > \
                                    np.mean(np.array(tmp_rew[:, player])) + self.domain.players * epsilon_t_s:
                        # The following two lines assume that the policies all lie on a simplex
                        updated_policy = np.zeros((self.domain.Dim, ))
                        updated_policy[highest_reward_action] = 1.
            else:
                for i in range(self.domain.players):
                    if np.mean(np.array(tmp_rew)[-1*self.averaging_window:-1, player]) - \
                                np.mean(np.array(tmp_rew)[-2*self.averaging_window:-1*self.averaging_window-1, player]) \
                                > epsilon_t_s:
                        appe = False
                        phi = self.compute_random_policy()

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # if config.debug_output_level >= 1:
        #     print '-> player', player
        #     print '   - which update is performed?', policy_gradient[player][action[player]] < 0
        #     print '   - temp policies last round:  %.4f' % tmp_pol[-1][player][0]
        #     print '          - round before that:  %.4f' % tmp_pol[lavi][player][0]
        #     print '   - temp value last round:     %.4f' % tmp_val[-1][player]
        #     print '          - round before that:  %.4f' % tmp_val[lavi][player]
        #     print '   - the winningness:           %.4f' % performance[player]
        #     print '   - the learning rate:         %.4f' % learning_rate[player]
        #     print '   - the policy gradient:       ', policy_gradient[player]
        #     print '   - the resulting policy:      ', updated_policy[player][0]
        #     print '   - the resulting polgrad:     ', (updated_policy[player]-tmp_pol[lavi][player])[0]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value
        temp_data['Performance'] = performance
        temp_data['True Value Function'] = self.domain.compute_value_function(updated_policy)
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        temp_data['Reward'] = reward
        temp_data['Action'] = action
        temp_data['Value Variance'] = val_var
        temp_data['Policy Variance'] = pol_var
        temp_data['Am I stable?'] = fixpoint
        temp_data['Am I winning?'] = winning
        self.book_keeping(temp_data)

        return self.temp_storage