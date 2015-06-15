__author__ = 'clemens'

import numpy as np
from VISolver.Projection import *
from VISolver.Solvers.Solver import Solver
from VISolver.Utilities import *
import VISolver.config
from VISolver.Estimators import decaying_average_estimator
from VISolver.Options import Reporting


class PGA_APP(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 delta0=1e-2,
                 growth_limit=2,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
                 averaging_meta_window=5,
                 learning_rate_t=.0001,
                 learning_rate_e=.0001,
                 discount_factor=.95,
                 q_approx=50):

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
        self.lr_t = learning_rate_t
        self.lr_e = learning_rate_e
        self.discounting = discount_factor
        self.no_q_approx = q_approx
        self.q_approx_range = np.array([1./(q_approx-1)*i for i in range(q_approx)])

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Action'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy'] = (np.ones((self.storage_size, 2, self.no_q_approx, 2))/2).tolist()
        self.temp_storage['Last Policy'] = (np.ones((self.storage_size, 2, self.no_q_approx, 2))/2).tolist()
        self.temp_storage['Policy Learning Rate'] = (np.ones((self.storage_size, 2))*options.init.step).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Variance'] = np.zeros((self.storage_size, 2)).tolist()
        # self.temp_storage['Projections'] = self.storage_size * [0]
        self.temp_storage['Reward'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Performance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Q Function'] = np.zeros((self.storage_size, 2, self.no_q_approx, 2)).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Am I winning?'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Variance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Am I stable?'] = np.zeros((self.storage_size, 2)).tolist()

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Last Policy',
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

    def am_i_winning(self, value_history, last_result):
        # deciding on the position in the range of the value history:
        min_value = min(value_history)
        max_value = max(value_history)
        range_value = max_value - min_value
        mean_value = np.mean(value_history)
        last_value = value_history[-1]
        # print 'value history: ', value_history
        # print 'the mean value of the value function ', mean_value

        # the winning part is a float in the range of winningness
        range_value = range_value if range_value != 0. else 1.
        val_index = [-1*i for i in range(self.averaging_window)]
        return (last_value-min_value)/range_value, \
               (mean_value-min_value)/range_value, \
               last_value > mean_value - np.mean(np.array(value_history)[val_index])#)/range_value)

    def am_i_at_a_fixpoint(self, variance_history, last_result):
        # deciding on the position in the range of the value history:
        min_value = min(variance_history)
        max_value = max(variance_history)
        range_value = max_value - min_value
        mean_value = np.mean(variance_history)
        last_value = variance_history[-1]

        # the winning part is a float in the range of winningness
        range_value = range_value if abs(range_value) > 1e-2 else 1.
        return (last_value-min_value)/range_value, last_value > mean_value

    def compute_value_function(self, reward_history, window_size):
        return np.mean(reward_history[-1*window_size:])

    def compute_last_update_index(self, averaging_window):
        return -1 * averaging_window

    def compute_update_direction(self, policy_gradient_history, value_history):
        indices_grad = np.array([-1*self.averaging_window*i for i in range(1, self.amw+1)], dtype='int')
        indices_val = np.array([-1*self.averaging_window*i-1 for i in range(self.amw)], dtype='int')
        val_diff = np.array(value_history)[indices_val] - np.array(value_history)[indices_val-1]
        direction = np.sum(np.multiply(np.array(policy_gradient_history)[indices_grad], val_diff))
        return direction

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

    def compute_q_index(self, value):
        try:
            print 'value', value
            print 'qindex', (np.abs(self.q_approx_range-np.array(value)[:, 0])).argmin()
            return (np.abs(self.q_approx_range-np.array(value)[:, 0])).argmin()
        except TypeError:
            return(np.abs(self.q_approx_range-value)).argmin()

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        last_policy = record.temp_storage['Last Policy'][-1]
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
        q_values = self.temp_storage['Q Function'][-1]
        # print q_values

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
        updated_policy = last_policy
        val_var = list(tmp_var[-1])
        pol_var = list(tmp_pol_var[-1])
        winning = [False, False]
        fixpoint = [False, False]
        iteration = len(record.perm_storage['Policy'])
        lavi = self.compute_last_update_index(self.averaging_window)

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # start by playing the game:
        print 'iteration', iteration
        for player in range(self.domain.players):
            # playing the game
            print 'first call'
            action[player] = self.domain.action(policy[player][self.compute_q_index(last_policy[player])])

        # compute the reward and the resulting value function:
        for player in range(self.domain.players):
            policy_index = self.compute_q_index(policy[player])
            print 'player', player
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]
            # compute the value of the current strategy
            value[player] = self.compute_value_function(np.hstack((np.array(tmp_rew)[:, player], [reward[player]])),
                                                        self.averaging_window)
            val_var[player] = (np.array(tmp_val)[-1*self.averaging_window:, player]).tolist() + [value[player]]
            # updating the q-value
            print 'second call'
            old_q_value = q_values[player][policy_index][action[player]]
            print 'third call'
            q_value_succ = [q_values[player][policy_index][i] for i in range(2)]
            new_q_value = (1 - self.lr_t) * old_q_value + \
                          self.lr_t * (reward[player] + self.discounting * max(q_value_succ))
            print 'fourth call'
            q_values[player][policy_index][action[player]] = new_q_value
            value = np.dot(q_values[player][policy_index], policy[player][policy_index])
            print 'new q value', q_values[player][policy_index], policy[player][policy_index], value

        # value = self.domain.compute_value_function(policy)
        # print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        if config.debug_output_level >= 1:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        for player in range(self.domain.players):
            for act in range(2):
                pass
            if config.debug_output_level >= 1:
                print '-> player', player
                print '   - which update is performed?', policy_gradient[player][action[player]] < 0
                print '   - temp policies last round:  %.4f' % tmp_pol[-1][player][0]
                print '          - round before that:  %.4f' % tmp_pol[lavi][player][0]
                print '   - temp value last round:     %.4f' % tmp_val[-1][player]
                print '          - round before that:  %.4f' % tmp_val[lavi][player]
                print '   - the winningness:           %.4f' % performance[player]
                print '   - the learning rate:         %.4f' % learning_rate[player]
                print '   - the policy gradient:       ', policy_gradient[player]
                print '   - the resulting policy:      ', updated_policy[player][0]
                print '   - the resulting polgrad:     ', (updated_policy[player]-tmp_pol[lavi][player])[0]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Last Policy'] = new_policy = updated_policy
        print 'updated policy', updated_policy
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
        temp_data['Q Function'] = q_values
        self.book_keeping(temp_data)

        return self.temp_storage