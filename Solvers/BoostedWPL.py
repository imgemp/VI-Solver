__author__ = 'clemens'

import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *
import config
from Options import Reporting
from Estimators import decaying_average_estimator


class BoostedWPL(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 learning_rate=0.03,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
                 averaging_meta_window=5,
                 no_forecasters=2,
                 estimator_decay=.9):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.storage_size = 150
        self.temp_storage = {}
        self.min_step = min_step
        self.max_step = max_step
        self.averaging_window = averaging_window
        self.exploration_trials = exploration_trials
        self.amw = averaging_meta_window
        self.no_forecasters = no_forecasters
        self.lr = learning_rate
        self.estimator_decay = estimator_decay
        self.learning_settings = [[.5, .5], [.75, .75], [1., 1.], [1.2, 2], [1.2, 3]]

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Action'] = np.zeros((self.storage_size,
                                                # self.domain.players,
                                                2,
                                                )).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Policy Learning Rate'] = (np.ones((self.storage_size,
                                                              # self.domain.players,
                                                              2,
                                                              self.no_forecasters))
                                                     * self.lr).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Reward'] = np.zeros((self.storage_size,
                                                # self.domain.players,
                                                2,
                                                )).tolist()
        self.temp_storage['Value Function'] = np.zeros((self.storage_size,
                                                        # self.domain.players,
                                                        2,
                                                        self.no_forecasters
                                                        )).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size,
                                                             # self.domain.players,
                                                             2,
                                                             )).tolist()
        self.temp_storage['Forecaster Policies'] = (np.zeros((self.storage_size,
                                                              # self.domain.players,
                                                              2,
                                                              self.no_forecasters,
                                                              # self.domain.Dim,
                                                              2,
                                                              ))/2.)
        for i in range(2):
            self.temp_storage['Forecaster Policies'][:, i, :, :] = np.array(start)
        self.temp_storage['Forecaster Policies'] = self.temp_storage['Forecaster Policies'].tolist()
        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   'Reward',
                                   'Value Function',
                                   'True Value Function',
                                   'Action',
                                   # 'Policy Estimates',
                                   ])

    def compute_value_function(self, reward, value_old):
        return (reward + self.estimator_decay * np.array(value_old)) / (1 + self.estimator_decay)

    def project_error(self, value, scaling_factor=1.3, exponent=3):
        # return value
        return np.sign(value) * ((abs(value) * scaling_factor)**exponent)
        # return np.sign(value) * (1 - np.cos(value * 3.14159))

    def singular_wpl_update(self, forecaster_policy, action, reward, prev_value, learning_setting):
        # print 'in wpl update:', forecaster_policy, action, reward, prev_value
        # compute the policy gradient:
        policy_gradient = np.zeros(2)
        policy_error = reward - prev_value
        # WPL projections
        policy_gradient[action] = self.project_error(policy_error, learning_setting[0], learning_setting[1])
        if policy_error < 0:
            policy_gradient[action] *= forecaster_policy[action]
        else:
            policy_gradient[action] *= 1 - forecaster_policy[action]
        # computing the policy gradient and the learning rate
        return policy_gradient, self.Proj.p(forecaster_policy, self.lr, policy_gradient)

    def weighted_average_forecaster(self, fc_policies, reward_history, action_history):
        fcp = np.array([fc_policies[i, :, action_history[i]] for i in range(len(action_history))])
        reward_action_function = np.dot(fcp.T, reward_history)
        # print 'raf', reward_action_function
        # print 'bla', reward_history, action_history, fcp
        return reward_action_function

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        tmp_policy = self.temp_storage['Policy']
        tmp_reward = self.temp_storage['Reward']
        tmp_action = self.temp_storage['Action']
        tmp_value = self.temp_storage['Value Function'][-1]
        tmp_forecaster_policies = np.array(self.temp_storage['Forecaster Policies'])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # Initialize Storage
        temp_data = {}
        learning_rate = np.zeros((2,)).tolist()
        policy_gradient = np.zeros((2, 2))#np.ones((2,))*.1 if tmp_pol_grad[0].any() == 0 else tmp_pol_grad[-1]
        value = np.zeros((2, self.no_forecasters)).tolist()
        action = [0, 0]
        reward = [0., 0.]
        policy_taken = np.array(policy)
        iteration = len(record.perm_storage['Policy'])
        updated_forecaster_policies = np.array(tmp_forecaster_policies[-1])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 1. start by calculating the forecaster's policy/recommendation
        for player in range(self.domain.players):
            policy_value_distribution = self.weighted_average_forecaster(tmp_forecaster_policies[player],
                                                                         tmp_reward[player],
                                                                         tmp_action[player])
            # print policy_value_distribution
            policy_taken[player] = updated_forecaster_policies[player][policy_value_distribution.argmax()]
            # policy_taken[player] = updated_forecaster_policies[player][0]

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 2. then play the game
        for player in range(self.domain.players):
            # playing the game
            action[player] = self.domain.action(policy_taken[player])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 3. compute the reward and the resulting value function:
        for player in range(self.domain.players):
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]

        # print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 4. updating the policies
        if config.debug_output_level >= 1:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        for player in range(2):
            # iterating over the different forecasters
            for i in range(self.no_forecasters):
                # compute the value of the current strategy
                value[player][i] = self.compute_value_function(reward[player], tmp_value[player][i])
                policy_gradient[player], \
                updated_forecaster_policies[player][i] = self.singular_wpl_update(
                                                          tmp_forecaster_policies[-1][player][i],
                                                          action[player],
                                                          reward[player],
                                                          value[player][i],
                                                          self.learning_settings[i])

            # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
            if config.debug_output_level >= 1:
                print '-> player', player
                print '   - which update is performed?', policy_gradient[player][action[player]] < 0
                print '   - temp policies last round:  %.4f' % tmp_policy[-1][player][0]
                print '   - temp value last round:     %.4f' % tmp_value[-1][player]
                print '   - the learning rate:         %.4f' % learning_rate[player]
                print '   - the policy gradient:       ', policy_gradient[player]
                print '   - the resulting policy:      ', policy_taken[player][0]

        # updated_policy[0] = [1., 0.]

        # Store Data
        temp_data['Policy'] = policy_taken
        temp_data['Value Function'] = value
        temp_data['True Value Function'] = self.domain.compute_value_function(policy_taken)
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        temp_data['Reward'] = reward
        temp_data['Action'] = action
        temp_data['Forecaster Policies'] = updated_forecaster_policies
        self.book_keeping(temp_data)

        return self.temp_storage