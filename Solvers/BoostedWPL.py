from __future__ import print_function
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
                 learning_rate=0.06,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
                 averaging_meta_window=5,
                 no_forecasters=5,
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
        self.learning_settings = [[1., .75, .08], [1., .3, .1], [1., 1., .03], [1.5, 2.5, .01], [2., 3., .008]]
        self.value_approx_range = np.array([1./(51-1)*i for i in range(51)])
        self.additive_ = compute_increase_vector(15, .7, .1)
        # self.ne_hypotheses = np.zeros(self.value_approx_range.shape)

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Action'] = np.zeros((self.storage_size,
                                                # self.domain.players,
                                                2,
                                                )).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Policy Learning Rate'] = (np.ones((self.storage_size,
                                                              # self.domain.players,
                                                              2,
                                                              self.get_forecaster_no('m')))
                                                     * self.lr).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Reward'] = np.zeros((self.storage_size,
                                                # self.domain.players,
                                                2,
                                                )).tolist()
        self.temp_storage['Value Function'] = np.zeros((self.storage_size,
                                                        # self.domain.players,
                                                        2,
                                                        self.get_forecaster_no('m'),
                                                        )).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size,
                                                             # self.domain.players,
                                                             2,
                                                             )).tolist()
        self.temp_storage['Forecaster Policies'] = (np.zeros((self.storage_size,
                                                              # self.domain.players,
                                                              2,
                                                              self.get_forecaster_no('m'),
                                                              # self.domain.Dim,
                                                              2,
                                                              ))/2.)
        self.temp_storage['NE guesser'] = np.zeros((self.storage_size,
                                                    # self.domain.players,
                                                    2,
                                                    51,
                                                    )).tolist()
        for i in range(2):
            for j in range(self.get_forecaster_no('w')):
                self.temp_storage['Forecaster Policies'][:, i, j, :] = np.array(start[i])
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
                                   'Forecaster Policies',
                                   'NE guesser',
                                   # 'Policy Estimates',
                                   ])

    def compute_value_function(self, reward, value_old):
        return (reward + self.estimator_decay * np.array(value_old)) / (1 + self.estimator_decay)

    def project_error(self, value, scaling_factor=1, exponent=1):
        return np.sign(value) * ((np.absolute(value) * scaling_factor)**exponent)
        # return np.sign(value) * (1 - np.cos((np.absolute(value) * scaling_factor)**exponent))

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
        return policy_gradient, self.Proj.p(forecaster_policy, learning_setting[2], policy_gradient)

    def weighted_average_forecaster(self, fc_policies, reward_history, action_history, averaging_type=None):
        # calculating the reward-action space
        fcp = np.array([fc_policies[i, :, action_history[i]]*(.5**(len(action_history)-i-1)) for i in range(len(action_history))])
        weighted_reward = np.array(reward_history)
        weighted_reward[weighted_reward < np.mean(weighted_reward)] *= 25.
        reward_action_function = np.mean(np.multiply(fcp, np.repeat(weighted_reward[None].T,
                                                                    fcp.shape[1], axis=1)),
                                         axis=0)
        # applying the weights
        weight = 18.#/(reward_action_function.max()+1e-9)
        if averaging_type == 'exponential':
            reward_action_function *= weight
            reward_action_function = np.exp(reward_action_function)
        best_estimate = np.dot(fc_policies[-1].T, reward_action_function[None].T)
        best_estimate = best_estimate.T[-1]/(np.sum(best_estimate)+1e-9)
        # returning after projecting back on the simplex
        if self.get_forecaster_no('r') > self.get_forecaster_no('w'):
            best_ne_index = self.get_forecaster_no('w') + reward_action_function[self.get_forecaster_no('r')-1:].argmax()
            if np.linalg.norm(best_estimate-fc_policies[-1, best_ne_index]) < .5:
                return fc_policies[-1, best_ne_index]
        return best_estimate

    def equilibrium_testing(self, fc_policies, ne_hypos, decision_type='majority', iteration=0):
        """
        given a set of policies, this method decides on whether or not an equilibrium has been found, based
        on the deviation of the different policy hypotheses.
        :param fc_policies:
        :return:
        """
        # 1. computing the symmetrical distance matrix between the different hypotheses
        distances = np.zeros((self.get_forecaster_no('w'), self.get_forecaster_no('w')))
        for i in range(self.get_forecaster_no('w')):
            for j in range(i+1, self.get_forecaster_no('w'), 1):
                distances[i, j] = np.linalg.norm(fc_policies[i] - fc_policies[j])
        if decision_type == 'majority' and iteration > 50:
            if np.sum(distances > .02) < 5:
                # exception = np.mean(distances > .1, axis=0).argmax()
                # if config.debug_output_level > 2:
                hypo_index = compute_value_index(self.value_approx_range,
                                                 np.mean(fc_policies[:self.get_forecaster_no('w')], axis=0)[0])
                ne_hypos = add_value_vector(ne_hypos, self.additive_, hypo_index)
                # print np.round(np.mean(fc_policies[:self.get_forecaster_no('w')], axis=0), 2), iteration
                # print ne_hypos
                if ne_hypos.max() > .5:
                    # print self.value_approx_range[ne_hypos.argmax()], ne_hypos.max()
                    pol = np.array([self.value_approx_range[ne_hypos.argmax()],
                                    1-self.value_approx_range[ne_hypos.argmax()]])
                    # self.get_forecaster_no('w') += 1
                    # TODO: making sure that we do not add the same ne forecaster again
                    # if we have seen it, update it instead
                    print('the policies:\n', fc_policies[self.get_forecaster_no('w'):])
                    print('the policies:\n', fc_policies-pol)
                    print('distances:', np.sqrt(np.sum(np.square(fc_policies[self.get_forecaster_no('w'):]-pol), axis=1)))
                    if np.sqrt(np.sum(np.square(fc_policies[self.get_forecaster_no('w'):]-pol), axis=1)).min() > .6:
                        # if we have not seen it, add it
                        self.learning_settings.append([1., 1., .0])
                        print(fc_policies, self.get_forecaster_no('r'), fc_policies[self.get_forecaster_no('r')])
                        fc_policies[self.get_forecaster_no('r')-1] = pol
                        print('added a new ne hypothesis:', pol)
                    else:
                        nep = np.sqrt(np.sum(np.square(fc_policies[self.get_forecaster_no('w'):]-pol), axis=0)).argmin()
                        fc_policies[self.get_forecaster_no('w')+nep] = (pol + fc_policies[self.get_forecaster_no('w')+nep])/2
                        print('modified the hypothesis', fc_policies[self.get_forecaster_no('w')+nep], 'to')
                    # print(self.value_approx_range[self.ne_hypotheses.argmax()])
                    ne_hypos *= .99
        return ne_hypos, fc_policies

    def get_forecaster_no(self, option='r'):
        if option == 'r':
            return len(self.learning_settings)
        if option == 'w':
            return self.no_forecasters
        if option == 'm':
            return self.no_forecasters + 3

    def update(self, record):
        # Retrieve Necessary Data
        iteration = len(record.perm_storage['Policy'])
        policy = record.temp_storage['Policy'][-1]
        tmp_policy = self.temp_storage['Policy']
        tmp_reward = np.array(self.temp_storage['Reward'][-1*iteration:])
        tmp_action = np.array(self.temp_storage['Action'][-1*iteration:])
        tmp_value = self.temp_storage['Value Function'][-1]
        ne_estimates = np.array(self.temp_storage['NE guesser'][-1])
        tmp_forecaster_policies = np.array(self.temp_storage['Forecaster Policies'][-1*iteration:])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # Initialize Storage
        temp_data = {}
        learning_rate = np.zeros((2,)).tolist()
        policy_gradient = np.zeros((2, 2))#np.ones((2,))*.1 if tmp_pol_grad[0].any() == 0 else tmp_pol_grad[-1]
        value = np.zeros((2, self.no_forecasters)).tolist()
        action = [0, 0]
        reward = [0., 0.]
        policy_taken = np.array(policy)
        updated_forecaster_policies = np.array(tmp_forecaster_policies[-1])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # 1. start by calculating the forecaster's policy/recommendation
        for player in range(self.domain.players):
            ne_estimates[player], updated_forecaster_policies[player] = self.equilibrium_testing(updated_forecaster_policies[player],
                                                                                                 ne_estimates[player],
                                                                                                 'majority',
                                                                                                 iteration)
            policy_taken[player] = self.weighted_average_forecaster(tmp_forecaster_policies[:, player],
                                                                    tmp_reward[:, player],
                                                                    tmp_action[:, player],
                                                                    averaging_type='exponential')
            # policy_value_distribution = self.weighted_average_forecaster(tmp_forecaster_policies[:, player],
            #                                                              tmp_reward[:, player],
            #                                                              tmp_action[:, player])
            # print policy_value_distribution
            # policy_taken[player] = updated_forecaster_policies[player][policy_value_distribution.argmax()]
            # policy_taken[player] = updated_forecaster_policies[player][0]

        # stupid random play
        # if iteration % 80 >= 40:
        #     policy_taken[0] = [1., 0.]
        # else:
        #     policy_taken[0] = [0., 1.]

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
            print ('-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-')
        # perform the update on the policies:
        for player in range(2):
            # iterating over the different forecasters
            for i in range(self.get_forecaster_no('w')):
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
                print ('-> player', player)
                print ('   - which update is performed?', policy_gradient[player][action[player]] < 0)
                print ('   - temp policies last round:  %.4f' % tmp_policy[-1][player][0])
                print ('   - temp value last round:     %.4f' % tmp_value[-1][player])
                print ('   - the learning rate:         %.4f' % learning_rate[player])
                print ('   - the policy gradient:       ', policy_gradient[player])
                print ('   - the resulting policy:      ', policy_taken[player][0])

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
        temp_data['NE guesser'] = ne_estimates
        self.book_keeping(temp_data)

        return self.temp_storage