from __future__ import division
__author__ = 'clemens'


import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *
from Options import Reporting
from Estimators import decaying_average_estimator

import config
# from pykalman import KalmanFilter as kf


class MultiAgentVI(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 delta0=1e-2,
                 growth_limit=2,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=1,
                 exploration_trials=0,
                 averaging_meta_window=30,
                 learning_rate_t=.001,
                 learning_rate_e=.01,
                 discount_factor=.9,
                 exploration_rate=.0,
                 discount_range_percentage=13,
                 discount_range_decay=.5,
                 value_approx_steps=51):

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
        self.exploration_rate = exploration_rate
        self.discounting = discount_factor
        self.disc_range = discount_range_percentage
        self.discount_range_decay = discount_range_decay
        self.value_approx_no = value_approx_steps
        self.disc_increase_vector = self.compute_increase_vector()
        self.value_approx_range = np.array([1./(value_approx_steps-1)*i for i in range(value_approx_steps)])

    def init_temp_storage(self, start, domain, options):
        self.temp_storage['Action'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy'] = self.storage_size * [start]
        # self.temp_storage['Policy Estimates'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy Learning Rate'] = (np.ones((self.storage_size, 2))*options.init.step).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        # self.temp_storage['Projections'] = self.storage_size * [0]
        self.temp_storage['Reward'] = np.zeros((self.storage_size, 2)).tolist()
        # self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2,
        #                                                 self.value_approx_no, self.value_approx_no))
        self.temp_storage['Value Function'] = np.ones((self.storage_size, 2, self.value_approx_no)).tolist()
        self.temp_storage['True Value Function'] = np.zeros((self.storage_size, 2)).tolist()

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   'Reward',
                                   'Value Function',
                                   'True Value Function',
                                   # 'Policy Estimates',
                                   ])

    def compute_increase_vector(self):
        dr_len = int(self.disc_range / 100. * self.value_approx_no)
        dr = dr_len if (dr_len % 2) == 1 else dr_len + 1
        vec = np.zeros((dr, ))
        middle = int((dr-1)/2)
        for i in range(0, middle+1, 1):
            vec[middle + i] = self.lr_t * (self.discount_range_decay ** i)
            vec[middle - i] = self.lr_t * (self.discount_range_decay ** i)
        # print 'vec: ', vec
        return vec

    def add_value_vector(self, value, reward, action, policy, index, current_value):
        # index += 1

        def compute_indices(ind, mid, div, van, val):
            miai = -1 * min(0, ind - mid - 1)
            maai = min(len(div), van - ind + mid + 1)
            mivi = max(0, ind - mid - 1)
            mavi = min(len(val), ind + mid)
            return miai, maai, mivi, mavi

        middle = int((len(self.disc_increase_vector) - 1)/2)
        pol = policy[0]
        if action == 1:
            pol = 1-policy[1]
        if reward > current_value:
            # compute the index to be strengthened
            new_index = self.compute_value_index(pol + (1 - (reward - current_value)) * self.lr_e)
        else:
            new_index = self.compute_value_index(pol + (reward - current_value) * self.lr_e)

        val_ret = np.array(value)
        min_add_index, max_add_index, min_val_index, max_val_index = compute_indices(new_index,
                                                                                     middle,
                                                                                     self.disc_increase_vector,
                                                                                     self.value_approx_no,
                                                                                     value)

        val_ret[min_val_index:max_val_index] += self.disc_increase_vector[min_add_index:max_add_index]
        return val_ret # / val_ret.__abs__().max()

    def compute_value_function(self, reward_history):
        estimator = decaying_average_estimator
        return estimator(reward_history, self.averaging_window, self.amw, decaying_rate=.9)

    def compute_value_index(self, value):
        try:
            return (np.abs(self.value_approx_range-np.array(value)[:, 0])).argmin()
        except (TypeError, IndexError):
            return(np.abs(self.value_approx_range-value)).argmin()

    def update(self, record):
        constant_player = False
        constant_player_policy = [.0, 1.0]
        # Retrieve Necessary Data
        policy = np.array(record.temp_storage['Policy'][-1])
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_grad = self.temp_storage['Policy Gradient (dPi)']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
        # tmp_rew = np.array(record.perm_storage['Reward'])
        tmp_val = self.temp_storage['Value Function']
        # tmp_act = self.temp_storage['Action']

        comp_val_hist = np.array(record.perm_storage['Value Function'])

        # Initialize Storage
        temp_data = {}
        learning_rate = np.zeros((2,)).tolist()
        # policy_gradient = np.ones((2,))
        policy_gradient = np.zeros((2,))#np.ones((2,))*.1 if tmp_pol_grad[0].any() == 0 else tmp_pol_grad[-1]
        value = np.zeros((2,)).tolist()
        performance = np.zeros((2,)).tolist()
        action = [0, 0]
        reward = [0., 0.]
        mean_val = [0., 0.]
        updated_policy = list(policy)
        iteration = len(record.perm_storage['Policy'])

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # select the policy greedily from each policy's value function
        value_function = np.array(self.temp_storage['Value Function'][-1])
        index_played = [0, 0]
        # start by playing the game:
        for player in range(self.domain.players):
            # playing the game
            # select the best policy greedily, with exploration
            if np.random.random() > self.exploration_rate:  # and iteration > self.exploration_trials:
                # max_val_indices = value_function[player] == value_function.max()
                max_val_indices = np.argwhere(value_function[player] == np.amax(value_function[player]))
                index_played[player] = np.random.choice(max_val_indices[:, 0])
            else:
                index_played[player] = np.random.randint(0, len(value_function[player]))
            updated_policy[player][0] = self.value_approx_range[index_played[player]]
            updated_policy[player][1] = 1-updated_policy[player][0]
            # print 'chose policy ', index_played[player], 'for player', player, 'with a value of', value_function[player][index_played[player]], 'resulting in the policy', updated_policy[player][0]

        # are we in the testing case for stationary play?
        if constant_player:
            updated_policy[0] = constant_player_policy
            index_played[0] = self.compute_value_index(updated_policy[0][0])

        # actually play according to the chosen policy
        for player in range(self.domain.players):
            action[player] = self.domain.action(updated_policy[player])

        # compute the reward and the resulting value function:
        for player in range(self.domain.players):
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]

        # value = self.domain.compute_value_function(policy)
        # print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        update = iteration > self.exploration_trials
        if update and config.debug_output_level:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        for player in range(self.domain.players):
            # TODO: only do updates every nth iteration?
            if update:
                # decide on the strategy:
                # do we have enough data? Are we winning?
                trew = list(tmp_rew[player])
                trew.append(reward[player])
                value_function[player] = self.add_value_vector(value_function[player],
                                                               reward[player],
                                                               action[player],
                                                               policy[player],
                                                               index_played[player],
                                                               self.compute_value_function(trew))
                # print value_function[self.compute_value_index(updated_policy[player][0])]
                if config.debug_output_level:
                    print '-> player', player
                    print '   - temp policies last round:  %.2f' % tmp_pol[-1][player][0]
                    print '   - temp value last round:     %.2f' % tmp_val[-1][player]
                    print '   - the learning rate:         %.2f' % learning_rate[player]
                    print '   - the policy gradient:       %.2f' % policy_gradient[player]
                    print '   - the resulting policy:      %.2f' % updated_policy[player][0]

                    # print 'value ', value_function[1][index_played[1]]
                # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # Store Data
        temp_data['Policy'] = updated_policy
        # temp_data['Policy Estimates'] = self.estimate_policy(self.temp_storage['Action']).tolist()
        temp_data['Value Function'] = value_function.tolist()
        temp_data['Performance'] = performance
        temp_data['True Value Function'] = self.domain.compute_value_function(updated_policy)
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        temp_data['Reward'] = reward
        temp_data['Action'] = action
        self.book_keeping(temp_data)

        return self.temp_storage