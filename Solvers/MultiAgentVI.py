__author__ = 'clemens'

import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *
from Options import Reporting
import config
from pykalman import KalmanFilter as kf


class MultiAgentVI(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 delta0=1e-2,
                 growth_limit=2,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
                 averaging_meta_window=10,
                 learning_rate_t=.0001,
                 learning_rate_e=.0001,
                 discount_factor=.95,
                 value_approx_steps=50):

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
        self.value_approx_no = value_approx_steps
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
        self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2, self.value_approx_no, 2))
        self.temp_storage['Value Function'][-1, 0, :, 0] = self.value_approx_range
        self.temp_storage['Value Function'][-1, 1, :, 0] = self.value_approx_range
        self.temp_storage['Value Function'] = self.temp_storage['Value Function'].tolist()
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

    def compute_value_function(self, reward_history, window_size):
        return np.mean(reward_history[-1*window_size:])

    def compute_value_index(self, value):
        try:
            return (np.abs(self.value_approx_range-np.array(value)[:, 0])).argmin()
        except (TypeError, IndexError):
            return(np.abs(self.value_approx_range-value)).argmin()

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_grad = self.temp_storage['Policy Gradient (dPi)']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
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
            # select the best policy greedily
            index_played[player] = value_function[player, :, 1].argmax()
            updated_policy[player][0] = value_function[player, index_played[player], 0]
            updated_policy[player][1] = 1-updated_policy[player][0]
            # play according to that policy
            action[player] = self.domain.action(updated_policy[player])

        # compute the reward and the resulting value function:
        for player in range(self.domain.players):
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]
            # compute the value of the current strategy
            value[player] = self.compute_value_function(np.hstack((np.array(tmp_rew)[:, player], [reward[player]])),
                                                        self.averaging_window)
        # value = self.domain.compute_value_function(policy)
        # print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
        # update = iteration > self.exploration_trials and (iteration % self.averaging_window) == 0
        update = True
        # update = iteration > self.exploration_trials
        if update and config.debug_output_level:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        for player in range(self.domain.players):
            # TODO: only do updates every nth iteration?
            if update:
                # decide on the strategy:
                # do we have enough data? Are we winning?
                value_function[self.compute_value_index(updated_policy[player][0])] += .1*reward[player]
                if config.debug_output_level:
                    print '-> player', player
                    print '   - temp policies last round:  %.2f' % tmp_pol[-1][player][0]
                    print '   - temp value last round:     %.2f' % tmp_val[-1][player]
                    print '   - the learning rate:         %.2f' % learning_rate[player]
                    print '   - the policy gradient:       %.2f' % policy_gradient[player]
                    print '   - the resulting policy:      %.2f' % updated_policy[player][0]

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