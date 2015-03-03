__author__ = 'clemens'

import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *
import config


class MyIGA(Solver):
    def __init__(self, domain,
                 P=IdentityProjection(),
                 delta0=1e-2,
                 growth_limit=2,
                 min_step=1e-5,
                 max_step=1e-3,
                 averaging_window=10,
                 exploration_trials=50,
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

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_grad = self.temp_storage['Policy Gradient (dPi)']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
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
        policy_gradient = np.zeros((2,))#np.ones((2,))*.1 if tmp_pol_grad[0].any() == 0 else tmp_pol_grad[-1]
        value = np.zeros((2,)).tolist()
        performance = np.zeros((2,)).tolist()
        action = [0, 0]
        reward = [0., 0.]
        mean_val = [0., 0.]
        updated_policy = list(policy)
        val_var = list(tmp_var[-1])
        pol_var = list(tmp_pol_var[-1])
        winning = [False, False]
        fixpoint = [False, False]
        iteration = len(record.perm_storage['Policy'])
        lavi = self.compute_last_update_index(self.averaging_window)

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
        update = iteration > self.exploration_trials and (iteration % self.averaging_window) == 0
        # update = False
        if update and config.show_debug_output:
            print '-~-~-~-~-~-~ new iteration (', iteration, ') ~-~-~-~-~-~-~-~-~-~-~-~-'
        # perform the update on the policies:
        for player in range(self.domain.players):
            # TODO: only do updates every nth iteration?
            # update = (len(record.perm_storage['Policy']) % 10) == 0
            # update = iteration > self.exploration_trials and (iteration % self.averaging_window) == 0
            if update:
                # decide on the strategy:
                # do we have enough data? Are we winning?
                performance[player], \
                    mean_val[player], \
                    winning[player] = self.am_i_winning(np.hstack((comp_val_hist[:, player],
                                                                   value[player])),
                                                        tmp_winning[:, player])
                # computing the policy gradient and the learning rate
                if not winning[player]:
                    # play randomly
                    # policy_gradient[player] = 2**(-1*self.compute_update_direction(np.array(tmp_pol_grad)[:, player],
                    #                                                                np.array(tmp_val)[:, player]))
                    policy_gradient[player] = 2**(-1*self.compute_update_direction(np.array(tmp_pol_grad)[:, player],
                                                                                   np.array(tmp_val)[:, player]))
                    learning_rate[player] = (-1 * mean_val[player] + performance[player]) + (np.random.random())

                else:
                    # play more sophisticated, for we are winning :-)
                    # reinforce the direction if we are doing better than last time, flip it otherwise
                    policy_gradient[player] = self.compute_update_direction(np.array(tmp_pol_grad)[:, player],
                                                                            np.array(tmp_val)[:, player])

                    # compute the learning rate
                    # the idea is that we are already winning. So we do not want to make jumps, generally.
                    # however, we want the least change if we are at an equilibrium - i.e. an area where a change in
                    # policy will only result in a small change in the value function.
                    relative_var, fixpoint[player] = self.am_i_at_a_fixpoint(val_var[player], last_stable[player])

                    # update the learning rate based on the variance
                    learning_rate[player] = 2 ** (-2. + 4. * relative_var) * .5 * (self.max_step - self.min_step)

                # -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~- -~*#*~-
                # compute the new policy
                updated_policy[player] = self.Proj.p(policy[player],
                                                     learning_rate[player],
                                                     policy_gradient[player])
                if config.show_debug_output:
                    print '-> player', player
                    print '   - is this a random play?    ', not winning[player]
                    print '   - temp policies last round:  %.2f' % tmp_pol[-1][player][0]
                    print '          - round before that:  %.2f' % tmp_pol[lavi][player][0]
                    print '   - temp value last round:     %.2f' % tmp_val[-1][player]
                    print '          - round before that:  %.2f' % tmp_val[lavi][player]
                    print '   - the winningness:           %.2f' % performance[player]
                    print '   - the learning rate:         %.2f' % learning_rate[player]
                    print '   - the policy gradient:       %.2f' % policy_gradient[player]
                    print '   - the resulting policy:      %.2f' % updated_policy[player][0]
                    print '   - the resulting polgrad:     %.2f' % (updated_policy[player]-tmp_pol[lavi][player])[0]

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