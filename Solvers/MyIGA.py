__author__ = 'clemens'

import numpy as np
from Projection import *
from Solvers.Solver import Solver
from Utilities import *


class MyIGA(Solver):
    def __init__(self, domain, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=1e-5, max_step=1e-3):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.storage_size = 50
        self.temp_storage = {}
        self.min_step = min_step
        self.max_step = max_step

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.storage_size * [[options.init.step], [options.init.step]]
        # self.temp_storage['Projections'] = self.storage_size * [0]
        self.temp_storage['Reward'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Variance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Action'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy Variance'] = np.zeros((self.storage_size, 2)).tolist()

        return self.temp_storage

    def am_i_winning(self, value_history):
        # deciding on the position in the range of the value history:
        min_value = min(value_history)
        max_value = max(value_history)
        range_value = max_value - min_value
        mean_value = np.mean(value_history)
        last_value = value_history[-1]
        # print 'value history: ', value_history

        # the winning part is a float in the range of winningness
        range_value = range_value if range_value != 0. else 1.
        return (last_value-min_value)/range_value, last_value > mean_value

    def am_i_at_a_fixpoint(self, variance_history):
        # deciding on the position in the range of the value history:
        min_value = min(variance_history)
        max_value = max(variance_history)
        range_value = max_value - min_value
        mean_value = np.mean(variance_history)
        last_value = variance_history[-1]

        # the winning part is a float in the range of winningness
        range_value = range_value if abs(range_value) > 1e-2 else 1.
        return (last_value-mean_value)/range_value, last_value > mean_value

    def compute_value_function(self, reward_history, window_size=10):
        return np.mean(reward_history[-1*window_size:])

    def update(self, record):
        exploration_trials = 80
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        learn_min = self.min_step
        learn_max = self.max_step
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_grad = self.temp_storage['Policy Gradient (dPi)']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
        tmp_val = self.temp_storage['Value Function']
        tmp_act = self.temp_storage['Action']
        tmp_var = self.temp_storage['Value Variance']
        tmp_pol_var = self.temp_storage['Policy Variance']

        # Initialize Storage
        temp_data = {}
        learning_rate = np.zeros((2,))
        policy_gradient = np.zeros((2,))
        value = np.zeros((2,))
        action = [0, 0]
        reward = [0, 0]
        updated_policy = list(policy)
        val_var = tmp_var[-1]
        pol_var = tmp_var[-1]

        print '-~-~-~-~-~-~ new iteration ~-~-~-~-~-~-~-~-~-~-~-~-'

        # start by playing the game:
        for player in range(self.domain.players):
            # playing the game
            action[player] = self.domain.action(policy[player])

        # compute the reward and the resulting value function:
        for player in range(self.domain.players):
            # computing the reward
            reward[player] = self.reward[player][action[0]][action[1]]
            # compute the value of the current strategy
            value[player] = self.compute_value_function(np.hstack((np.array(tmp_rew)[:, player], [reward[player]])))

        # perform the update on the policies:
        for player in range(self.domain.players):
            # print '- player', player
            # TODO: only do updates every nth iteration?
            # update = (len(record.perm_storage['Policy']) % 10) == 0
            update = True
            if update:

                # decide on the strategy:
                # do we have enough data? Are we winning?
                performance, winning = self.am_i_winning(np.array(tmp_val)[:, player] + [value[player]])
                print 'performance: ', performance, ' winning: ', winning
                if not winning or len(record.perm_storage['Policy']) < exploration_trials:
                    # play randomly
                    learning_rate[player] = (1 - performance) * (-.5 + np.random.random())
                    policy_gradient[player] = 1.
                    updated_policy[player] = self.Proj.p(policy[player], learning_rate[player], policy_gradient[player])
                    # print ' * RANDOM PLAY'
                    # print '   original policy: ', policy[player]
                    # print '   learning rate  : ', learning_rate[player]
                    # print '   updated policy : ', updated_policy[player]
                else:
                    # play more sophisticated, for we are winning :-)
                    # reinforce the direction if we are doing better than last time, flip it otherwise
                    policy_gradient[player] = (tmp_pol[-1][player][0]-tmp_pol[-2][player][0]) * \
                                              (tmp_val[-1][player]-tmp_val[-2][player])

                    # compute the learning rate
                    # the idea is that we are already winning. So we do not want to make jumps, generally.
                    # however, we want the least change if we are at an equilibrium - i.e. an area where a change in
                    # policy will only result in a small change in the value function.
                    _tmp_val = (np.array(tmp_val)[-10:, player]).tolist()
                    _tmp_val.append(value[player])
                    val_var[player] = _tmp_val
                    relative_var, _ = self.am_i_at_a_fixpoint(val_var[player])

                    # update the learning rate based on the variance
                    learning_rate[player] = 2 ** (-2. + 4. * relative_var) * .5 * (self.max_step - self.min_step)
                    print(learning_rate[player])

                    # compute the new policy
                    updated_policy[player] = self.Proj.p(policy[player], learning_rate[player], policy_gradient[player])

        # Record Projections
        # temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        temp_data['Reward'] = reward
        temp_data['Action'] = action
        temp_data['Value Variance'] = val_var
        temp_data['Policy Variance'] = pol_var
        self.book_keeping(temp_data)

        return self.temp_storage