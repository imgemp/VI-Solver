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
        self.MinStep = min_step
        self.MaxStep = max_step

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.storage_size * [start]
        self.temp_storage['Value Function'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy Gradient (dPi)'] = self.storage_size * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.storage_size * [[options.init.step], [options.init.step]]
        self.temp_storage['Projections'] = self.storage_size * [0]
        self.temp_storage['Reward'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Value Variance'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Action'] = np.zeros((self.storage_size, 2)).tolist()
        self.temp_storage['Policy Variance'] = np.zeros((self.storage_size, 2)).tolist()

        return self.temp_storage

    def update(self, record):
        # Retrieve Necessary Data
        policy = record.temp_storage['Policy'][-1]
        learn_min = self.MinStep
        learn_max = self.MaxStep
        tmp_pol = self.temp_storage['Policy']
        tmp_pol_lr = self.temp_storage['Policy Learning Rate']
        tmp_rew = self.temp_storage['Reward']
        tmp_val = self.temp_storage['Value Function']
        tmp_act = self.temp_storage['Action']
        tmp_var = self.temp_storage['Value Variance']
        tmp_pol_var = self.temp_storage['Policy Variance']



        # Initialize Storage
        temp_data = {}

        # playing the game
        action = [0, 0]
        for i in range(2):
            action[i] = self.domain.action(policy[i])
        # computing the reward
        reward = [0, 0]
        for i in range(2):
            reward[i] = self.reward[i][action[0]][action[1]]

        # compute the value of the current strategy
        # value = self.domain.compute_value_function(policy) # the analytical solution
        value = np.mean(record.temp_storage['Reward'][-10:], axis=0)

        # computing the variance of the value function
        # val_var = np.var(np.vstack((record.temp_storage['Value Function'][-10:], value[None])))
        val_var = (np.vstack((record.temp_storage['Value Function'][-10:], value[None])))
        val_var = np.var(np.multiply(val_var, .5*(np.array(tmp_rew[-11:])+1)), axis=0)
        # computing the variance of the policy function
        pol = np.array([])
        pol_var = np.var(np.vstack((record.temp_storage['Value Function'][-10:], np.array(policy)[:, 0][None])), axis=0)

        # is the variance increasing?
        var_inc = [False, False]
        for i in range(2):
            # if val_var[i] > np.mean(np.array(tmp_var), axis=0)[i]:
            if val_var[i] > tmp_var[-1][i]:
                var_inc[i] = True

        # compute the average reward expected this far
        exp_reward = np.mean(record.perm_storage['Value Function'][-5000:], axis=0)

        # decide on the learning rate
        learning_rate = [0, 0]
        for i in range(2):
            # learning_rate[i] = learn_min if value[i] > exp_reward[i] and var_inc[i] else learn_max
            # learning_rate[i] = np.mean(np.array(tmp_var), axis=0)[i]/val_var[i]*learn_min \
            #     if val_var[i]>1e-4 and val_var[i]<1e3 and value[i]>exp_reward[i] else learn_min
            learning_rate[i] = np.mean(np.array(tmp_pol_var), axis=0)[i]/pol_var[i]*learn_min \
                if pol_var[i]>1e-4 and pol_var[i]<1e3 and value[i]>exp_reward[i] else learn_min
        # print learning_rate

        # compute the estimate for the opponent's policy
        policy_estimate = np.array([[0., 0.], [0., 0.]])
        for i in range(2):
            tmp_est = np.mean(tmp_act[i][-20:])
            policy_estimate[i] = np.array([tmp_est, 1-tmp_est])

        # estimate the gradient
        policy_gradient = [ policy[1]*self.domain.u-     (self.reward[0][1][1]-self.reward[0][0][1]),
                            policy[0]*self.domain.uprime-(self.reward[1][1][1]-self.reward[1][1][0])]

        # perform update on the policies and project them into the feasible space
        updated_policy = [  self.Proj.p(policy[0], learning_rate[0], policy_gradient[0]),  # Player 1
                            self.Proj.p(policy[1], learning_rate[1], policy_gradient[1])]  # Player 2

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

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