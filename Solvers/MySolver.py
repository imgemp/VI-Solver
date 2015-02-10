__author__ = 'clemens'

from VISolver.Projection import *
from Solver import Solver
from VISolver.Utilities import *


class MySolver(Solver):
    def __init__(self, domain, look_back=10, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=-1e10, max_step=1e10):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.Proj = P
        self.StorageSize = look_back
        self.temp_storage = {}
        self.Delta0 = delta0
        self.GrowthLimit = growth_limit
        self.MinStep = min_step
        self.MaxStep = max_step
        self.Mod = 1e6  # (100)
        self.Agg = 1  # (10)
        self.agent_i = 0

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.StorageSize * [start]
        self.temp_storage['Value Function'] = 50 * self.StorageSize * [np.zeros(domain.r_reward.shape)]
        self.temp_storage['Policy Gradient (dPi)'] = self.StorageSize * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.StorageSize * [options.init.step]
        self.temp_storage['Projections'] = self.StorageSize * [0]
        self.temp_storage['Action taken'] = self.StorageSize * [[.5, .5]]

        return self.temp_storage

    def rl_risk_learner(self, reward_history):
        pass

    def update(self, record):
        # -~-~ Retrieve Necessary Data ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
        policy = record.temp_storage['Policy'][-1]
        policy_grad = record.temp_storage['Policy Gradient (dPi)'][-1]
        pol_learning_rate = record.temp_storage['Policy Learning Rate'][-1]

        # Initialize Storage
        temp_data = {}

        # -~-~ play the game with the current strategy ~-~-~-~-~-~-~-~-~-~-~-~-~-~-
        action_taken = [self.domain.action(policy[0]), self.domain.action(policy[1])]
        # observe the rewards
        reward = [self.reward[0][action_taken[0]][action_taken[1]], self.reward[1][action_taken[0]][action_taken[1]]]
        policy_estimates = np.mean(self.temp_storage['Action taken'], axis=0) # works as a surprisingly good approx

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
        # estimate the gradient
        policy_gradient = [ policy_estimates[1]*self.domain.u-     (self.reward[0][1][1]-self.reward[0][0][1]),
                            policy_estimates[0]*self.domain.uprime-(self.reward[1][1][1]-self.reward[1][1][0])]

        # perform update on the policies and project them into the feasible space
        updated_policy = [  self.Proj.p(policy[0], pol_learning_rate, policy_gradient[0]),  # Player 1
                            self.Proj.p(policy[1], pol_learning_rate, policy_gradient[1])]  # Player 2

        # compute the value of the current strategy
        complete_policy_approx_matrix = np.hstack((np.array(policy_estimates)[None].T, 1-np.array(policy_estimates)[None].T))
        value = self.domain.compute_value_function(policy, complete_policy_approx_matrix)

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['Value Function'] = value
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = pol_learning_rate
        temp_data['Action taken'] = action_taken
        self.book_keeping(temp_data)

        return self.temp_storage