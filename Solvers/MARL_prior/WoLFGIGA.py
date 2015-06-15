__author__ = 'clemens'

from VISolver.Projection import *
from VISolver.Solvers.Solver import Solver
from VISolver.Utilities import *
from VISolver.Options import Reporting


class WoLFGIGA(Solver):
    def __init__(self, domain, P=IdentityProjection(), delta0=1e-2, growth_limit=2, min_step=1e-5, max_step=1e-3):

        self.reward = [domain.r_reward, domain.c_reward]
        self.domain = domain
        self.nash_eq_value = domain.compute_nash_eq_value()
        self.Proj = P
        self.StorageSize = 3
        self.temp_storage = {}
        self.storage_size = 250
        self.MinStep = min_step
        self.MaxStep = max_step

    def init_temp_storage(self, start, domain, options):

        self.temp_storage['Policy'] = self.StorageSize * [start]
        self.temp_storage['x Policy'] = self.StorageSize * [start]
        self.temp_storage['z Policy'] = self.StorageSize * [start]
        self.temp_storage['Value Function'] = self.StorageSize * [np.zeros(domain.r_reward.shape)]
        self.temp_storage['Policy Gradient (dPi)'] = self.StorageSize * [np.zeros(domain.b.shape)]
        self.temp_storage['Policy Learning Rate'] = self.StorageSize * [[options.init.step], [options.init.step]]
        self.temp_storage['Projections'] = self.StorageSize * [0]

        return self.temp_storage

    def reporting_options(self):
        return Reporting(requests=[self.domain.ne_l2error,
                                   'Policy',
                                   'Policy Gradient (dPi)',
                                   'Policy Learning Rate',
                                   # 'Reward',
                                   'Value Function',
                                   # 'True Value Function',
                                   # 'Action',
                                   # 'Policy Estimates',
                                   ])

    def update(self, record):
        # Retrieve Necessary Data
        x_policy = record.temp_storage['x Policy'][-1]
        z_policy = record.temp_storage['z Policy'][-1]
        learn_min = self.MinStep
        learn_max = self.MaxStep

        # Initialize Storage
        temp_data = {}

        # compute the value of the current strategy
        value = self.domain.compute_value_function(x_policy)

        # decide on the learning rate
        # learning_rate = [0, 0]
        # learning_rate[0] = learn_min if value[0] > self.nash_eq_value[0] else learn_max
        # learning_rate[1] = learn_min if value[1] > self.nash_eq_value[1] else learn_max
        learning_rate = 1e-4

        # estimate the gradient
        policy_gradient = [ x_policy[1]*self.domain.u-     (self.reward[0][1][1]-self.reward[0][0][1]),
                            x_policy[0]*self.domain.uprime-(self.reward[1][1][1]-self.reward[1][1][0])]

        # perform update on the policies and project them into the feasible space
        updated_x_policy = np.array([  self.Proj.p(x_policy[0], learning_rate, policy_gradient[0]),  # Player 1
                                       self.Proj.p(x_policy[1], learning_rate, policy_gradient[1])])  # Player 2
        updated_z_policy = np.array([  self.Proj.p(z_policy[0], learning_rate/3., policy_gradient[0]),  # Player 1
                                       self.Proj.p(z_policy[1], learning_rate/3., policy_gradient[1])])  # Player 2

        # computing the second update learning rate
        lr_difference = min(1, np.linalg.norm(updated_z_policy - z_policy) /
                               np.linalg.norm(updated_z_policy - updated_x_policy))

        # computing the final update for the x policy
        updated_policy = updated_x_policy + lr_difference * (updated_z_policy - updated_x_policy)

        # Record Projections
        temp_data['Projections'] = 1 + self.temp_storage['Projections'][-1]

        # Store Data
        temp_data['Policy'] = updated_policy
        temp_data['x Policy'] = updated_x_policy
        temp_data['z Policy'] = updated_z_policy
        temp_data['Value Function'] = value
        temp_data['Policy Gradient (dPi)'] = policy_gradient
        temp_data['Policy Learning Rate'] = learning_rate
        self.book_keeping(temp_data)

        return self.temp_storage